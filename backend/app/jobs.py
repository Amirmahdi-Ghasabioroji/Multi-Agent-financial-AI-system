"""Durable, bounded background execution for synchronous MAFAS work."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from datetime import date, datetime
import re
from threading import BoundedSemaphore
from typing import Any, Callable, Protocol

from loguru import logger
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from backend.app.db import Conversation, Job, JobEvent, Message, new_id, utc_now

TERMINAL_STATUSES = frozenset({"succeeded", "failed"})


class EventEmitter(Protocol):
    """Callback accepted by all job runners."""

    def __call__(
        self, event_type: str, message: str, data: dict[str, Any] | None = None
    ) -> None: ...


JobRunner = Callable[[dict[str, Any], EventEmitter], Any]


class JobQueueFull(RuntimeError):
    """Raised when all worker and queue slots are occupied."""

    def __init__(self, job_id: str) -> None:
        super().__init__("The background job queue is full.")
        self.job_id = job_id


class BoundedExecutor:
    """ThreadPoolExecutor guarded by a non-blocking capacity semaphore."""

    def __init__(self, max_workers: int, max_queue: int) -> None:
        self._pool = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="mafas-job"
        )
        self._slots = BoundedSemaphore(max_workers + max_queue)

    def submit(self, fn: Callable[..., Any], *args: Any) -> Future[Any]:
        """Submit work only when a worker or bounded queue slot is available."""
        if not self._slots.acquire(blocking=False):
            raise RuntimeError("executor capacity exhausted")
        try:
            future = self._pool.submit(fn, *args)
        except BaseException:
            self._slots.release()
            raise
        future.add_done_callback(lambda _: self._slots.release())
        return future

    def shutdown(self) -> None:
        """Stop accepting work and cancel tasks that have not started."""
        self._pool.shutdown(wait=False, cancel_futures=True)


def json_safe(value: Any) -> Any:
    """Convert common model and scalar types into JSON-compatible values."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "model_dump"):
        return json_safe(value.model_dump(mode="json"))
    return str(value)


def safe_error(exc: Exception) -> str:
    """Return a useful exception summary with common credential forms redacted."""
    message = str(exc)
    message = re.sub(
        r"(?i)(api[_-]?key|token|secret|password)(\s*[=:]\s*)[^&\s,;]+",
        r"\1\2[REDACTED]",
        message,
    )
    message = re.sub(r"(://[^:/@\s]+:)[^@\s]+@", r"\1[REDACTED]@", message)
    return f"{type(exc).__name__}: {message}"[:4000]


def result_summary(kind: str, result: dict[str, Any] | list[Any]) -> str:
    """Create a compact assistant message while the full JSON stays on the job."""
    if not isinstance(result, dict):
        return f"{kind.replace('_', ' ').title()} completed."
    briefing = result.get("briefing", result)
    summary = briefing.get("summary") if isinstance(briefing, dict) else None
    decision = result.get("decision")
    reason = result.get("no_trade_reason")
    if summary:
        prefix = f"Decision: {str(decision).replace('_', ' ').upper()}. " if decision else ""
        suffix = f" Reason: {reason}" if reason else ""
        return f"{prefix}{summary}{suffix}"[:2_000]
    if result.get("narrative"):
        return str(result["narrative"])[:2_000]
    cards = result.get("cards")
    if isinstance(cards, list) and cards:
        verdicts = [
            str(card.get("verdict", ""))
            for card in cards
            if isinstance(card, dict) and card.get("verdict")
        ]
        if verdicts:
            return " ".join(verdicts)[:2_000]
    return f"{kind.replace('_', ' ').title()} completed successfully."


def append_event(
    session: Session,
    job_id: str,
    event_type: str,
    message: str,
    data: dict[str, Any] | None = None,
) -> JobEvent:
    """Append one monotonically sequenced event within the caller's transaction."""
    latest = session.scalar(
        select(func.max(JobEvent.sequence)).where(JobEvent.job_id == job_id)
    )
    event = JobEvent(
        job_id=job_id,
        sequence=int(latest or 0) + 1,
        event_type=event_type,
        message=message,
        data=json_safe(data or {}),
    )
    session.add(event)
    return event


class JobService:
    """Persist jobs and execute every job kind through one bounded thread pool."""

    def __init__(
        self,
        sessions: sessionmaker[Session],
        runners: dict[str, JobRunner],
        *,
        max_workers: int = 1,
        max_queue: int = 16,
    ) -> None:
        self._sessions = sessions
        self._runners = runners
        self._executor = BoundedExecutor(max_workers, max_queue)

    def mark_orphaned_jobs_failed(self) -> int:
        """Fail non-terminal jobs left behind by an earlier backend process."""
        with self._sessions() as session:
            jobs = list(
                session.scalars(
                    select(Job).where(Job.status.in_(("queued", "running")))
                )
            )
            now = utc_now()
            for job in jobs:
                previous = job.status
                job.status = "failed"
                job.completed_at = now
                job.error = (
                    "Backend restarted while this job was running."
                    if previous == "running"
                    else "Queued job was not resumed after backend restart."
                )
                append_event(
                    session,
                    job.id,
                    "failed",
                    job.error,
                    {"orphaned": True, "previous_status": previous},
                )
            session.commit()
        if jobs:
            logger.warning("Marked {} orphaned jobs as failed", len(jobs))
        return len(jobs)

    def submit(
        self,
        kind: str,
        payload: dict[str, Any],
        *,
        conversation_id: str | None = None,
    ) -> Job:
        """Persist a queued job and schedule its runner."""
        if kind not in self._runners:
            raise ValueError(f"Unsupported job kind: {kind}")

        job_id = new_id()
        metadata = {
            "demo_mode": bool(payload.get("demo_mode") or kind == "demo"),
            "executor": "thread_pool",
        }
        with self._sessions() as session:
            job = Job(
                id=job_id,
                kind=kind,
                status="queued",
                payload=json_safe(payload),
                job_metadata=metadata,
                conversation_id=conversation_id,
            )
            session.add(job)
            append_event(session, job_id, "queued", f"{kind} job queued")
            if conversation_id:
                session.add(
                    Message(
                        conversation_id=conversation_id,
                        role="user",
                        content=str(
                            payload.get("query")
                            or payload.get("ticker")
                            or f"Run {kind.replace('_', ' ')} agent"
                        ),
                        job_id=job_id,
                        metadata_json={
                            "kind": kind,
                            "demo_mode": metadata["demo_mode"],
                        },
                    )
                )
                conversation = session.get(Conversation, conversation_id)
                if conversation is not None:
                    conversation.updated_at = utc_now()
            session.commit()

        try:
            self._executor.submit(self._execute, job_id)
        except RuntimeError as exc:
            with self._sessions() as session:
                failed = session.get(Job, job_id)
                if failed is not None:
                    failed.status = "failed"
                    failed.error = "Background job queue is full."
                    failed.completed_at = utc_now()
                    append_event(
                        session,
                        job_id,
                        "failed",
                        failed.error,
                        {"queue_full": True},
                    )
                    session.commit()
            raise JobQueueFull(job_id) from exc

        with self._sessions() as session:
            persisted = session.get(Job, job_id)
            if persisted is None:  # defensive: the just-created row must exist
                raise RuntimeError("Job persistence failed.")
            session.expunge(persisted)
            return persisted

    def _execute(self, job_id: str) -> None:
        """Run one synchronous core operation and persist its outcome."""
        with self._sessions() as session:
            job = session.get(Job, job_id)
            if job is None:
                return
            job.status = "running"
            job.started_at = utc_now()
            append_event(session, job_id, "running", f"{job.kind} job started")
            session.commit()
            kind = job.kind
            payload = dict(job.payload)

        def emit(
            event_type: str,
            message: str,
            data: dict[str, Any] | None = None,
        ) -> None:
            with self._sessions() as event_session:
                if event_session.get(Job, job_id) is None:
                    return
                append_event(event_session, job_id, event_type, message, data)
                event_session.commit()

        try:
            result = json_safe(self._runners[kind](payload, emit))
            if not isinstance(result, (dict, list)):
                result = {"value": result}
            with self._sessions() as session:
                job = session.get(Job, job_id)
                if job is None:
                    return
                job.status = "succeeded"
                job.result = result
                job.completed_at = utc_now()
                job.job_metadata = {
                    **job.job_metadata,
                    "demo_mode": bool(
                        payload.get("demo_mode")
                        or kind == "demo"
                        or (isinstance(result, dict) and result.get("demo_mode"))
                    ),
                }
                append_event(
                    session,
                    job_id,
                    "succeeded",
                    f"{kind} job completed",
                )
                if job.conversation_id:
                    session.add(
                        Message(
                            conversation_id=job.conversation_id,
                            role="assistant",
                            content=result_summary(kind, result),
                            job_id=job.id,
                            metadata_json={
                                "kind": kind,
                                "demo_mode": job.job_metadata["demo_mode"],
                            },
                        )
                    )
                    conversation = session.get(Conversation, job.conversation_id)
                    if conversation is not None:
                        conversation.updated_at = utc_now()
                session.commit()
        except Exception as exc:  # noqa: BLE001 - jobs must persist all failures
            logger.exception("{} job {} failed", kind, job_id)
            error = safe_error(exc)
            with self._sessions() as session:
                job = session.get(Job, job_id)
                if job is None:
                    return
                job.status = "failed"
                job.error = error
                job.completed_at = utc_now()
                append_event(
                    session,
                    job_id,
                    "failed",
                    f"{kind} job failed",
                    {"error_type": type(exc).__name__},
                )
                session.commit()

    def shutdown(self) -> None:
        """Release executor resources."""
        self._executor.shutdown()
