"""FastAPI entry point for the MAFAS dashboard backend."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from backend.app.config import (
    DEFAULT_WATCHLIST,
    Settings,
    load_playbook_defaults,
)
from backend.app.db import Conversation, Database, Job, JobEvent, Message, utc_now
from backend.app.health import HealthChecker
from backend.app.jobs import (
    TERMINAL_STATUSES,
    JobQueueFull,
    JobRunner,
    JobService,
)
from backend.app.runners import build_default_runners
from backend.app.schemas import (
    AnalystRunRequest,
    ConversationCreate,
    ConversationDetail,
    ConversationRead,
    ConversationUpdate,
    CorpusRefreshRequest,
    CorpusResetRequest,
    DefaultsResponse,
    DemoRunRequest,
    EvaluationRunRequest,
    ExecutionRunRequest,
    HealthResponse,
    JobAccepted,
    JobEventRead,
    JobKind,
    JobRead,
    JobStatus,
    JobSubmitRequest,
    MessageCreate,
    MessageRead,
    PipelineRunRequest,
    PlaybookDTO,
    RiskRunRequest,
    StrategyRunRequest,
    SystemSummary,
)


def _message_read(message: Message) -> MessageRead:
    return MessageRead(
        id=message.id,
        conversation_id=message.conversation_id,
        role=message.role,
        content=message.content,
        job_id=message.job_id,
        metadata=message.metadata_json,
        created_at=message.created_at,
    )


def _conversation_read(conversation: Conversation) -> ConversationRead:
    return ConversationRead(
        id=conversation.id,
        title=conversation.title,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
    )


def _conversation_detail(conversation: Conversation) -> ConversationDetail:
    return ConversationDetail(
        **_conversation_read(conversation).model_dump(),
        messages=[_message_read(message) for message in conversation.messages],
    )


def _job_read(job: Job) -> JobRead:
    return JobRead(
        id=job.id,
        kind=job.kind,  # type: ignore[arg-type]
        status=job.status,  # type: ignore[arg-type]
        payload=job.payload,
        metadata=job.job_metadata,
        result=job.result,
        error=job.error,
        conversation_id=job.conversation_id,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
    )


def _job_event_read(event: JobEvent) -> JobEventRead:
    return JobEventRead(
        sequence=event.sequence,
        event_type=event.event_type,
        message=event.message,
        data=event.data,
        created_at=event.created_at,
    )


def _markdown_export(job: JobRead) -> str:
    """Render a durable job record as a readable Markdown document."""
    lines = [
        f"# MAFAS {job.kind.replace('_', ' ').title()} Job",
        "",
        f"- **Job ID:** `{job.id}`",
        f"- **Status:** {job.status}",
        f"- **Created:** {job.created_at.isoformat()}",
        f"- **Demo mode:** {str(bool(job.metadata.get('demo_mode'))).lower()}",
    ]
    if job.error:
        lines.extend(["", "## Error", "", job.error])
    lines.extend(
        [
            "",
            "## Result",
            "",
            "```json",
            json.dumps(job.result, indent=2, ensure_ascii=False, default=str),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def create_app(
    settings: Settings | None = None,
    runner_overrides: dict[str, JobRunner] | None = None,
) -> FastAPI:
    """Application factory supporting SQLite and mocked runners in tests."""
    runtime = settings or Settings.from_env()
    database = Database(runtime.database_url, echo=runtime.sql_echo)
    runners = build_default_runners()
    if runner_overrides:
        runners.update(runner_overrides)

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        database.create_all()
        service = JobService(
            database.sessions,
            runners,
            max_workers=runtime.job_max_workers,
            max_queue=runtime.job_max_queue,
        )
        service.mark_orphaned_jobs_failed()
        application.state.job_service = service
        try:
            yield
        finally:
            service.shutdown()
            database.dispose()

    app = FastAPI(
        title="MAFAS Dashboard API",
        version="1.0.0",
        description="Durable local API for the Multi-Agent Financial AI System.",
        lifespan=lifespan,
    )
    app.state.settings = runtime
    app.state.database = database

    allow_credentials = runtime.cors_allow_credentials and "*" not in runtime.cors_origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(runtime.cors_origins),
        allow_credentials=allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    router = APIRouter(prefix=runtime.api_prefix)

    def get_session() -> Iterator[Session]:
        with database.sessions() as session:
            yield session

    def get_job_service(request: Request) -> JobService:
        service: JobService | None = getattr(request.app.state, "job_service", None)
        if service is None:
            raise HTTPException(status_code=503, detail="Job service is not started.")
        return service

    def require_conversation(
        conversation_id: str | None, session: Session
    ) -> None:
        if conversation_id and session.get(Conversation, conversation_id) is None:
            raise HTTPException(status_code=404, detail="Conversation not found.")

    def conversation_context(conversation_id: str, session: Session) -> str:
        """Build a small, durable context window from the latest completed runs."""
        recent = list(
            session.scalars(
                select(Job)
                .where(
                    Job.conversation_id == conversation_id,
                    Job.status == "succeeded",
                    Job.result.is_not(None),
                )
                .order_by(Job.completed_at.desc())
                .limit(3)
            )
        )
        blocks: list[str] = []
        for job in reversed(recent):
            result = job.result if isinstance(job.result, dict) else {}
            briefing = result.get("briefing", result)
            risk = result.get("risk", {})
            strategy = result.get("strategy", {})
            lines = [
                f"Earlier user query: {job.payload.get('query', job.kind)}",
                f"Decision: {result.get('decision', 'not applicable')}",
            ]
            if isinstance(briefing, dict) and briefing.get("summary"):
                lines.append(f"Analyst summary: {briefing['summary']}")
            if isinstance(risk, dict) and risk.get("narrative"):
                lines.append(f"Risk summary: {risk['narrative']}")
            if isinstance(strategy, dict) and strategy.get("narrative"):
                lines.append(f"Strategy summary: {strategy['narrative']}")
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)[-8_000:]

    def submit_job(
        kind: str,
        payload: dict[str, Any],
        service: JobService,
        session: Session,
    ) -> JobAccepted:
        conversation_id = payload.get("conversation_id")
        require_conversation(conversation_id, session)
        payload = dict(payload)
        if conversation_id and not payload.get("context"):
            context = conversation_context(conversation_id, session)
            if context:
                payload["context"] = context
        try:
            job = service.submit(
                kind,
                payload,
                conversation_id=conversation_id,
            )
        except JobQueueFull as exc:
            raise HTTPException(
                status_code=503,
                detail={
                    "message": "Background job queue is full.",
                    "failed_job_id": exc.job_id,
                },
            ) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return JobAccepted(id=job.id, status="queued")

    @router.get("/health", response_model=HealthResponse, tags=["system"])
    def health() -> dict[str, Any]:
        """Return dependency readiness without exposing configuration secrets."""
        return HealthChecker(runtime, database.sessions).snapshot()

    @router.get("/system/summary", response_model=SystemSummary, tags=["system"])
    def system_summary(session: Session = Depends(get_session)) -> SystemSummary:
        """Return dashboard counts plus the latest dependency state."""
        snapshot = HealthChecker(runtime, database.sessions).snapshot()
        qdrant = snapshot["services"]["qdrant"]
        return SystemSummary(
            conversations=session.scalar(select(func.count(Conversation.id))) or 0,
            messages=session.scalar(select(func.count(Message.id))) or 0,
            jobs=session.scalar(select(func.count(Job.id))) or 0,
            active_jobs=session.scalar(
                select(func.count(Job.id)).where(
                    Job.status.in_(("queued", "running"))
                )
            )
            or 0,
            corpus_count=qdrant.get("count"),
            health_status=snapshot["status"],
        )

    @router.get("/config/defaults", response_model=DefaultsResponse, tags=["config"])
    def config_defaults() -> DefaultsResponse:
        """Expose safe watchlist and playbook defaults from the existing core."""
        playbooks = [PlaybookDTO.model_validate(item) for item in load_playbook_defaults()]
        return DefaultsResponse(
            watchlist=list(DEFAULT_WATCHLIST),
            playbooks=playbooks,
            job_kinds=[
                "pipeline",
                "analyst",
                "risk",
                "strategy",
                "execution",
                "corpus_refresh",
                "corpus_reset",
                "demo",
                "evaluation",
            ],
        )

    @router.post(
        "/jobs", response_model=JobAccepted, status_code=202, tags=["jobs"]
    )
    def create_job(
        body: JobSubmitRequest,
        service: JobService = Depends(get_job_service),
        session: Session = Depends(get_session),
    ) -> JobAccepted:
        """Submit any supported job kind using a generic payload."""
        return submit_job(body.kind, body.payload, service, session)

    @router.get("/jobs", response_model=list[JobRead], tags=["jobs"])
    def list_jobs(
        kind: JobKind | None = None,
        status: JobStatus | None = None,
        limit: int = Query(50, ge=1, le=200),
        offset: int = Query(0, ge=0),
        session: Session = Depends(get_session),
    ) -> list[JobRead]:
        """List newest jobs with optional kind and status filters."""
        statement = select(Job)
        if kind:
            statement = statement.where(Job.kind == kind)
        if status:
            statement = statement.where(Job.status == status)
        jobs = session.scalars(
            statement.order_by(Job.created_at.desc()).offset(offset).limit(limit)
        )
        return [_job_read(job) for job in jobs]

    @router.get("/jobs/{job_id}", response_model=JobRead, tags=["jobs"])
    def get_job(job_id: str, session: Session = Depends(get_session)) -> JobRead:
        """Retrieve one durable job."""
        job = session.get(Job, job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found.")
        return _job_read(job)

    @router.delete("/jobs/{job_id}", status_code=204, tags=["jobs"])
    def delete_job(
        job_id: str, session: Session = Depends(get_session)
    ) -> Response:
        """Delete a terminal job and its events."""
        job = session.get(Job, job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found.")
        if job.status not in TERMINAL_STATUSES:
            raise HTTPException(
                status_code=409, detail="Only terminal jobs can be deleted."
            )
        session.delete(job)
        session.commit()
        return Response(status_code=204)

    @router.get(
        "/jobs/{job_id}/events",
        response_class=StreamingResponse,
        tags=["jobs"],
    )
    async def stream_job_events(job_id: str) -> StreamingResponse:
        """Replay persisted events, then poll until the job reaches a terminal state."""
        with database.sessions() as session:
            if session.get(Job, job_id) is None:
                raise HTTPException(status_code=404, detail="Job not found.")

        async def event_stream() -> AsyncIterator[str]:
            cursor = 0
            idle_polls = 0
            while True:
                with database.sessions() as session:
                    job = session.get(Job, job_id)
                    if job is None:
                        return
                    events = list(
                        session.scalars(
                            select(JobEvent)
                            .where(
                                JobEvent.job_id == job_id,
                                JobEvent.sequence > cursor,
                            )
                            .order_by(JobEvent.sequence)
                        )
                    )
                    status = job.status

                for event in events:
                    cursor = event.sequence
                    body = _job_event_read(event).model_dump(mode="json")
                    yield (
                        f"id: {event.sequence}\n"
                        f"event: {event.event_type}\n"
                        f"data: {json.dumps(body, ensure_ascii=False)}\n\n"
                    )
                if status in TERMINAL_STATUSES:
                    return
                idle_polls += 1
                if idle_polls % 20 == 0:
                    yield ": keep-alive\n\n"
                await asyncio.sleep(runtime.sse_poll_seconds)

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    def exported_job(job_id: str, session: Session) -> JobRead:
        job = session.get(Job, job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found.")
        return _job_read(job)

    def export_response(job: JobRead, export_format: str) -> Response:
        filename = f"mafas-{job.kind}-{job.id}"
        if export_format == "json":
            return JSONResponse(
                job.model_dump(mode="json"),
                headers={
                    "Content-Disposition": f'attachment; filename="{filename}.json"'
                },
            )
        return PlainTextResponse(
            _markdown_export(job),
            media_type="text/markdown",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}.md"'
            },
        )

    @router.get("/jobs/{job_id}/export", tags=["exports"])
    def export_job(
        job_id: str,
        export_format: Literal["json", "markdown"] = Query(alias="format"),
        session: Session = Depends(get_session),
    ) -> Response:
        """Export a job as JSON or Markdown; PDF is handled by frontend printing."""
        return export_response(exported_job(job_id, session), export_format)

    @router.get("/jobs/{job_id}/export/json", tags=["exports"])
    def export_job_json(
        job_id: str, session: Session = Depends(get_session)
    ) -> Response:
        return export_response(exported_job(job_id, session), "json")

    @router.get("/jobs/{job_id}/export/markdown", tags=["exports"])
    def export_job_markdown(
        job_id: str, session: Session = Depends(get_session)
    ) -> Response:
        return export_response(exported_job(job_id, session), "markdown")

    @router.post(
        "/pipeline/run",
        response_model=JobAccepted,
        status_code=202,
        tags=["runs"],
    )
    @router.post(
        "/runs/pipeline",
        response_model=JobAccepted,
        status_code=202,
        include_in_schema=False,
    )
    def run_pipeline(
        body: PipelineRunRequest,
        service: JobService = Depends(get_job_service),
        session: Session = Depends(get_session),
    ) -> JobAccepted:
        return submit_job("pipeline", body.model_dump(mode="json"), service, session)

    @router.post(
        "/agents/analyst/run",
        response_model=JobAccepted,
        status_code=202,
        tags=["runs"],
    )
    @router.post(
        "/runs/analyst",
        response_model=JobAccepted,
        status_code=202,
        include_in_schema=False,
    )
    def run_analyst(
        body: AnalystRunRequest,
        service: JobService = Depends(get_job_service),
        session: Session = Depends(get_session),
    ) -> JobAccepted:
        return submit_job("analyst", body.model_dump(mode="json"), service, session)

    @router.post(
        "/agents/risk/run",
        response_model=JobAccepted,
        status_code=202,
        tags=["runs"],
    )
    @router.post(
        "/runs/risk",
        response_model=JobAccepted,
        status_code=202,
        include_in_schema=False,
    )
    def run_risk(
        body: RiskRunRequest,
        service: JobService = Depends(get_job_service),
        session: Session = Depends(get_session),
    ) -> JobAccepted:
        return submit_job("risk", body.model_dump(mode="json"), service, session)

    @router.post(
        "/agents/strategy/run",
        response_model=JobAccepted,
        status_code=202,
        tags=["runs"],
    )
    @router.post(
        "/runs/strategy",
        response_model=JobAccepted,
        status_code=202,
        include_in_schema=False,
    )
    def run_strategy(
        body: StrategyRunRequest,
        service: JobService = Depends(get_job_service),
        session: Session = Depends(get_session),
    ) -> JobAccepted:
        return submit_job("strategy", body.model_dump(mode="json"), service, session)

    @router.post(
        "/agents/execution/run",
        response_model=JobAccepted,
        status_code=202,
        tags=["runs"],
    )
    @router.post(
        "/runs/execution",
        response_model=JobAccepted,
        status_code=202,
        include_in_schema=False,
    )
    def run_execution(
        body: ExecutionRunRequest,
        service: JobService = Depends(get_job_service),
        session: Session = Depends(get_session),
    ) -> JobAccepted:
        return submit_job("execution", body.model_dump(mode="json"), service, session)

    @router.post(
        "/corpus/refresh",
        response_model=JobAccepted,
        status_code=202,
        tags=["corpus"],
    )
    def refresh_corpus(
        body: CorpusRefreshRequest,
        service: JobService = Depends(get_job_service),
        session: Session = Depends(get_session),
    ) -> JobAccepted:
        return submit_job(
            "corpus_refresh", body.model_dump(mode="json"), service, session
        )

    @router.post(
        "/corpus/reset",
        response_model=JobAccepted,
        status_code=202,
        tags=["corpus"],
    )
    def reset_corpus(
        body: CorpusResetRequest,
        service: JobService = Depends(get_job_service),
        session: Session = Depends(get_session),
    ) -> JobAccepted:
        payload = body.model_dump(mode="json")
        payload.pop("confirmation", None)
        return submit_job("corpus_reset", payload, service, session)

    @router.post(
        "/demo/run",
        response_model=JobAccepted,
        status_code=202,
        tags=["runs"],
    )
    def run_demo(
        body: DemoRunRequest,
        service: JobService = Depends(get_job_service),
        session: Session = Depends(get_session),
    ) -> JobAccepted:
        return submit_job("demo", body.model_dump(mode="json"), service, session)

    @router.post(
        "/evaluation/run",
        response_model=JobAccepted,
        status_code=202,
        tags=["evaluation"],
    )
    @router.post(
        "/runs/evaluation",
        response_model=JobAccepted,
        status_code=202,
        include_in_schema=False,
    )
    def run_evaluation(
        body: EvaluationRunRequest,
        service: JobService = Depends(get_job_service),
        session: Session = Depends(get_session),
    ) -> JobAccepted:
        return submit_job("evaluation", body.model_dump(mode="json"), service, session)

    @router.post(
        "/conversations",
        response_model=ConversationRead,
        status_code=201,
        tags=["conversations"],
    )
    def create_conversation(
        body: ConversationCreate, session: Session = Depends(get_session)
    ) -> ConversationRead:
        conversation = Conversation(title=body.title.strip())
        session.add(conversation)
        session.commit()
        session.refresh(conversation)
        return _conversation_read(conversation)

    @router.get(
        "/conversations",
        response_model=list[ConversationRead],
        tags=["conversations"],
    )
    def list_conversations(
        limit: int = Query(50, ge=1, le=200),
        offset: int = Query(0, ge=0),
        session: Session = Depends(get_session),
    ) -> list[ConversationRead]:
        conversations = session.scalars(
            select(Conversation)
            .order_by(Conversation.updated_at.desc())
            .offset(offset)
            .limit(limit)
        )
        return [_conversation_read(item) for item in conversations]

    @router.get(
        "/conversations/{conversation_id}",
        response_model=ConversationDetail,
        tags=["conversations"],
    )
    def get_conversation(
        conversation_id: str, session: Session = Depends(get_session)
    ) -> ConversationDetail:
        conversation = session.scalar(
            select(Conversation)
            .options(selectinload(Conversation.messages))
            .where(Conversation.id == conversation_id)
        )
        if conversation is None:
            raise HTTPException(status_code=404, detail="Conversation not found.")
        return _conversation_detail(conversation)

    @router.patch(
        "/conversations/{conversation_id}",
        response_model=ConversationRead,
        tags=["conversations"],
    )
    def update_conversation(
        conversation_id: str,
        body: ConversationUpdate,
        session: Session = Depends(get_session),
    ) -> ConversationRead:
        conversation = session.get(Conversation, conversation_id)
        if conversation is None:
            raise HTTPException(status_code=404, detail="Conversation not found.")
        conversation.title = body.title.strip()
        conversation.updated_at = utc_now()
        session.commit()
        return _conversation_read(conversation)

    @router.delete(
        "/conversations/{conversation_id}",
        status_code=204,
        tags=["conversations"],
    )
    def delete_conversation(
        conversation_id: str, session: Session = Depends(get_session)
    ) -> Response:
        conversation = session.get(Conversation, conversation_id)
        if conversation is None:
            raise HTTPException(status_code=404, detail="Conversation not found.")
        session.delete(conversation)
        session.commit()
        return Response(status_code=204)

    @router.get(
        "/conversations/{conversation_id}/messages",
        response_model=list[MessageRead],
        tags=["conversations"],
    )
    def list_messages(
        conversation_id: str, session: Session = Depends(get_session)
    ) -> list[MessageRead]:
        require_conversation(conversation_id, session)
        messages = session.scalars(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at)
        )
        return [_message_read(message) for message in messages]

    @router.post(
        "/conversations/{conversation_id}/messages",
        response_model=MessageRead,
        status_code=201,
        tags=["conversations"],
    )
    def create_message(
        conversation_id: str,
        body: MessageCreate,
        session: Session = Depends(get_session),
    ) -> MessageRead:
        conversation = session.get(Conversation, conversation_id)
        if conversation is None:
            raise HTTPException(status_code=404, detail="Conversation not found.")
        message = Message(
            conversation_id=conversation_id,
            role=body.role,
            content=body.content,
            job_id=body.job_id,
            metadata_json=body.metadata,
        )
        conversation.updated_at = utc_now()
        session.add(message)
        session.commit()
        session.refresh(message)
        return _message_read(message)

    app.include_router(router)
    return app


app = create_app()
