"""End-to-end API tests with SQLite and mocked synchronous runners."""

from __future__ import annotations

import time
from typing import Any

from fastapi.testclient import TestClient

from backend.app.config import Settings
from backend.app.db import Database, Job
from backend.app.jobs import JobRunner
from backend.app.main import create_app


def wait_for_terminal(
    client: TestClient, job_id: str, timeout: float = 3.0
) -> dict[str, Any]:
    """Poll the durable job endpoint until a worker records its outcome."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(f"/api/v1/jobs/{job_id}")
        assert response.status_code == 200
        job = response.json()
        if job["status"] in {"succeeded", "failed"}:
            return job
        time.sleep(0.01)
    raise AssertionError(f"Job {job_id} did not finish before timeout")


def test_health_and_system_summary(client: TestClient) -> None:
    health = client.get("/api/v1/health")
    assert health.status_code == 200
    payload = health.json()
    assert payload["services"]["api"]["status"] == "ok"
    assert payload["services"]["database"]["status"] == "ok"
    assert payload["status"] == "degraded"
    assert "url" not in str(payload).lower()

    summary = client.get("/api/v1/system/summary")
    assert summary.status_code == 200
    assert summary.json()["jobs"] == 0
    assert summary.json()["conversations"] == 0

    defaults = client.get("/api/v1/config/defaults")
    assert defaults.status_code == 200
    assert "AAPL" in defaults.json()["watchlist"]
    assert any(
        item["key"] == "trend_following"
        for item in defaults.json()["playbooks"]
    )


def test_conversation_crud_and_messages(client: TestClient) -> None:
    created = client.post(
        "/api/v1/conversations", json={"title": "Rate outlook"}
    )
    assert created.status_code == 201
    conversation_id = created.json()["id"]

    message = client.post(
        f"/api/v1/conversations/{conversation_id}/messages",
        json={
            "role": "user",
            "content": "What changed in the latest meeting?",
            "metadata": {"source": "dashboard"},
        },
    )
    assert message.status_code == 201
    assert message.json()["metadata"] == {"source": "dashboard"}

    detail = client.get(f"/api/v1/conversations/{conversation_id}")
    assert detail.status_code == 200
    assert detail.json()["messages"][0]["content"].startswith("What changed")

    renamed = client.patch(
        f"/api/v1/conversations/{conversation_id}",
        json={"title": "Updated rate outlook"},
    )
    assert renamed.status_code == 200
    assert renamed.json()["title"] == "Updated rate outlook"
    assert len(client.get("/api/v1/conversations").json()) == 1

    deleted = client.delete(f"/api/v1/conversations/{conversation_id}")
    assert deleted.status_code == 204
    assert client.get(f"/api/v1/conversations/{conversation_id}").status_code == 404


def test_conversation_runs_persist_messages_and_build_bounded_context(
    client: TestClient,
) -> None:
    conversation = client.post(
        "/api/v1/conversations", json={"title": "Inflation follow-up"}
    ).json()
    conversation_id = conversation["id"]

    first = client.post(
        "/api/v1/agents/analyst/run",
        json={
            "query": "What is the inflation outlook?",
            "conversation_id": conversation_id,
        },
    )
    first_job = wait_for_terminal(client, first.json()["id"])
    assert first_job["result"]["received_context"] is None

    second = client.post(
        "/api/v1/agents/analyst/run",
        json={
            "query": "What changed?",
            "conversation_id": conversation_id,
        },
    )
    second_job = wait_for_terminal(client, second.json()["id"])
    assert "What is the inflation outlook?" in second_job["result"]["received_context"]
    assert "Mocked analyst result." in second_job["result"]["received_context"]

    messages = client.get(
        f"/api/v1/conversations/{conversation_id}/messages"
    ).json()
    assert [message["role"] for message in messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert all(message["job_id"] for message in messages)


def test_job_persists_across_app_restart(
    settings: Settings, runner_overrides: dict[str, JobRunner]
) -> None:
    first_app = create_app(settings, runner_overrides)
    with TestClient(first_app) as first:
        accepted = first.post(
            "/api/v1/agents/analyst/run",
            json={"query": "Summarise inflation risks"},
        )
        assert accepted.status_code == 202
        job_id = accepted.json()["id"]
        completed = wait_for_terminal(first, job_id)
        assert completed["status"] == "succeeded"
        assert completed["result"]["summary"] == "Mocked analyst result."

    second_app = create_app(settings, runner_overrides)
    with TestClient(second_app) as second:
        persisted = second.get(f"/api/v1/jobs/{job_id}")
        assert persisted.status_code == 200
        assert persisted.json()["status"] == "succeeded"
        assert len(second.get("/api/v1/jobs").json()) == 1


def test_startup_marks_orphaned_running_job_failed(
    settings: Settings, runner_overrides: dict[str, JobRunner]
) -> None:
    database = Database(settings.database_url)
    database.create_all()
    with database.sessions() as session:
        session.add(
            Job(
                id="00000000-0000-0000-0000-000000000001",
                kind="demo",
                status="running",
                payload={},
                job_metadata={"demo_mode": True},
            )
        )
        session.commit()
    database.dispose()

    app = create_app(settings, runner_overrides)
    with TestClient(app) as client:
        job = client.get(
            "/api/v1/jobs/00000000-0000-0000-0000-000000000001"
        ).json()
        assert job["status"] == "failed"
        assert "restarted" in job["error"].lower()
        events = client.get(
            "/api/v1/jobs/00000000-0000-0000-0000-000000000001/events"
        )
        assert '"orphaned": true' in events.text


def test_sse_exports_demo_and_delete(client: TestClient) -> None:
    accepted = client.post(
        "/api/v1/demo/run",
        json={
            "query": "Demo rate outlook",
            "tickers": ["AAPL", "MSFT"],
            "demo_mode": True,
        },
    )
    assert accepted.status_code == 202
    job_id = accepted.json()["id"]
    completed = wait_for_terminal(client, job_id)
    assert completed["status"] == "succeeded"
    assert completed["metadata"]["demo_mode"] is True
    assert completed["result"]["demo_mode"] is True
    assert completed["result"]["briefing"]["confidence"] == 0.82

    events = client.get(f"/api/v1/jobs/{job_id}/events")
    assert events.status_code == 200
    assert events.headers["content-type"].startswith("text/event-stream")
    assert "event: queued" in events.text
    assert "event: succeeded" in events.text

    json_export = client.get(f"/api/v1/jobs/{job_id}/export/json")
    assert json_export.status_code == 200
    assert json_export.json()["id"] == job_id
    assert json_export.headers["content-disposition"].endswith('.json"')

    markdown = client.get(f"/api/v1/jobs/{job_id}/export?format=markdown")
    assert markdown.status_code == 200
    assert markdown.headers["content-type"].startswith("text/markdown")
    assert "# MAFAS Demo Job" in markdown.text
    assert '"demo_mode": true' in markdown.text

    deleted = client.delete(f"/api/v1/jobs/{job_id}")
    assert deleted.status_code == 204
    assert client.get(f"/api/v1/jobs/{job_id}").status_code == 404


def test_evaluation_run_returns_structured_report(client: TestClient) -> None:
    accepted = client.post(
        "/api/v1/evaluation/run",
        json={"suites": ["simulation"]},
    )
    assert accepted.status_code == 202
    completed = wait_for_terminal(client, accepted.json()["id"])
    assert completed["status"] == "succeeded"
    assert completed["kind"] == "evaluation"
    assert completed["result"]["suites"][0]["suite"] == "simulation"


def test_corpus_reset_requires_exact_confirmation(client: TestClient) -> None:
    rejected = client.post(
        "/api/v1/corpus/reset",
        json={"confirmation": "reset financial docs"},
    )
    assert rejected.status_code == 422

    accepted = client.post(
        "/api/v1/corpus/reset",
        json={"confirmation": "RESET FINANCIAL DOCS"},
    )
    assert accepted.status_code == 202
    completed = wait_for_terminal(client, accepted.json()["id"])
    assert completed["status"] == "succeeded"
    assert completed["result"] == {"operation": "reset", "completed": True}
    assert "confirmation" not in completed["payload"]
