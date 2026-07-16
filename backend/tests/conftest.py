"""Shared SQLite application fixtures for backend API tests."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from backend.app.config import Settings
from backend.app.jobs import EventEmitter, JobRunner
from backend.app.main import create_app


def analyst_runner(payload: dict[str, Any], emit: EventEmitter) -> dict[str, Any]:
    """Fast deterministic substitute for the synchronous Analyst Agent."""
    emit("progress", "Mock analyst retrieved evidence", {"stage": "analyst"})
    return {
        "query": payload["query"],
        "summary": "Mocked analyst result.",
        "confidence": 0.8,
        "demo_mode": False,
        "received_context": payload.get("context"),
    }


def corpus_reset_runner(
    payload: dict[str, Any], emit: EventEmitter
) -> dict[str, Any]:
    """Fast deterministic substitute for corpus deletion/re-ingestion."""
    emit("progress", "Mock corpus reset", {"stage": "corpus"})
    return {"operation": "reset", "completed": True}


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        database_url=f"sqlite:///{(tmp_path / 'dashboard.db').as_posix()}",
        cors_origins=("http://localhost:3000",),
        job_max_workers=1,
        job_max_queue=4,
        sse_poll_seconds=0.01,
        health_timeout_seconds=0.05,
        qdrant_url="http://127.0.0.1:9",
        ollama_url="http://127.0.0.1:9",
    )


@pytest.fixture
def runner_overrides() -> dict[str, JobRunner]:
    return {
        "analyst": analyst_runner,
        "corpus_reset": corpus_reset_runner,
    }


@pytest.fixture
def client(
    settings: Settings, runner_overrides: dict[str, JobRunner]
) -> Iterator[TestClient]:
    app = create_app(settings, runner_overrides)
    with TestClient(app) as test_client:
        yield test_client
