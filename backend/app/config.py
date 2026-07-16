"""Environment-backed configuration and core package bootstrapping."""

from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MAFAS_ROOT = REPOSITORY_ROOT / "mafas"
DEFAULT_WATCHLIST = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "JPM"]


def ensure_core_import_path() -> Path:
    """Make the existing top-level ``agents``, ``rag`` and ``data`` packages importable."""
    path = str(MAFAS_ROOT)
    if path not in sys.path:
        sys.path.insert(0, path)
    return MAFAS_ROOT


def load_playbook_defaults() -> list[dict[str, Any]]:
    """Load the pure playbook module without importing the heavy ``agents`` package."""
    module_name = "_mafas_dashboard_strategy_playbooks"
    module = sys.modules.get(module_name)
    if module is None:
        source = MAFAS_ROOT / "agents" / "strategy_playbooks.py"
        spec = importlib.util.spec_from_file_location(module_name, source)
        if spec is None or spec.loader is None:
            raise RuntimeError("Could not load strategy playbook defaults.")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

    return [
        {
            "key": item.key,
            "name": item.name,
            "description": item.description,
            "directional": item.directional,
            "tags": list(item.tags),
        }
        for item in module.PLAYBOOKS.values()
    ]


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_origins() -> tuple[str, ...]:
    raw = os.getenv("CORS_ORIGINS", "http://localhost:3000")
    return tuple(item.strip() for item in raw.split(",") if item.strip())


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime settings; secret values are kept internal and never returned by APIs."""

    database_url: str = "mysql+pymysql://mafas:mafas@mysql:3306/mafas"
    api_prefix: str = "/api/v1"
    cors_origins: tuple[str, ...] = ("http://localhost:3000",)
    cors_allow_credentials: bool = True
    job_max_workers: int = 1
    job_max_queue: int = 16
    sse_poll_seconds: float = 0.25
    health_timeout_seconds: float = 1.5
    qdrant_url: str = "http://qdrant:6333"
    qdrant_collection: str = "financial_docs"
    ollama_url: str = "http://ollama:11434"
    fred_configured: bool = False
    twelve_data_configured: bool = False
    edgar_configured: bool = False
    news_configured: bool = True
    sql_echo: bool = False

    @classmethod
    def from_env(cls) -> "Settings":
        """Load settings from process environment variables."""
        return cls(
            database_url=os.getenv(
                "DATABASE_URL", "mysql+pymysql://mafas:mafas@mysql:3306/mafas"
            ),
            api_prefix=os.getenv("API_PREFIX", "/api/v1"),
            cors_origins=_env_origins(),
            cors_allow_credentials=_env_bool("CORS_ALLOW_CREDENTIALS", True),
            job_max_workers=max(1, _env_int("JOB_MAX_WORKERS", 1)),
            job_max_queue=max(0, _env_int("JOB_MAX_QUEUE", 16)),
            sse_poll_seconds=max(0.05, _env_float("SSE_POLL_SECONDS", 0.25)),
            health_timeout_seconds=max(
                0.05, _env_float("HEALTH_TIMEOUT_SECONDS", 1.5)
            ),
            qdrant_url=os.getenv("QDRANT_URL", "http://qdrant:6333").rstrip("/"),
            qdrant_collection=os.getenv("QDRANT_COLLECTION", "financial_docs"),
            ollama_url=os.getenv("OLLAMA_URL", "http://ollama:11434").rstrip("/"),
            fred_configured=bool(os.getenv("FRED_API_KEY", "").strip()),
            twelve_data_configured=bool(
                os.getenv("TWELVE_DATA_API_KEY", "").strip()
            ),
            edgar_configured=bool(
                os.getenv("SEC_USER_AGENT", os.getenv("EDGAR_USER_AGENT", "")).strip()
            ),
            news_configured=_env_bool("NEWS_ENABLED", True),
            sql_echo=_env_bool("SQL_ECHO", False),
        )
