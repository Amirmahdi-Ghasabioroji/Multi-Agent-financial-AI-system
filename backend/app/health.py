"""Non-secret-bearing health checks for backend dependencies."""

from __future__ import annotations

from typing import Any

import httpx
from sqlalchemy import text
from sqlalchemy.orm import Session, sessionmaker

from backend.app.config import Settings


class HealthChecker:
    """Probe runtime services with short timeouts and safe summaries."""

    def __init__(
        self, settings: Settings, sessions: sessionmaker[Session]
    ) -> None:
        self.settings = settings
        self.sessions = sessions

    def _database(self) -> dict[str, Any]:
        try:
            with self.sessions() as session:
                session.execute(text("SELECT 1"))
            return {"status": "ok"}
        except Exception as exc:  # noqa: BLE001 - health must degrade, not raise
            return {
                "status": "unavailable",
                "detail": f"Database probe failed ({type(exc).__name__}).",
            }

    def _qdrant(self) -> dict[str, Any]:
        timeout = self.settings.health_timeout_seconds
        try:
            ready = httpx.get(
                f"{self.settings.qdrant_url}/readyz",
                timeout=timeout,
            )
            ready.raise_for_status()
            check: dict[str, Any] = {"status": "ok"}
            try:
                collection = httpx.get(
                    (
                        f"{self.settings.qdrant_url}/collections/"
                        f"{self.settings.qdrant_collection}"
                    ),
                    timeout=timeout,
                )
                if collection.status_code == 200:
                    body = collection.json()
                    count = body.get("result", {}).get("points_count")
                    if isinstance(count, int):
                        check["count"] = count
                elif collection.status_code == 404:
                    check["detail"] = "Configured collection does not exist yet."
            except Exception:
                check["detail"] = "Ready; corpus count unavailable."
            return check
        except Exception as exc:  # noqa: BLE001
            return {
                "status": "unavailable",
                "detail": f"Qdrant probe failed ({type(exc).__name__}).",
            }

    def _ollama(self) -> dict[str, Any]:
        try:
            response = httpx.get(
                f"{self.settings.ollama_url}/api/tags",
                timeout=self.settings.health_timeout_seconds,
            )
            response.raise_for_status()
            models = response.json().get("models", [])
            count = len(models) if isinstance(models, list) else 0
            return {"status": "ok", "count": count}
        except Exception as exc:  # noqa: BLE001
            return {
                "status": "unavailable",
                "detail": f"Ollama probe failed ({type(exc).__name__}).",
            }

    @staticmethod
    def _configured(enabled: bool, label: str) -> dict[str, Any]:
        if enabled:
            return {"status": "ok", "detail": f"{label} is configured."}
        return {
            "status": "not_configured",
            "detail": f"{label} is not configured.",
        }

    def snapshot(self) -> dict[str, Any]:
        """Return API, database, model, vector, and external-source checks."""
        services = {
            "api": {"status": "ok"},
            "database": self._database(),
            "qdrant": self._qdrant(),
            "ollama": self._ollama(),
            "fred": self._configured(self.settings.fred_configured, "FRED"),
            "twelve_data": self._configured(
                self.settings.twelve_data_configured, "Twelve Data"
            ),
            "sec_edgar": self._configured(
                self.settings.edgar_configured, "SEC EDGAR"
            ),
            "news": self._configured(self.settings.news_configured, "News feeds"),
        }
        required = ("api", "database", "qdrant", "ollama")
        status = (
            "ok"
            if all(services[name]["status"] == "ok" for name in required)
            else "degraded"
        )
        return {"status": status, "services": services}
