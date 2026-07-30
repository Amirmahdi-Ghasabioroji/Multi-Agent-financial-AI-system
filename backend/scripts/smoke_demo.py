from fastapi.testclient import TestClient

from backend.app.config import Settings
from backend.app.main import create_app


def main() -> None:
    settings = Settings(
        database_url="sqlite:///:memory:",
        qdrant_url="http://127.0.0.1:9",
        ollama_url="http://127.0.0.1:9",
        health_timeout_seconds=0.05,
        sse_poll_seconds=0.01,
    )
    app = create_app(settings)
    with TestClient(app) as client:
        accepted = client.post(
            "/api/v1/demo/run",
            json={
                "query": "Demo rate outlook",
                "tickers": ["AAPL"],
                "demo_mode": True,
            },
        )
        assert accepted.status_code == 202, accepted.text
        job_id = accepted.json()["id"]
        import time

        job = {}
        for _ in range(100):
            job = client.get(f"/api/v1/jobs/{job_id}").json()
            if job["status"] in {"succeeded", "failed"}:
                break
            time.sleep(0.05)
        assert job["status"] == "succeeded", job
        assert job["result"]["decision"] == "trade"
        assert job["metadata"]["demo_mode"] is True
        print(
            "demo-smoke-ok",
            job_id,
            job["result"]["briefing"]["confidence"],
        )


if __name__ == "__main__":
    main()
