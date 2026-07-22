# ---- Builder: compile wheels and install Python dependencies ----
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DEFAULT_TIMEOUT=300 \
    PIP_RETRIES=10

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

COPY mafas/requirements.txt ./mafas/requirements.txt
COPY backend/requirements.txt ./backend/requirements.txt

RUN python -m pip install --upgrade pip setuptools wheel \
    || (echo "ERROR: pip bootstrap failed." >&2 && exit 1)

# PyTorch CPU wheel is ~180MB; network drops show up as IncompleteRead.
RUN set -e; \
    for attempt in 1 2 3 4 5; do \
        echo "Installing PyTorch CPU (attempt ${attempt}/5)..."; \
        if python -m pip install --retries 10 --timeout 300 \
            --index-url https://download.pytorch.org/whl/cpu \
            --extra-index-url https://pypi.org/simple \
            torch; then \
            echo "PyTorch installed successfully."; \
            exit 0; \
        fi; \
        echo "PyTorch download failed (attempt ${attempt}/5); retrying in 20s..."; \
        sleep 20; \
    done; \
    echo "ERROR: PyTorch install failed after 5 attempts." >&2; \
    exit 1

RUN python -m pip install --retries 10 --timeout 300 "scipy>=1.11.0" \
    || (echo "ERROR: scipy install failed." >&2 && exit 1)

RUN python -m pip install --retries 10 --timeout 300 -r mafas/requirements.txt \
    || (echo "ERROR: mafas requirements install failed." >&2 && exit 1)

RUN python -m pip install --retries 10 --timeout 300 \
        fastapi==0.139.1 \
        "uvicorn[standard]==0.51.0" \
        SQLAlchemy==2.0.51 \
        PyMySQL==1.2.0 \
        alembic==1.18.5 \
    || (echo "ERROR: backend API requirements install failed." >&2 && exit 1)

# ---- Runtime: slim image with only runtime libraries ----
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    CACHE_DIR=/app/mafas/data/cache

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libxml2 \
    libxslt1.1 \
    zlib1g \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid 1000 --create-home appuser

COPY mafas ./mafas
COPY backend ./backend

RUN mkdir -p /app/mafas/data/cache \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=8s --retries=12 --start-period=120s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health', timeout=5)" || exit 1

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
