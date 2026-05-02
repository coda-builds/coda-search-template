# ── Build stage ───────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Runtime system dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq5 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY app/ ./app/

# Pre-download the embedding model into a known path so it is baked into the
# image. HF_HOME overrides the default ~/.cache/huggingface location — this
# ensures the same path is used at both build time (root) and runtime (appuser).
ENV HF_HOME=/app/.cache/huggingface

RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); \
print('Model cached.')"

# Non-root user for security — create AFTER the model download so ownership
# can be transferred in the same layer.
RUN useradd --system --no-create-home --uid 1001 appuser \
    && chown -R appuser:appuser /app/.cache
USER appuser

EXPOSE 8000

# Railway injects PORT as an env var; uvicorn reads it here.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --log-level info"]
