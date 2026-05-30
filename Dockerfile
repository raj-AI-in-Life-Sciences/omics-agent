# ── Stage 1: base ─────────────────────────────────────────────────────────────
FROM python:3.11-slim AS base

WORKDIR /app

# System deps for faiss-cpu and psycopg2
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Stage 2: api ──────────────────────────────────────────────────────────────
FROM base AS api

COPY omics_agent/ omics_agent/
COPY generate_results.py .
COPY pyproject.toml .

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

CMD ["uvicorn", "omics_agent.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
