# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# Advanced Multi-Source RAG — CPU image (add --build-arg TORCH_INDEX for CUDA)
#
#   docker build -t enterprise-rag .
#   docker run --rm -p 8501:8501 --env-file .env -v "$PWD/data:/app/data" enterprise-rag
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    TOKENIZERS_PARALLELISM=false \
    HF_HOME=/app/.cache/huggingface

ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu

WORKDIR /app

# System dependencies: PyMuPDF wheels need libgl/libglib; curl for healthcheck.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install torch first so the CPU wheel is used (avoids the multi-GB CUDA wheel).
RUN pip install --upgrade pip && \
    pip install --index-url "${TORCH_INDEX}" torch

COPY requirements.txt ./
RUN pip install -r requirements.txt

# Copy project (see .dockerignore for exclusions)
COPY . .

# Non-root user, writable data/cache directories
RUN useradd --create-home --uid 1000 appuser && \
    mkdir -p /app/data /app/.cache && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

# Entrypoint: optionally ingest demo data + build indexes, then serve the UI.
CMD ["bash", "-lc", "\
  if [ \"${AUTO_INGEST_SAMPLES:-true}\" = \"true\" ] && [ ! -f data/indexes/manifest.json ]; then \
     python scripts/build_index.py || echo 'index build failed (continuing)'; \
  fi; \
  streamlit run app/streamlit_app.py \
     --server.address=0.0.0.0 \
     --server.port=8501 \
     --server.enableCORS=false \
     --server.enableXsrfProtection=false"]
