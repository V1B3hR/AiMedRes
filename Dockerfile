# AiMedRes – Backend API Dockerfile
# Provides a containerised environment for the AiMedRes research platform.
#
# ⚠️ RESEARCH USE ONLY — NOT FOR CLINICAL DEPLOYMENT without proper institutional validation.
#
# Build:  docker build -t aimedres .
# Run:    docker run -p 8000:8000 -e FLASK_SECRET_KEY=<32+ chars> -e JWT_SECRET=<32+ chars> aimedres
# The image intentionally fails closed without those runtime secrets.
#
# For a full production stack (with PostgreSQL, Redis, Nginx) see
# deployment/technical/docker-compose.yml.

FROM python:3.10-slim

# Bump this only with a security review; it intentionally invalidates the
# certificate-store layer on rebuild after the 2026-10-19 eMudhra compromise.
ARG CA_CERTIFICATES_REFRESH=2026-10-25

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive \
    AIMEDRES_HOST=0.0.0.0 \
    AIMEDRES_PORT=8000 \
    AIMEDRES_ENV=production

# Non-root user (least-privilege)
RUN groupadd -r aimedres && \
    useradd -r -g aimedres -s /bin/false -d /app aimedres

# System dependencies. Rebuild with --pull so the base image and this CA bundle
# are refreshed; Docker services do not inherit Render's native CA store.
RUN test -n "$CA_CERTIFICATES_REFRESH" && apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer cache)
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy project and install editable
COPY --chown=aimedres:aimedres . .
RUN pip install --no-cache-dir -e .

# Runtime directories
RUN mkdir -p /app/data /app/logs /app/outputs /app/models && \
    chown -R aimedres:aimedres /app/data /app/logs /app/outputs /app/models

VOLUME ["/app/data", "/app/logs", "/app/outputs", "/app/models"]

USER aimedres

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:${AIMEDRES_PORT}/health || exit 1

CMD ["aimedres", "serve", "--host", "0.0.0.0", "--port", "8000"]
