# AiMedRes Healthcare Deployment Dockerfile
# Production-ready containerized deployment for healthcare environments
# Supports HIPAA/GDPR compliance requirements

FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Create non-root user for security (least privilege principle)
RUN groupadd -r aimedres && \
    useradd -r -g aimedres -s /bin/bash -d /app aimedres

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Build dependencies
    build-essential \
    gcc \
    g++ \
    # Git for version control integration
    git \
    # Utilities
    curl \
    wget \
    ca-certificates \
    # PostgreSQL client for database operations
    postgresql-client \
    # Health monitoring tools
    procps \
    # Cleanup
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set working directory
WORKDIR /app

# Copy requirements files
COPY requirements.txt requirements-ml.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r requirements-ml.txt

# Copy application code
COPY --chown=aimedres:aimedres . .

# Install the application
RUN pip install --no-cache-dir -e .

# Create necessary directories with proper permissions
RUN mkdir -p /app/data /app/logs /app/outputs /app/models /app/backups && \
    mkdir -p /app/data/{imaging,clinical,raw,processed} && \
    mkdir -p /app/logs/{application,audit,security} && \
    mkdir -p /app/outputs/{predictions,reports,visualizations} && \
    chown -R aimedres:aimedres /app

# Create volume mount points for persistent data
VOLUME ["/app/data", "/app/logs", "/app/outputs", "/app/models", "/app/backups"]

# Switch to non-root user
USER aimedres

# Expose ports
# 8000: Main API service
# 8001: Admin/monitoring interface
# 8002: Health check endpoint
EXPOSE 8000 8001 8002

# Health check configuration
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8002/health || exit 1

# Set entrypoint script (will be created separately)
ENTRYPOINT ["/app/deployment/technical/docker-entrypoint.sh"]

# Default command - can be overridden
CMD ["python", "-m", "src.aimedres.main", "--config", "/app/config.yml"]
