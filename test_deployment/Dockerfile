
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    sqlite3 \
    libsqlite3-dev \
    redis-tools \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install CUDA support (optional)
RUN if [ "False" = "True" ]; then \
    apt-get update && \
    apt-get install -y nvidia-cuda-toolkit && \
    rm -rf /var/lib/apt/lists/*; \
fi

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/knowledge_base /app/logs /app/workspace

# Set environment variables
ENV FLASK_APP=app.py
ENV PYTHONPATH=/app
ENV KNOWLEDGE_BASE_PATH=/app/knowledge_base
ENV LOG_LEVEL=INFO

# Expose ports
EXPOSE 8080
EXPOSE 6379

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run application
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "4", "--timeout", "300", "app:app"]
