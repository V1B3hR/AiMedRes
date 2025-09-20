"""
DuetMind Adaptive - Configuration Constants

This module contains configuration constants to improve maintainability
and eliminate magic numbers/strings throughout the codebase.
"""

# Performance Monitoring Constants
DEFAULT_MONITORING_INTERVAL = 5  # seconds
MAX_REQUEST_TIMESTAMPS = 1000
MAX_MEMORY_HISTORY = 100
MAX_CPU_HISTORY = 100
THROUGHPUT_WINDOW_SECONDS = 60

# Memory and Performance Thresholds
MEMORY_ALERT_THRESHOLD_MB = 512
CPU_ALERT_THRESHOLD_PERCENT = 80
RESPONSE_TIME_ALERT_THRESHOLD = 5.0  # seconds
ERROR_RATE_ALERT_THRESHOLD = 0.1  # 10%

# Network and API Configuration
DEFAULT_API_PORT = 8080
DEFAULT_RATE_LIMIT = 100  # requests per minute
DEFAULT_MAX_CONCURRENT_REQUESTS = 50
DEFAULT_REQUEST_TIMEOUT = 30  # seconds

# Cache Configuration
DEFAULT_CACHE_TTL = 3600  # 1 hour in seconds
MAX_CACHE_SIZE = 1000  # number of items
CACHE_CLEANUP_INTERVAL = 300  # 5 minutes

# Database Configuration
DEFAULT_DB_POOL_SIZE = 10
DEFAULT_DB_TIMEOUT = 30
DEFAULT_BATCH_SIZE = 100

# Docker and Deployment Constants
DOCKER_PYTHON_VERSION = "3.11-slim"
NGINX_PORT = 80
GRAFANA_PORT = 3000
PROMETHEUS_PORT = 9090

# Logging Configuration
DEFAULT_LOG_LEVEL = "INFO"
MAX_LOG_FILE_SIZE = 10 * 1024 * 1024  # 10MB
MAX_LOG_BACKUP_COUNT = 5

# Security Constants
JWT_EXPIRY_HOURS = 24
MAX_LOGIN_ATTEMPTS = 5
SESSION_TIMEOUT_MINUTES = 30

# Medical AI Constants
DEFAULT_CONFIDENCE_THRESHOLD = 0.75
MAX_PREDICTION_AGE_HOURS = 24
DEFAULT_RISK_THRESHOLD = 0.8

# Enterprise Configuration
ENTERPRISE_MIN_WORKERS = 4
ENTERPRISE_MAX_WORKERS = 16
ENTERPRISE_SCALING_FACTOR = 2.0