"""
DuetMind Adaptive - Configuration Constants

This module contains configuration constants to improve maintainability
and eliminate magic numbers/strings throughout the codebase.
"""

# Performance Monitoring Configuration
DEFAULT_MONITORING_INTERVAL_SECONDS = 5
MAX_REQUEST_TIMESTAMPS_STORED = 1000
MAX_MEMORY_HISTORY_ENTRIES = 100
MAX_CPU_HISTORY_ENTRIES = 100
THROUGHPUT_CALCULATION_WINDOW_SECONDS = 60

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

# Input Validation Constants
MAX_TASK_LENGTH = 2000  # Maximum characters for task input
MAX_QUERY_LENGTH = 500  # Maximum characters for search query
MAX_KNOWLEDGE_SEARCH_RESULTS = 50  # Maximum search results returned
DEFAULT_KNOWLEDGE_SEARCH_LIMIT = 10  # Default number of search results

# API Processing Constants
DEFAULT_MEMORY_LIMIT_MB = 500  # Default memory limit for caching
VECTOR_SEARCH_BATCH_THRESHOLD = 100  # Threshold for parallel processing
DEFAULT_NETWORK_SIZE = 50  # Default neural network size
NODE_BATCH_PROCESSING_THRESHOLD = 10  # Minimum nodes for batch processing
DEFAULT_NODE_ENERGY = 10.0  # Default energy value for nodes

# Drift Detection Constants  
DRIFT_DETECTION_KS_THRESHOLD = 0.05  # P-value threshold for KS test
DRIFT_DETECTION_WASSERSTEIN_THRESHOLD = 0.1  # Normalized Wasserstein threshold
DRIFT_DETECTION_TV_THRESHOLD = 0.2  # Total Variation distance threshold
DRIFT_DETECTION_JS_THRESHOLD = 0.1  # Jensen-Shannon divergence threshold
DRIFT_DETECTION_CHI2_THRESHOLD = 0.05  # P-value threshold for chi-square test