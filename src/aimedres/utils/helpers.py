# -*- coding: utf-8 -*-
"""
AimedRes: AI-Powered Medical Research Assistant (v2)
=====================================================

This single-file script is a fully integrated and improved version of the AimedRes
repository. It combines a Flask API, performance monitoring, caching, a knowledge
graph, a medical AI agent, and a suite of robust utility functions into one
cohesive application.

To run this script:
1. Install dependencies: pip install Flask psutil numpy scipy
2. Execute: python <this_file_name>.py
"""
import hashlib
import logging
import sys
import threading
import time
import uuid
from collections import deque
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import psutil
from flask import Flask, g, jsonify, request
from scipy import stats

# ==============================================================================
# 1. CONFIGURATION CONSTANTS
# ==============================================================================


class Config:
    """
    Configuration constants to improve maintainability and eliminate magic
    numbers/strings throughout the codebase.
    """

    # Performance Monitoring
    DEFAULT_MONITORING_INTERVAL_SECONDS: int = 1
    MAX_REQUEST_TIMESTAMPS_STORED: int = 1000
    MAX_MEMORY_HISTORY_ENTRIES: int = 100
    MAX_CPU_HISTORY_ENTRIES: int = 100
    THROUGHPUT_CALCULATION_WINDOW_SECONDS: int = 60

    # Medical-grade Performance Thresholds (milliseconds)
    TARGET_RESPONSE_TIME_MS: float = 100.0
    CRITICAL_RESPONSE_TIME_MS: float = 150.0

    # Resource Alerting Thresholds
    MEMORY_ALERT_THRESHOLD_MB: int = 512
    CPU_ALERT_THRESHOLD_PERCENT: int = 80
    RESPONSE_TIME_ALERT_THRESHOLD_S: float = 0.150  # 150ms in seconds
    ERROR_RATE_ALERT_THRESHOLD: float = 0.1  # 10%

    # Network and API Configuration
    DEFAULT_API_PORT: int = 8080

    # Cache Configuration
    DEFAULT_CACHE_TTL_SECONDS: int = 3600  # 1 hour
    MAX_CACHE_SIZE: int = 1000
    CACHE_CLEANUP_INTERVAL_SECONDS: int = 300  # 5 minutes

    # Logging Configuration
    DEFAULT_LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Input Validation
    MAX_TASK_LENGTH: int = 2000

    # Drift Detection
    DRIFT_DETECTION_KS_THRESHOLD: float = 0.05


# ==============================================================================
# 2. UTILITY FUNCTIONS
# ==============================================================================


class Utils:
    """A collection of static utility functions for the application."""

    @staticmethod
    def generate_request_id() -> str:
        """Generate a unique request ID for tracking."""
        return str(uuid.uuid4())

    @staticmethod
    def generate_error_id() -> str:
        """Generate a unique error ID for error tracking."""
        return str(uuid.uuid4())

    @staticmethod
    def hash_data(data: str) -> str:
        """Generate a SHA256 hash for data integrity and caching."""
        return hashlib.sha256(data.encode()).hexdigest()

    @staticmethod
    def create_audit_event(
        event_type: str, request_id: str, details: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Create a standardized audit event structure."""
        return {
            "event_type": event_type,
            "timestamp": datetime.now().isoformat(),
            "request_id": request_id,
            "event_id": Utils.generate_request_id(),
            "details": details or {},
        }

    @staticmethod
    def format_response_time(response_time_seconds: float) -> str:
        """Format response time in ms or s for display."""
        if response_time_seconds < 1.0:
            return f"{response_time_seconds * 1000:.1f}ms"
        return f"{response_time_seconds:.3f}s"

    @staticmethod
    def calculate_percentage(part: float, total: float) -> float:
        """Safely calculate percentage, avoiding division by zero."""
        if total == 0:
            return 0.0
        return (part / total) * 100

    @staticmethod
    def format_memory_size(size_bytes: int) -> str:
        """Format memory size in a human-readable format (B, KB, MB, GB)."""
        if size_bytes == 0:
            return "0B"
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} PB"

    @staticmethod
    def validate_config_keys(
        config: Dict[str, Any], required_keys: List[str]
    ) -> Tuple[bool, List[str]]:
        """Validate that required configuration keys are present."""
        missing_keys = [key for key in required_keys if key not in config]
        return len(missing_keys) == 0, missing_keys

    @staticmethod
    def safe_json_extract(data: Dict[str, Any], path: str, default: Any = None) -> Any:
        """Safely extract nested JSON values using dot notation."""
        try:
            keys = path.split(".")
            result = data
            for key in keys:
                result = result[key]
            return result
        except (KeyError, TypeError, AttributeError):
            return default


# ==============================================================================
# 3. PERFORMANCE MONITOR
# ==============================================================================


class PerformanceMonitor:
    """Monitors and reports application performance metrics."""

    def __init__(self, interval: int = Config.DEFAULT_MONITORING_INTERVAL_SECONDS):
        self.process = psutil.Process()
        self.monitoring_interval = interval

        self.request_timestamps: Deque[float] = deque(maxlen=Config.MAX_REQUEST_TIMESTAMPS_STORED)
        self.response_times: Deque[float] = deque(maxlen=Config.MAX_REQUEST_TIMESTAMPS_STORED)
        self.error_count: int = 0
        self.success_count: int = 0

        self.memory_history: Deque[float] = deque(maxlen=Config.MAX_MEMORY_HISTORY_ENTRIES)
        self.cpu_history: Deque[float] = deque(maxlen=Config.MAX_CPU_HISTORY_ENTRIES)

        self.is_running = False
        self.monitor_thread = threading.Thread(target=self._monitor_system_resources, daemon=True)

    def start(self):
        if not self.is_running:
            self.is_running = True
            self.monitor_thread.start()
            logging.info("Performance monitor started.")

    def stop(self):
        self.is_running = False
        logging.info("Performance monitor stopping.")

    def _monitor_system_resources(self):
        while self.is_running:
            try:
                self.memory_history.append(self.process.memory_info().rss)  # In bytes
                self.cpu_history.append(self.process.cpu_percent(interval=0.1))
            except psutil.NoSuchProcess:
                logging.warning("Monitoring process not found. Stopping monitor.")
                self.is_running = False
            except Exception as e:
                logging.error(f"Error in performance monitor thread: {e}")
            time.sleep(self.monitoring_interval)

    def record_request_start(self):
        g.start_time = time.perf_counter()

    def record_request_end(self, status_code: int):
        if "start_time" in g:
            response_time = time.perf_counter() - g.start_time
            self.response_times.append(response_time)
            self.request_timestamps.append(time.time())
            if 200 <= status_code < 400:
                self.success_count += 1
            else:
                self.error_count += 1

    def get_metrics(self) -> Dict[str, Any]:
        now = time.time()

        recent_requests = [
            t
            for t in self.request_timestamps
            if now - t <= Config.THROUGHPUT_CALCULATION_WINDOW_SECONDS
        ]
        throughput_rpm = len(recent_requests) * (60 / Config.THROUGHPUT_CALCULATION_WINDOW_SECONDS)

        total_requests = self.success_count + self.error_count
        error_rate_percent = Utils.calculate_percentage(self.error_count, total_requests)

        avg_response_time, p95, p99 = 0, 0, 0
        if self.response_times:
            times_np = np.array(list(self.response_times))
            avg_response_time = np.mean(times_np)
            p95, p99 = np.percentile(times_np, [95, 99])

        avg_mem_bytes = np.mean(list(self.memory_history)) if self.memory_history else 0
        avg_cpu = np.mean(list(self.cpu_history)) if self.cpu_history else 0

        avg_mem_mb = avg_mem_bytes / (1024 * 1024)

        alerts = []
        if avg_mem_mb > Config.MEMORY_ALERT_THRESHOLD_MB:
            alerts.append(f"High Memory Usage: {avg_mem_mb:.2f} MB")
        if avg_cpu > Config.CPU_ALERT_THRESHOLD_PERCENT:
            alerts.append(f"High CPU Usage: {avg_cpu:.2f}%")
        if avg_response_time > Config.RESPONSE_TIME_ALERT_THRESHOLD_S:
            alerts.append(f"High Response Time: {avg_response_time*1000:.2f} ms")
        if error_rate_percent > (Config.ERROR_RATE_ALERT_THRESHOLD * 100) and total_requests > 10:
            alerts.append(f"High Error Rate: {error_rate_percent:.2f}%")

        return {
            "throughput_rpm": round(throughput_rpm, 2),
            "total_requests": total_requests,
            "error_rate_percent": round(error_rate_percent, 2),
            "response_time": {
                "average_s": round(avg_response_time, 4),
                "p95_s": round(p95, 4),
                "p99_s": round(p99, 4),
                "average_formatted": Utils.format_response_time(avg_response_time),
                "p95_formatted": Utils.format_response_time(p95),
            },
            "resource_usage": {
                "cpu_percent_avg": round(avg_cpu, 2),
                "memory_avg_bytes": int(avg_mem_bytes),
                "memory_avg_formatted": Utils.format_memory_size(int(avg_mem_bytes)),
            },
            "alerts": alerts or ["System nominal."],
        }


# ==============================================================================
# 4. CACHE MANAGER
# ==============================================================================


class CacheManager:
    """A simple in-memory cache with TTL, size limits, and hashed keys."""

    def __init__(
        self, max_size: int = Config.MAX_CACHE_SIZE, ttl: int = Config.DEFAULT_CACHE_TTL_SECONDS
    ):
        self.cache: Dict[str, Tuple[Any, float]] = {}
        self.max_size = max_size
        self.ttl = ttl
        self.lock = threading.Lock()

        self.is_running = False
        self.cleanup_thread = threading.Thread(target=self._cleanup_expired_entries, daemon=True)

    def start(self) -> None:
        if not self.is_running:
            self.is_running = True
            self.cleanup_thread.start()
            logging.info("Cache manager started.")

    def stop(self) -> None:
        self.is_running = False
        logging.info("Cache manager stopping.")

    def get(self, key: str) -> Optional[Any]:
        hashed_key = Utils.hash_data(key)
        with self.lock:
            if hashed_key in self.cache:
                value, expiration = self.cache[hashed_key]
                if time.time() < expiration:
                    return value
                del self.cache[hashed_key]
        return None

    def set(self, key: str, value: Any) -> None:
        hashed_key = Utils.hash_data(key)
        with self.lock:
            if len(self.cache) >= self.max_size and hashed_key not in self.cache:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]

            expiration = time.time() + self.ttl
            self.cache[hashed_key] = (value, expiration)

    def _cleanup_expired_entries(self) -> None:
        while self.is_running:
            with self.lock:
                expired_keys = [
                    key for key, (_, expiration) in self.cache.items() if time.time() >= expiration
                ]
                for key in expired_keys:
                    del self.cache[key]
            time.sleep(Config.CACHE_CLEANUP_INTERVAL_SECONDS)


# ==============================================================================
# 5. KNOWLEDGE GRAPH, DRIFT DETECTOR, and MOCK LLM
# (Structurally unchanged, kept for completeness)
# ==============================================================================


class DriftDetector:
    def __init__(self):
        self.reference_data: Optional[np.ndarray] = None

    def set_reference_data(self, data: List[float]):
        self.reference_data = np.array(data)

    def check_drift(self, current_data: List[float]) -> Dict[str, Any]:
        if self.reference_data is None or len(current_data) < 2:
            return {"drift_detected": False}
        _, p_value = stats.ks_2samp(self.reference_data, np.array(current_data))
        drift_detected = p_value < Config.DRIFT_DETECTION_KS_THRESHOLD
        if drift_detected:
            logging.warning(f"Data drift detected! p-value: {p_value:.4f}")
        return {"drift_detected": drift_detected, "p_value": p_value}


class KnowledgeGraph:
    def __init__(self):
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.edges: Dict[str, Dict[str, Any]] = {}

    def add_node(self, node_id: str, node_type: str, attributes: Dict[str, Any]):
        if node_id not in self.nodes:
            self.nodes[node_id] = {"type": node_type, **attributes}

    def add_edge(self, source_id: str, target_id: str, relationship: str):
        if source_id in self.nodes and target_id in self.nodes:
            if source_id not in self.edges:
                self.edges[source_id] = {}
            self.edges[source_id][target_id] = {"relationship": relationship}

    def search_nodes(self, query: str) -> List[Dict[str, Any]]:
        return [
            {"id": nid, **attrs}
            for nid, attrs in self.nodes.items()
            if query.lower() in attrs.get("name", "").lower()
        ]


def populate_sample_knowledge_graph(kg: KnowledgeGraph):
    logging.info("Populating knowledge graph with sample data...")
    kg.add_node("D003924", "disease", {"name": "Type 2 Diabetes"})
    kg.add_node("T0001", "treatment", {"name": "Metformin"})
    kg.add_node("T0003", "treatment", {"name": "Lifestyle changes"})
    kg.add_edge("D003924", "T0001", "treated_by")
    kg.add_edge("D003924", "T0003", "treated_by")
    logging.info("Knowledge graph populated.")


class MockLanguageModel:
    def generate_summary(self, context: str, query: str) -> str:
        if not context:
            return f"I have no information regarding '{query}'."
        return f"Summary for '{query}': Key information found relates to {context}."


# ==============================================================================
# 6. MEDICAL AI AGENT
# ==============================================================================


class MedicalAgent:
    def __init__(
        self,
        knowledge_graph: KnowledgeGraph,
        cache: CacheManager,
        drift_detector: DriftDetector,
        llm: MockLanguageModel,
    ):
        self.kg = knowledge_graph
        self.cache = cache
        self.drift_detector = drift_detector
        self.llm = llm
        self.drift_detector.set_reference_data(list(np.random.normal(0.85, 0.1, 100)))

    def process_task(self, task: str) -> Dict[str, Any]:
        cached_result = self.cache.get(task)
        if cached_result:
            logging.info(f"Cache hit for task: '{task[:50]}...'")
            return {"response": cached_result, "source": "cache"}

        search_results = self.kg.search_nodes(task)
        context = ", ".join([res.get("name", res["id"]) for res in search_results])
        response = self.llm.generate_summary(context, task)

        current_metric = [np.random.normal(0.84, 0.12)]
        drift_info = self.drift_detector.check_drift(current_metric * 30)

        self.cache.set(task, response)
        logging.info(f"Cached new response for task: '{task[:50]}...'")

        return {
            "response": response,
            "source": "generated",
            "knowledge_graph_hits": len(search_results),
            "drift_detection": drift_info,
        }


# ==============================================================================
# 7. FLASK APPLICATION AND API ENDPOINTS
# ==============================================================================

app = Flask(__name__)
logging.basicConfig(level=Config.DEFAULT_LOG_LEVEL, format=Config.LOG_FORMAT)

# Instantiate Core Components
performance_monitor = PerformanceMonitor()
cache_manager = CacheManager()
drift_detector = DriftDetector()
knowledge_graph = KnowledgeGraph()
populate_sample_knowledge_graph(knowledge_graph)
mock_llm = MockLanguageModel()
medical_agent = MedicalAgent(knowledge_graph, cache_manager, drift_detector, mock_llm)


@app.before_request
def before_request_hook():
    performance_monitor.record_request_start()
    g.request_id = Utils.generate_request_id()


@app.after_request
def after_request_hook(response):
    performance_monitor.record_request_end(response.status_code)
    response.headers["X-Request-ID"] = g.get("request_id", "unknown")
    return response


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok"}), 200


@app.route("/metrics", methods=["GET"])
def get_metrics():
    return jsonify(performance_monitor.get_metrics())


@app.route("/api/v1/task", methods=["POST"])
def handle_task():
    request_id = g.request_id
    logging.info(f"[ReqID: {request_id}] Received new task request.")

    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON body.", "request_id": request_id}), 400

    task = Utils.safe_json_extract(data, "task")
    if not task or not isinstance(task, str):
        return (
            jsonify(
                {
                    "error": "Missing or invalid 'task' field in request body.",
                    "request_id": request_id,
                }
            ),
            400,
        )

    if len(task) > Config.MAX_TASK_LENGTH:
        return (
            jsonify(
                {
                    "error": f"Task exceeds maximum length of {Config.MAX_TASK_LENGTH} characters.",
                    "request_id": request_id,
                }
            ),
            413,
        )

    try:
        result = medical_agent.process_task(task)

        # Create and log an audit event for the successful request
        audit_event = Utils.create_audit_event(
            event_type="TASK_PROCESSED",
            request_id=request_id,
            details={"task_length": len(task), "source": result.get("source")},
        )
        logging.info(f"AUDIT: {audit_event}")

        return jsonify({"request_id": request_id, **result}), 200

    except Exception as e:
        error_id = Utils.generate_error_id()
        logging.error(
            f"[ReqID: {request_id}] [ErrorID: {error_id}] Error processing task: {e}", exc_info=True
        )
        return (
            jsonify(
                {
                    "error": "An internal server error occurred.",
                    "error_id": error_id,
                    "request_id": request_id,
                }
            ),
            500,
        )


# ==============================================================================
# 8. MAIN EXECUTION BLOCK
# ==============================================================================

if __name__ == "__main__":
    # Validate configuration on startup
    is_valid, missing_keys = Utils.validate_config_keys(
        vars(Config), ["DEFAULT_API_PORT", "DEFAULT_LOG_LEVEL", "MAX_TASK_LENGTH"]
    )
    if not is_valid:
        logging.critical(
            f"CRITICAL: Missing required configuration keys: {missing_keys}. Shutting down."
        )
        sys.exit(1)

    try:
        performance_monitor.start()
        cache_manager.start()

        app.run(host="0.0.0.0", port=Config.DEFAULT_API_PORT, debug=False)

    except KeyboardInterrupt:
        logging.info("Application shutting down due to user interrupt.")
    finally:
        performance_monitor.stop()
        cache_manager.stop()
        logging.info("Shutdown complete.")
