# -*- coding: utf-8 -*-
"""
AimedRes: AI-Powered Medical Research Assistant
=================================================

This single-file script is a fully integrated and improved version of the AimedRes
repository. It combines a Flask API, performance monitoring, caching, a knowledge
graph, and a medical AI agent into one cohesive application.

To run this script:
1. Install dependencies: pip install Flask psutil numpy scipy
2. Execute: python <this_file_name>.py
"""

import time
import threading
import logging
from collections import deque
from typing import Dict, Any, List, Optional, Tuple, Deque
import psutil
import numpy as np
from scipy import stats
from flask import Flask, jsonify, request, g

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
    DEFAULT_RATE_LIMIT: int = 100  # requests per minute
    DEFAULT_MAX_CONCURRENT_REQUESTS: int = 50

    # Cache Configuration
    DEFAULT_CACHE_TTL_SECONDS: int = 3600  # 1 hour
    MAX_CACHE_SIZE: int = 1000
    CACHE_CLEANUP_INTERVAL_SECONDS: int = 300  # 5 minutes

    # Logging Configuration
    DEFAULT_LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    # Input Validation
    MAX_TASK_LENGTH: int = 2000
    MAX_QUERY_LENGTH: int = 500

    # Drift Detection Constants
    DRIFT_DETECTION_KS_THRESHOLD: float = 0.05  # P-value for Kolmogorov-Smirnov test


# ==============================================================================
# 2. PERFORMANCE MONITOR
# ==============================================================================

class PerformanceMonitor:
    """
    Monitors and reports application performance metrics like response time,
    throughput, CPU/memory usage, and error rates.
    """
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

    def start(self) -> None:
        """Starts the background monitoring thread."""
        if not self.is_running:
            self.is_running = True
            self.monitor_thread.start()
            logging.info("Performance monitor started.")

    def stop(self) -> None:
        """Stops the background monitoring thread."""
        self.is_running = False
        logging.info("Performance monitor stopping.")

    def _monitor_system_resources(self) -> None:
        """Periodically records CPU and memory usage."""
        while self.is_running:
            try:
                # Memory usage in MB
                mem_info = self.process.memory_info()
                self.memory_history.append(mem_info.rss / (1024 * 1024))
                
                # CPU usage as a percentage
                self.cpu_history.append(self.process.cpu_percent(interval=0.1))
            except psutil.NoSuchProcess:
                logging.warning("Monitoring process not found. Stopping monitor.")
                self.is_running = False
            except Exception as e:
                logging.error(f"Error in performance monitor thread: {e}")
            
            time.sleep(self.monitoring_interval)

    def record_request_start(self) -> None:
        """Records the start time of a request."""
        g.start_time = time.perf_counter()

    def record_request_end(self, status_code: int) -> None:
        """Records request completion and calculates response time."""
        if 'start_time' in g:
            response_time = time.perf_counter() - g.start_time
            self.response_times.append(response_time)
            self.request_timestamps.append(time.time())
            
            if 200 <= status_code < 400:
                self.success_count += 1
            else:
                self.error_count += 1

    def get_metrics(self) -> Dict[str, Any]:
        """Calculates and returns a dictionary of current performance metrics."""
        now = time.time()
        
        # Throughput (requests per minute)
        recent_requests = [t for t in self.request_timestamps if now - t <= Config.THROUGHPUT_CALCULATION_WINDOW_SECONDS]
        throughput_rpm = len(recent_requests) * (60 / Config.THROUGHPUT_CALCULATION_WINDOW_SECONDS)

        total_requests = self.success_count + self.error_count
        error_rate = (self.error_count / total_requests) if total_requests > 0 else 0

        # Response time stats
        if not self.response_times:
            avg_response_time, p95_response_time, p99_response_time = 0, 0, 0
        else:
            response_times_np = np.array(list(self.response_times))
            avg_response_time = np.mean(response_times_np)
            p95_response_time = np.percentile(response_times_np, 95)
            p99_response_time = np.percentile(response_times_np, 99)

        # Resource usage stats
        avg_mem = np.mean(list(self.memory_history)) if self.memory_history else 0
        avg_cpu = np.mean(list(self.cpu_history)) if self.cpu_history else 0

        # Alerts
        alerts = []
        if avg_mem > Config.MEMORY_ALERT_THRESHOLD_MB:
            alerts.append(f"High Memory Usage: {avg_mem:.2f} MB")
        if avg_cpu > Config.CPU_ALERT_THRESHOLD_PERCENT:
            alerts.append(f"High CPU Usage: {avg_cpu:.2f}%")
        if avg_response_time > Config.RESPONSE_TIME_ALERT_THRESHOLD_S:
            alerts.append(f"High Response Time: {avg_response_time*1000:.2f} ms")
        if error_rate > Config.ERROR_RATE_ALERT_THRESHOLD and total_requests > 10:
            alerts.append(f"High Error Rate: {error_rate*100:.2f}%")

        return {
            "throughput_rpm": round(throughput_rpm, 2),
            "total_requests": total_requests,
            "error_rate_percent": round(error_rate * 100, 2),
            "response_time_ms": {
                "average": round(avg_response_time * 1000, 2),
                "p95": round(p95_response_time * 1000, 2),
                "p99": round(p99_response_time * 1000, 2),
            },
            "resource_usage": {
                "cpu_percent_avg": round(avg_cpu, 2),
                "memory_mb_avg": round(avg_mem, 2),
            },
            "alerts": alerts or ["System nominal."],
        }


# ==============================================================================
# 3. CACHE MANAGER
# ==============================================================================

class CacheManager:
    """
    A simple in-memory cache with Time-To-Live (TTL) and size limits.
    """
    def __init__(self, max_size: int = Config.MAX_CACHE_SIZE, ttl: int = Config.DEFAULT_CACHE_TTL_SECONDS):
        self.cache: Dict[str, Tuple[Any, float]] = {}
        self.max_size = max_size
        self.ttl = ttl
        self.lock = threading.Lock()
        
        self.is_running = False
        self.cleanup_thread = threading.Thread(target=self._cleanup_expired_entries, daemon=True)

    def start(self) -> None:
        """Starts the background cache cleanup thread."""
        if not self.is_running:
            self.is_running = True
            self.cleanup_thread.start()
            logging.info("Cache manager started.")
    
    def stop(self) -> None:
        """Stops the background cache cleanup thread."""
        self.is_running = False
        logging.info("Cache manager stopping.")

    def get(self, key: str) -> Optional[Any]:
        """Retrieves an item from the cache if it exists and is not expired."""
        with self.lock:
            if key in self.cache:
                value, expiration = self.cache[key]
                if time.time() < expiration:
                    return value
                # Entry expired, remove it
                del self.cache[key]
        return None

    def set(self, key: str, value: Any) -> None:
        """Adds an item to the cache."""
        with self.lock:
            if len(self.cache) >= self.max_size:
                # Evict the oldest entry (simple strategy)
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
            
            expiration = time.time() + self.ttl
            self.cache[key] = (value, expiration)

    def _cleanup_expired_entries(self) -> None:
        """Periodically removes expired entries from the cache."""
        while self.is_running:
            with self.lock:
                expired_keys = [
                    key for key, (_, expiration) in self.cache.items()
                    if time.time() >= expiration
                ]
                for key in expired_keys:
                    del self.cache[key]
            time.sleep(Config.CACHE_CLEANUP_INTERVAL_SECONDS)


# ==============================================================================
# 4. DATA DRIFT DETECTOR
# ==============================================================================

class DriftDetector:
    """
    Detects data drift between a reference distribution and a current distribution
    using the Kolmogorov-Smirnov (KS) test.
    """
    def __init__(self):
        self.reference_data: Optional[np.ndarray] = None
        logging.info("Drift Detector initialized.")

    def set_reference_data(self, data: List[float]) -> None:
        """Sets the baseline data distribution for comparison."""
        self.reference_data = np.array(data)
        logging.info(f"Drift detector reference data set with {len(data)} points.")

    def check_drift(self, current_data: List[float]) -> Dict[str, Any]:
        """
        Compares current data to the reference data to detect drift.
        Returns a dictionary with drift status and test statistics.
        """
        if self.reference_data is None or len(self.reference_data) < 20:
            return {"drift_detected": False, "reason": "No or insufficient reference data."}
        
        if len(current_data) < 20:
            return {"drift_detected": False, "reason": "Insufficient current data for comparison."}

        current_data_np = np.array(current_data)
        
        # Two-sample Kolmogorov-Smirnov test
        ks_statistic, p_value = stats.ks_2samp(self.reference_data, current_data_np)
        
        drift_detected = p_value < Config.DRIFT_DETECTION_KS_THRESHOLD
        
        if drift_detected:
            logging.warning(f"Data drift detected! KS test p-value: {p_value:.4f}")
        
        return {
            "drift_detected": drift_detected,
            "p_value": p_value,
            "ks_statistic": ks_statistic,
            "threshold": Config.DRIFT_DETECTION_KS_THRESHOLD,
        }


# ==============================================================================
# 5. KNOWLEDGE GRAPH
# ==============================================================================

class KnowledgeGraph:
    """
    Represents a medical knowledge graph with nodes (concepts) and
    edges (relationships).
    """
    def __init__(self):
        self.nodes: Dict[str, Dict[str, Any]] = {}  # {node_id: {attributes}}
        self.edges: Dict[str, Dict[str, Any]] = {}  # {source_id: {target_id: {attributes}}}

    def add_node(self, node_id: str, node_type: str, attributes: Dict[str, Any]) -> None:
        """Adds a node to the graph."""
        if node_id not in self.nodes:
            self.nodes[node_id] = {"type": node_type, **attributes}

    def add_edge(self, source_id: str, target_id: str, relationship: str, attributes: Dict[str, Any] = None) -> None:
        """Adds a directed edge between two nodes."""
        if source_id in self.nodes and target_id in self.nodes:
            if source_id not in self.edges:
                self.edges[source_id] = {}
            self.edges[source_id][target_id] = {"relationship": relationship, **(attributes or {})}

    def search_nodes(self, query: str, node_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Performs a simple case-insensitive search for nodes containing the query
        string in their ID or name attribute.
        """
        results = []
        query_lower = query.lower()
        for node_id, attrs in self.nodes.items():
            if node_type and attrs.get("type") != node_type:
                continue
            
            # Search in node_id and 'name' attribute
            if query_lower in node_id.lower() or query_lower in attrs.get("name", "").lower():
                results.append({"id": node_id, **attrs})
        return results

    def get_neighbors(self, node_id: str) -> List[Dict[str, Any]]:
        """Gets all neighbors and their relationship data for a given node."""
        neighbors = []
        if node_id in self.edges:
            for target_id, attrs in self.edges[node_id].items():
                neighbors.append({
                    "target_node": {"id": target_id, **self.nodes.get(target_id, {})},
                    "relationship": attrs
                })
        return neighbors

def populate_sample_knowledge_graph(kg: KnowledgeGraph):
    """Fills the KnowledgeGraph with sample medical data for demonstration."""
    logging.info("Populating knowledge graph with sample data...")
    # Diseases
    kg.add_node("D003924", "disease", {"name": "Type 2 Diabetes", "description": "A chronic condition that affects the way the body processes blood sugar (glucose)."})
    kg.add_node("D006973", "disease", {"name": "Hypertension", "description": "A condition in which the force of the blood against the artery walls is too high."})

    # Symptoms
    kg.add_node("S00001", "symptom", {"name": "Frequent urination"})
    kg.add_node("S00002", "symptom", {"name": "Increased thirst"})
    kg.add_node("S00003", "symptom", {"name": "Blurred vision"})
    kg.add_node("S00004", "symptom", {"name": "Headache"})

    # Treatments
    kg.add_node("T0001", "treatment", {"name": "Metformin", "class": "Biguanide"})
    kg.add_node("T0002", "treatment", {"name": "Insulin Therapy"})
    kg.add_node("T0003", "treatment", {"name": "Lifestyle changes", "details": "Diet, exercise, weight loss"})
    kg.add_node("T0004", "treatment", {"name": "Lisinopril", "class": "ACE inhibitor"})

    # Relationships
    kg.add_edge("D003924", "S00001", "has_symptom")
    kg.add_edge("D003924", "S00002", "has_symptom")
    kg.add_edge("D003924", "S00003", "has_symptom")
    kg.add_edge("D006973", "S00004", "has_symptom")

    kg.add_edge("D003924", "T0001", "treated_by")
    kg.add_edge("D003924", "T0002", "treated_by")
    kg.add_edge("D003924", "T0003", "treated_by")
    kg.add_edge("D006973", "T0004", "treated_by")
    
    kg.add_edge("D003924", "D006973", "is_comorbidity_of")
    kg.add_edge("D006973", "D003924", "is_comorbidity_of")
    logging.info("Knowledge graph populated.")


# ==============================================================================
# 6. MEDICAL AI AGENT
# ==============================================================================

class MockLanguageModel:
    """A mock class to simulate responses from a large language model."""
    def generate_summary(self, context: str, query: str) -> str:
        """Generates a plausible-sounding summary based on context."""
        logging.info(f"MockLLM generating summary for query: '{query}'")
        if not context:
            return f"I'm sorry, but I have no information regarding '{query}'. Please provide more context."
        
        return (f"Based on the available information regarding '{query}', here is a summary: "
                f"The key concepts involved are {context}. It is important to consider all aspects "
                f"before drawing a conclusion. Consulting a medical professional is always recommended.")

class MedicalAgent:
    """
    The core AI agent that processes tasks, interacts with the knowledge graph,
    and uses a language model to generate responses.
    """
    def __init__(self, knowledge_graph: KnowledgeGraph, cache: CacheManager, drift_detector: DriftDetector, llm: MockLanguageModel):
        self.kg = knowledge_graph
        self.cache = cache
        self.drift_detector = drift_detector
        self.llm = llm
        
        # Example reference data for drift detection (e.g., confidence scores)
        self.drift_detector.set_reference_data(list(np.random.normal(0.85, 0.1, 100)))

    def _search_knowledge_base(self, query: str) -> List[Dict[str, Any]]:
        """Searches the knowledge graph for relevant information."""
        logging.info(f"Searching knowledge graph for query: '{query}'")
        # A simple strategy: search for diseases, symptoms, and treatments
        results = []
        results.extend(self.kg.search_nodes(query, "disease"))
        results.extend(self.kg.search_nodes(query, "symptom"))
        results.extend(self.kg.search_nodes(query, "treatment"))
        return results

    def _synthesize_response(self, query: str, search_results: List[Dict[str, Any]]) -> str:
        """Generates a coherent response using the search results and an LLM."""
        if not search_results:
            return self.llm.generate_summary("", query)

        # Create a context string from search results
        context_parts = []
        for res in search_results:
            name = res.get('name', res.get('id'))
            desc = res.get('description', '')
            context_parts.append(f"{name}: {desc}")
        
        context = "; ".join(context_parts)
        return self.llm.generate_summary(context, query)

    def process_task(self, task: str) -> Dict[str, Any]:
        """
        Processes a user task, from caching and searching to response generation.
        """
        # 1. Check cache first
        cached_result = self.cache.get(task)
        if cached_result:
            logging.info(f"Cache hit for task: '{task}'")
            return {"response": cached_result, "source": "cache"}

        # 2. Search knowledge base
        search_results = self._search_knowledge_base(task)

        # 3. Synthesize response
        response = self._synthesize_response(task, search_results)
        
        # 4. Perform drift detection (on a simulated metric like confidence)
        current_metric = [np.random.normal(0.84, 0.12)] # Simulate a new data point
        drift_info = self.drift_detector.check_drift(current_metric * 30) # multiply to meet sample size

        # 5. Cache the new result
        self.cache.set(task, response)
        logging.info(f"Cached new response for task: '{task}'")

        return {
            "response": response,
            "source": "generated",
            "knowledge_graph_hits": len(search_results),
            "drift_detection": drift_info,
        }

# ==============================================================================
# 7. FLASK APPLICATION AND API ENDPOINTS
# ==============================================================================

# --- Application Setup ---
app = Flask(__name__)
logging.basicConfig(level=Config.DEFAULT_LOG_LEVEL, format=Config.LOG_FORMAT)

# --- Instantiate Core Components ---
performance_monitor = PerformanceMonitor()
cache_manager = CacheManager()
drift_detector = DriftDetector()
knowledge_graph = KnowledgeGraph()
populate_sample_knowledge_graph(knowledge_graph)
mock_llm = MockLanguageModel()

medical_agent = MedicalAgent(
    knowledge_graph=knowledge_graph,
    cache=cache_manager,
    drift_detector=drift_detector,
    llm=mock_llm
)

# --- Flask Request Hooks for Performance Monitoring ---
@app.before_request
def before_request_hook():
    """Executed before each request to record the start time."""
    performance_monitor.record_request_start()

@app.after_request
def after_request_hook(response):
    """Executed after each request to record metrics."""
    performance_monitor.record_request_end(response.status_code)
    return response

# --- API Endpoints ---
@app.route("/")
def index():
    """Welcome endpoint."""
    return jsonify({
        "message": "Welcome to the AimedRes API",
        "version": "1.0",
        "documentation": "See /health, /metrics, and POST /api/v1/task"
    })

@app.route("/health", methods=['GET'])
def health_check():
    """Provides a simple health check for the service."""
    return jsonify({"status": "ok"}), 200

@app.route("/metrics", methods=['GET'])
def get_metrics():
    """Returns a snapshot of the application's performance metrics."""
    return jsonify(performance_monitor.get_metrics())

@app.route("/api/v1/task", methods=['POST'])
def handle_task():
    """Main endpoint to process a medical research task."""
    data = request.get_json()
    if not data or "task" not in data:
        return jsonify({"error": "Missing 'task' field in request body"}), 400

    task = data["task"]
    if not isinstance(task, str) or len(task) > Config.MAX_TASK_LENGTH:
        return jsonify({"error": f"Invalid 'task'. Must be a string under {Config.MAX_TASK_LENGTH} characters."}), 400

    try:
        result = medical_agent.process_task(task)
        return jsonify(result), 200
    except Exception as e:
        logging.error(f"Error processing task '{task}': {e}", exc_info=True)
        return jsonify({"error": "An internal server error occurred."}), 500

# ==============================================================================
# 8. MAIN EXECUTION BLOCK
# ==============================================================================

if __name__ == '__main__':
    try:
        # Start background services
        performance_monitor.start()
        cache_manager.start()

        # Start the Flask web server
        # For production, use a WSGI server like Gunicorn instead of app.run()
        # Example: gunicorn --workers 4 --bind 0.0.0.0:8080 aimed_res_app:app
        app.run(host='0.0.0.0', port=Config.DEFAULT_API_PORT, debug=False)

    except KeyboardInterrupt:
        logging.info("Application shutting down.")
    finally:
        # Gracefully stop background services
        performance_monitor.stop()
        cache_manager.stop()
        logging.info("Shutdown complete.")

# Export constants at module level for backward compatibility
# Only export those that exist in Config class
DEFAULT_MONITORING_INTERVAL_SECONDS = Config.DEFAULT_MONITORING_INTERVAL_SECONDS
MAX_REQUEST_TIMESTAMPS_STORED = Config.MAX_REQUEST_TIMESTAMPS_STORED
MAX_MEMORY_HISTORY_ENTRIES = Config.MAX_MEMORY_HISTORY_ENTRIES
MAX_CPU_HISTORY_ENTRIES = Config.MAX_CPU_HISTORY_ENTRIES
THROUGHPUT_CALCULATION_WINDOW_SECONDS = Config.THROUGHPUT_CALCULATION_WINDOW_SECONDS
DEFAULT_API_PORT = Config.DEFAULT_API_PORT
DEFAULT_RATE_LIMIT = Config.DEFAULT_RATE_LIMIT
DEFAULT_MAX_CONCURRENT_REQUESTS = Config.DEFAULT_MAX_CONCURRENT_REQUESTS
MAX_TASK_LENGTH = Config.MAX_TASK_LENGTH
MAX_QUERY_LENGTH = Config.MAX_QUERY_LENGTH

# These constants don't exist in Config, so we define them here
# based on the original constants.py requirements
MAX_KNOWLEDGE_SEARCH_RESULTS = 50
DEFAULT_KNOWLEDGE_SEARCH_LIMIT = 10
DEFAULT_MEMORY_LIMIT_MB = 1024
VECTOR_SEARCH_BATCH_THRESHOLD = 100
DEFAULT_NETWORK_SIZE = 100
NODE_BATCH_PROCESSING_THRESHOLD = 10
DEFAULT_NODE_ENERGY = 1.0
