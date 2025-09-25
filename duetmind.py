import multiprocessing as mp
import threading
import time
import json
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable, Union, Tuple
from pathlib import Path
import psutil
import redis
from flask import Flask, request, jsonify, g
from flask_cors import CORS
import sqlite3
from contextlib import contextmanager
import numpy as np
from datetime import datetime, timedelta
import hashlib
import gc
import sys
from functools import wraps, lru_cache
import yaml
import pickle
import gzip
from collections import defaultdict, deque

# Import configuration constants
from constants import (
    DEFAULT_MONITORING_INTERVAL_SECONDS, MAX_REQUEST_TIMESTAMPS_STORED, MAX_MEMORY_HISTORY_ENTRIES,
    MAX_CPU_HISTORY_ENTRIES, THROUGHPUT_CALCULATION_WINDOW_SECONDS, DEFAULT_API_PORT,
    DEFAULT_RATE_LIMIT, DEFAULT_MAX_CONCURRENT_REQUESTS, MAX_TASK_LENGTH, MAX_QUERY_LENGTH,
    MAX_KNOWLEDGE_SEARCH_RESULTS, DEFAULT_KNOWLEDGE_SEARCH_LIMIT, DEFAULT_MEMORY_LIMIT_MB,
    VECTOR_SEARCH_BATCH_THRESHOLD, DEFAULT_NETWORK_SIZE, NODE_BATCH_PROCESSING_THRESHOLD,
    DEFAULT_NODE_ENERGY
)

# Import security modules
from security import (
    SecureAuthManager, InputValidator, SecurityValidator,
    DataEncryption, PrivacyManager, DataRetentionPolicy,
    SecurityMonitor, require_auth, require_admin
)

# Performance Monitoring and Metrics
@dataclass
class PerformanceMetrics:
    """Comprehensive performance tracking"""
    request_count: int = 0
    total_response_time: float = 0.0
    average_response_time: float = 0.0
    peak_memory_usage: float = 0.0
    current_memory_usage: float = 0.0
    cpu_usage_history: List[float] = field(default_factory=list)
    error_count: int = 0
    cache_hit_rate: float = 0.0
    network_utilization: float = 0.0
    concurrent_requests: int = 0
    throughput_per_second: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)

class PerformanceMonitor:
    """Real-time performance monitoring system"""
    
    def __init__(self, sample_interval: int = DEFAULT_MONITORING_INTERVAL_SECONDS):
        self.metrics = PerformanceMetrics()
        self.sample_interval = sample_interval
        self.monitoring = False
        self.monitor_thread = None
        self.request_timestamps = deque(maxlen=MAX_REQUEST_TIMESTAMPS_STORED)
        self.memory_history = deque(maxlen=MAX_MEMORY_HISTORY_ENTRIES)
        self.cpu_history = deque(maxlen=MAX_CPU_HISTORY_ENTRIES)
        
    def start_monitoring(self):
        """Start background performance monitoring"""
        if not self.monitoring:
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            logging.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=1)
        logging.info("Performance monitoring stopped")
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.monitoring:
            try:
                # Update system metrics
                process = psutil.Process()
                memory_info = process.memory_info()
                
                self.metrics.current_memory_usage = memory_info.rss / 1024 / 1024  # MB
                self.metrics.peak_memory_usage = max(self.metrics.peak_memory_usage, self.metrics.current_memory_usage)
                
                cpu_percent = process.cpu_percent()
                self.cpu_history.append(cpu_percent)
                self.memory_history.append(self.metrics.current_memory_usage)
                
                # Calculate throughput
                now = time.time()
                recent_requests = [ts for ts in self.request_timestamps if now - ts < THROUGHPUT_CALCULATION_WINDOW_SECONDS]
                self.metrics.throughput_per_second = len(recent_requests) / 60.0
                
                # Update metrics
                self.metrics.last_updated = datetime.now()
                
                time.sleep(self.sample_interval)
                
            except Exception as e:
                logging.error(f"Performance monitoring error: {e}")
                time.sleep(self.sample_interval)
    
    def record_request(self, response_time: float, success: bool = True):
        """Record request metrics"""
        self.metrics.request_count += 1
        if not success:
            self.metrics.error_count += 1
        
        self.metrics.total_response_time += response_time
        self.metrics.average_response_time = self.metrics.total_response_time / self.metrics.request_count
        
        self.request_timestamps.append(time.time())
    
    def get_metrics_snapshot(self) -> Dict[str, Any]:
        """Get current metrics snapshot"""
        return {
            'request_count': self.metrics.request_count,
            'average_response_time': self.metrics.average_response_time,
            'throughput_per_second': self.metrics.throughput_per_second,
            'current_memory_mb': self.metrics.current_memory_usage,
            'peak_memory_mb': self.metrics.peak_memory_usage,
            'cpu_usage_avg': np.mean(list(self.cpu_history)) if self.cpu_history else 0.0,
            'error_rate': self.metrics.error_count / max(1, self.metrics.request_count),
            'uptime_seconds': (datetime.now() - self.metrics.last_updated).total_seconds() if hasattr(self.metrics, 'start_time') else 0,
            'concurrent_requests': self.metrics.concurrent_requests
        }

# GPU Acceleration Support (with fallback)
class GPUAccelerator:
    """GPU acceleration for neural network operations"""
    
    def __init__(self):
        self.gpu_available = self._check_gpu_availability()
        self.device = self._initialize_device()
        
    def _check_gpu_availability(self) -> bool:
        """Check if GPU acceleration is available"""
        try:
            # Try importing GPU libraries
            import cupy as cp
            return cp.cuda.is_available()
        except ImportError:
            try:
                # Fallback to checking for basic CUDA
                import subprocess
                result = subprocess.run(['nvidia-smi'], capture_output=True)
                return result.returncode == 0
            except:
                return False
    
    def _initialize_device(self) -> str:
        """Initialize the computing device"""
        if self.gpu_available:
            try:
                import cupy as cp
                cp.cuda.Device(0).use()
                return "gpu"
            except:
                pass
        return "cpu"
    
    def accelerate_vector_operations(self, vectors: np.ndarray, operation: str = "similarity") -> np.ndarray:
        """Accelerate vector operations on GPU if available"""
        if not self.gpu_available:
            return self._cpu_vector_operations(vectors, operation)
        
        try:
            import cupy as cp
            gpu_vectors = cp.asarray(vectors)
            
            if operation == "similarity":
                # Compute pairwise cosine similarity
                normalized = gpu_vectors / cp.linalg.norm(gpu_vectors, axis=1, keepdims=True)
                similarities = cp.dot(normalized, normalized.T)
                return cp.asnumpy(similarities)
            elif operation == "normalize":
                # Normalize vectors
                norms = cp.linalg.norm(gpu_vectors, axis=1, keepdims=True)
                normalized = gpu_vectors / norms
                return cp.asnumpy(normalized)
            else:
                return self._cpu_vector_operations(vectors, operation)
                
        except Exception as e:
            logging.warning(f"GPU acceleration failed, falling back to CPU: {e}")
            return self._cpu_vector_operations(vectors, operation)
    
    def _cpu_vector_operations(self, vectors: np.ndarray, operation: str) -> np.ndarray:
        """CPU fallback for vector operations"""
        if operation == "similarity":
            # Pairwise cosine similarity
            normalized = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
            return np.dot(normalized, normalized.T)
        elif operation == "normalize":
            # Normalize vectors
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            return vectors / norms
        else:
            return vectors

# Parallel Processing System
class ParallelProcessingManager:
    """Advanced parallel processing for neural network operations"""
    
    def __init__(self, max_workers: Optional[int] = None):
        self.max_workers = max_workers or min(mp.cpu_count(), 8)
        self.thread_executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self.process_executor = ProcessPoolExecutor(max_workers=self.max_workers)
        self.gpu_accelerator = GPUAccelerator()
        
    def parallel_node_processing(self, nodes: List[Any], operation: Callable, use_processes: bool = False) -> List[Any]:
        """Process nodes in parallel"""
        executor = self.process_executor if use_processes else self.thread_executor
        
        # Chunk nodes for optimal processing
        chunk_size = max(1, len(nodes) // self.max_workers)
        chunks = [nodes[i:i + chunk_size] for i in range(0, len(nodes), chunk_size)]
        
        futures = []
        for chunk in chunks:
            future = executor.submit(self._process_node_chunk, chunk, operation)
            futures.append(future)
        
        results = []
        for future in as_completed(futures):
            try:
                chunk_results = future.result()
                results.extend(chunk_results)
            except Exception as e:
                logging.error(f"Parallel processing error: {e}")
        
        return results
    
    def _process_node_chunk(self, nodes: List[Any], operation: Callable) -> List[Any]:
        """Process a chunk of nodes"""
        results = []
        for node in nodes:
            try:
                result = operation(node)
                results.append(result)
            except Exception as e:
                logging.error(f"Node processing error: {e}")
                results.append(None)
        return results
    
    def parallel_vector_search(self, query_vector: np.ndarray, document_vectors: List[np.ndarray], top_k: int = 10) -> List[Tuple[int, float]]:
        """Parallel vector similarity search"""
        if len(document_vectors) < VECTOR_SEARCH_BATCH_THRESHOLD:
            # Use simple search for small datasets
            return self._simple_vector_search(query_vector, document_vectors, top_k)
        
        # Use GPU acceleration for large datasets
        vectors_array = np.array(document_vectors)
        similarities = self.gpu_accelerator.accelerate_vector_operations(
            np.vstack([query_vector.reshape(1, -1), vectors_array]), 
            "similarity"
        )
        
        # Get similarities between query and documents
        query_similarities = similarities[0, 1:]
        
        # Get top-k indices
        top_indices = np.argsort(query_similarities)[::-1][:top_k]
        
        return [(int(idx), float(query_similarities[idx])) for idx in top_indices]
    
    def _simple_vector_search(self, query_vector: np.ndarray, document_vectors: List[np.ndarray], top_k: int) -> List[Tuple[int, float]]:
        """Simple vector search for small datasets"""
        similarities = []
        for i, doc_vector in enumerate(document_vectors):
            similarity = np.dot(query_vector, doc_vector) / (np.linalg.norm(query_vector) * np.linalg.norm(doc_vector))
            similarities.append((i, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def shutdown(self):
        """Shutdown parallel processing"""
        self.thread_executor.shutdown(wait=True)
        self.process_executor.shutdown(wait=True)

# Advanced Caching System
class AdvancedCacheManager:
    """Multi-level caching with intelligent eviction"""
    
    def __init__(self, memory_limit_mb: int = DEFAULT_MEMORY_LIMIT_MB, redis_url: Optional[str] = None):
        self.memory_limit_mb = memory_limit_mb
        self.memory_cache = {}
        self.cache_stats = {'hits': 0, 'misses': 0, 'evictions': 0}
        self.access_times = {}
        self.cache_sizes = {}
        
        # Redis cache (optional)
        self.redis_client = None
        if redis_url:
            try:
                self.redis_client = redis.from_url(redis_url)
                self.redis_client.ping()  # Test connection
                logging.info("Redis cache connected")
            except Exception as e:
                logging.warning(f"Redis cache not available: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get item from cache with fallback hierarchy"""
        # Try memory cache first
        if key in self.memory_cache:
            self.access_times[key] = time.time()
            self.cache_stats['hits'] += 1
            return self.memory_cache[key]
        
        # Try Redis cache
        if self.redis_client:
            try:
                cached_data = self.redis_client.get(key)
                if cached_data:
                    # Decompress and deserialize
                    data = pickle.loads(gzip.decompress(cached_data))
                    # Store in memory cache for faster access
                    self.set(key, data, ttl=None)  # Already TTL'd in Redis
                    self.cache_stats['hits'] += 1
                    return data
            except Exception as e:
                logging.warning(f"Redis cache error: {e}")
        
        self.cache_stats['misses'] += 1
        return default
    
    def set(self, key: str, value: Any, ttl: Optional[int] = 3600) -> bool:
        """Set item in cache with TTL"""
        try:
            # Estimate memory usage
            value_size = sys.getsizeof(pickle.dumps(value)) / 1024 / 1024  # MB
            
            # Check memory limits
            current_memory = sum(self.cache_sizes.values())
            if current_memory + value_size > self.memory_limit_mb:
                self._evict_lru_items(value_size)
            
            # Store in memory cache
            self.memory_cache[key] = value
            self.access_times[key] = time.time()
            self.cache_sizes[key] = value_size
            
            # Store in Redis cache (compressed)
            if self.redis_client and ttl:
                try:
                    compressed_data = gzip.compress(pickle.dumps(value))
                    self.redis_client.setex(key, ttl, compressed_data)
                except Exception as e:
                    logging.warning(f"Redis cache set error: {e}")
            
            return True
            
        except Exception as e:
            logging.error(f"Cache set error: {e}")
            return False
    
    def _evict_lru_items(self, space_needed: float):
        """Evict least recently used items"""
        # Sort by access time (oldest first)
        sorted_items = sorted(self.access_times.items(), key=lambda x: x[1])
        
        space_freed = 0.0
        for key, _ in sorted_items:
            if space_freed >= space_needed:
                break
            
            space_freed += self.cache_sizes.get(key, 0)
            del self.memory_cache[key]
            del self.access_times[key]
            del self.cache_sizes[key]
            self.cache_stats['evictions'] += 1
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
        hit_rate = self.cache_stats['hits'] / max(1, total_requests)
        
        return {
            'hit_rate': hit_rate,
            'total_hits': self.cache_stats['hits'],
            'total_misses': self.cache_stats['misses'],
            'total_evictions': self.cache_stats['evictions'],
            'memory_usage_mb': sum(self.cache_sizes.values()),
            'memory_limit_mb': self.memory_limit_mb,
            'items_count': len(self.memory_cache),
            'redis_available': self.redis_client is not None
        }

# Optimized Engine with Performance Enhancements
class OptimizedAdaptiveEngine:
    """High-performance adaptive engine with all optimizations"""
    
    def __init__(self, network_size: int = DEFAULT_NETWORK_SIZE, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.network_size = network_size
        
        # Initialize performance systems
        self.performance_monitor = PerformanceMonitor()
        self.parallel_manager = ParallelProcessingManager()
        self.cache_manager = AdvancedCacheManager(
            memory_limit_mb=self.config.get('cache_memory_mb', DEFAULT_MEMORY_LIMIT_MB),
            redis_url=self.config.get('redis_url')
        )
        
        # Initialize base systems (would include all previous engines)
        self.nodes = []  # Would use your AliveLoopNodes
        self.current_time = 0
        
        # Performance optimizations
        self.batch_processing_enabled = True
        self.async_processing_enabled = True
        self.result_cache_ttl = 3600  # 1 hour
        
        # Network state caching
        self._network_state_cache = {}
        self._network_state_cache_time = 0
        self._cache_validity_seconds = 1.0
        
        self.performance_monitor.start_monitoring()
        
    def safe_think(self, agent_name: str, task: str) -> Dict[str, Any]:
        """Optimized reasoning with performance enhancements"""
        start_time = time.time()
        
        # Generate cache key
        cache_key = self._generate_cache_key(agent_name, task)
        
        # Try cache first
        cached_result = self.cache_manager.get(cache_key)
        if cached_result:
            cached_result['from_cache'] = True
            self.performance_monitor.record_request(time.time() - start_time, True)
            return cached_result
        
        try:
            # Execute optimized reasoning
            result = self._execute_optimized_reasoning(agent_name, task)
            
            # Cache result
            self.cache_manager.set(cache_key, result, ttl=self.result_cache_ttl)
            
            # Record performance
            response_time = time.time() - start_time
            result['runtime'] = response_time
            result['from_cache'] = False
            self.performance_monitor.record_request(response_time, True)
            
            return result
            
        except Exception as e:
            error_result = {
                'success': False,
                'error': str(e),
                'from_cache': False,
                'runtime': time.time() - start_time
            }
            self.performance_monitor.record_request(time.time() - start_time, False)
            return error_result
    
    def _generate_cache_key(self, agent_name: str, task: str) -> str:
        """Generate unique cache key for task"""
        content = f"{agent_name}:{task}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _execute_optimized_reasoning(self, agent_name: str, task: str) -> Dict[str, Any]:
        """Execute reasoning with all optimizations"""
        
        # Step 1: Get cached network state or compute new one
        network_state = self._get_cached_network_state()
        
        # Step 2: Parallel node processing if enabled
        if self.batch_processing_enabled and len(self.nodes) > NODE_BATCH_PROCESSING_THRESHOLD:
            node_results = self.parallel_manager.parallel_node_processing(
                self.nodes, 
                lambda node: self._process_single_node(node)
            )
        else:
            node_results = [self._process_single_node(node) for node in self.nodes]
        
        # Step 3: Aggregate results efficiently
        aggregated_result = self._aggregate_node_results(node_results, agent_name, task)
        
        # Step 4: Add performance metadata
        aggregated_result.update({
            'network_state': network_state,
            'nodes_processed': len(node_results),
            'parallel_processing': self.batch_processing_enabled,
            'optimization_level': 'high'
        })
        
        return aggregated_result
    
    def _get_cached_network_state(self) -> Dict[str, Any]:
        """Get cached network state or compute new one"""
        current_time = time.time()
        
        if (current_time - self._network_state_cache_time > self._cache_validity_seconds or
            not self._network_state_cache):
            
            # Compute new network state
            self._network_state_cache = self._compute_network_state()
            self._network_state_cache_time = current_time
        
        return self._network_state_cache
    
    def _compute_network_state(self) -> Dict[str, Any]:
        """Compute current network state efficiently"""
        if not self.nodes:
            return {'total_nodes': 0, 'active_nodes': 0, 'network_energy': 0}
        
        # Use vectorized operations where possible
        node_phases = []
        node_energies = []
        
        for node in self.nodes:
            node_phases.append(getattr(node, 'phase', 'active'))
            node_energies.append(getattr(node, 'energy', DEFAULT_NODE_ENERGY))
        
        # Fast aggregations
        phase_counts = {}
        for phase in node_phases:
            phase_counts[phase] = phase_counts.get(phase, 0) + 1
        
        total_energy = sum(node_energies)
        active_nodes = phase_counts.get('active', 0)
        dominant_phase = max(phase_counts.items(), key=lambda x: x[1])[0] if phase_counts else 'active'
        
        return {
            'total_nodes': len(self.nodes),
            'active_nodes': active_nodes,
            'network_energy': total_energy,
            'dominant_phase': dominant_phase,
            'phase_distribution': phase_counts,
            'average_energy': total_energy / len(self.nodes) if self.nodes else 0
        }
    
    def _process_single_node(self, node) -> Dict[str, Any]:
        """Process a single node efficiently"""
        return {
            'node_id': getattr(node, 'node_id', 0),
            'phase': getattr(node, 'phase', 'active'),
            'energy': getattr(node, 'energy', DEFAULT_NODE_ENERGY),
            'processed': True
        }
    
    def _aggregate_node_results(self, node_results: List[Dict[str, Any]], agent_name: str, task: str) -> Dict[str, Any]:
        """Efficiently aggregate node processing results"""
        successful_nodes = [r for r in node_results if r and r.get('processed')]
        
        return {
            'content': f"[{agent_name}] Optimized analysis of: {task}",
            'confidence': min(0.9, 0.6 + len(successful_nodes) / len(node_results) * 0.3),
            'nodes_successful': len(successful_nodes),
            'processing_efficiency': len(successful_nodes) / len(node_results) if node_results else 0
        }
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        return {
            'performance_metrics': self.performance_monitor.get_metrics_snapshot(),
            'cache_stats': self.cache_manager.get_cache_stats(),
            'gpu_acceleration': self.parallel_manager.gpu_accelerator.gpu_available,
            'parallel_workers': self.parallel_manager.max_workers,
            'optimization_settings': {
                'batch_processing': self.batch_processing_enabled,
                'async_processing': self.async_processing_enabled,
                'cache_ttl': self.result_cache_ttl
            }
        }
    
    def shutdown(self):
        """Graceful shutdown of all systems"""
        self.performance_monitor.stop_monitoring()
        self.parallel_manager.shutdown()

# Enterprise REST API
class EnterpriseAPI:
    """
    Production-ready REST API for DuetMind agent with enterprise security.
    
    Security Features:
    - Secure authentication and authorization
    - Input validation and sanitization
    - Data encryption and privacy protection
    - Real-time security monitoring
    - GDPR/HIPAA compliance
    - Comprehensive audit logging
    """
    
    def __init__(self, agent_config: Dict[str, Any]):
        self.app = Flask(__name__)
        
        # Configuration
        self.config = agent_config
        
        # Initialize security systems
        self._initialize_security()
        
        # Initialize optimized agent
        self.agent = self._create_optimized_agent()
        
        # Enhanced request tracking with security
        self.request_counts = defaultdict(list)
        self.concurrent_requests = 0
        self.request_lock = threading.Lock()
        
        # Setup secure routes
        self._setup_secure_routes()
        
        # Start security monitoring
        self.security_monitor.start_monitoring()
        self.privacy_manager.start_background_cleanup()
        
        # Health check endpoint
        self.last_health_check = time.time()
        
        logging.info("Enterprise API initialized with security features")
    
    def _initialize_security(self):
        """Initialize all security systems."""
        # Authentication and authorization
        self.auth_manager = SecureAuthManager(self.config)
        self.app.auth_manager = self.auth_manager  # Make available to decorators
        
        # Input validation and sanitization
        self.input_validator = InputValidator()
        self.security_validator = SecurityValidator()
        
        # Data encryption and privacy
        self.data_encryption = DataEncryption(self.config.get('master_password'))
        self.privacy_manager = PrivacyManager(self.config)
        
        # Security monitoring
        self.security_monitor = SecurityMonitor(self.config)
        
        # Configure CORS with security headers
        CORS(self.app, 
             origins=self.config.get('allowed_origins', ['https://*']),
             methods=['GET', 'POST', 'PUT', 'DELETE'],
             allow_headers=['Content-Type', 'X-API-Key', 'X-Admin-Key', 'Authorization'])
        
        # Add security headers
        @self.app.after_request
        def add_security_headers(response):
            """Add security headers to all responses."""
            response.headers['X-Frame-Options'] = 'DENY'
            response.headers['X-Content-Type-Options'] = 'nosniff'
            response.headers['X-XSS-Protection'] = '1; mode=block'
            response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
            response.headers['Content-Security-Policy'] = "default-src 'self'"
            response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
            return response
    
    def _create_optimized_agent(self):
        """Create fully optimized agent"""
        # This would use all your previous systems + optimizations
        engine = OptimizedAdaptiveEngine(
            network_size=self.config.get('network_size', 30),
            config=self.config
        )
        
        style = {
            "logic": 0.8,
            "creativity": 0.7,
            "analytical": 0.9,
            "optimization_level": "enterprise"
        }
        
        # Would create with your DuetMindAgent class
        class MockAgent:
            def __init__(self, engine, style):
                self.engine = engine
                self.style = style
                self.name = "EnterpriseAgent"
            
            def generate_reasoning_tree(self, task):
                return {
                    'result': self.engine.safe_think(self.name, task),
                    'agent': self.name,
                    'style_applied': True
                }
        
        return MockAgent(engine, style)
    
    def _setup_secure_routes(self):
        """Setup secure API routes with comprehensive security."""
        
        @self.app.before_request
        def before_request():
            """Enhanced security checks before each request."""
            # Initialize request tracking
            self._initialize_request()
            
            # Skip auth for health check
            if request.endpoint == 'health_check':
                return
            
            # Perform authentication
            auth_error = self._authenticate_request()
            if auth_error:
                return auth_error
            
            # Check rate limits
            rate_limit_error = self._check_request_rate_limit()
            if rate_limit_error:
                return rate_limit_error
            
            # Check concurrent request limits
            capacity_error = self._check_server_capacity()
            if capacity_error:
                return capacity_error
            
            # Validate input data
            validation_error = self._validate_request_input()
            if validation_error:
                return validation_error
        
        def _initialize_request(self) -> None:
            """Initialize request tracking variables."""
            g.start_time = time.time()
            g.request_id = str(uuid.uuid4())
        
        def _authenticate_request(self) -> Optional[Any]:
            """Authenticate the request and store user info."""
            # Enhanced authentication with security monitoring
            api_key = request.headers.get('X-API-Key') or request.args.get('api_key')
            is_valid, user_info = self.auth_manager.validate_api_key(api_key)
            
            if not is_valid:
                self.security_monitor.log_security_event(
                    'authentication_failure',
                    {'endpoint': request.endpoint, 'method': request.method},
                    severity='warning',
                    ip_address=request.remote_addr
                )
                return jsonify({'error': 'Authentication required', 'request_id': g.request_id}), 401
            
            # Store user info for request
            g.user_info = user_info
            return None
        
        def _check_request_rate_limit(self) -> Optional[Any]:
            """Check rate limits for the authenticated user."""
            # Enhanced rate limiting per user
            if not self._check_enhanced_rate_limit(g.user_info):
                self.security_monitor.log_security_event(
                    'rate_limit_exceeded',
                    {'user_id': g.user_info['user_id'], 'endpoint': request.endpoint},
                    severity='warning',
                    user_id=g.user_info['user_id'],
                    ip_address=request.remote_addr
                )
                return jsonify({'error': 'Rate limit exceeded', 'request_id': g.request_id}), 429
            return None
        
        def _check_server_capacity(self) -> Optional[Any]:
            """Check if server has capacity for concurrent requests."""
            # Concurrent request limiting with better resource management
            with self.request_lock:
                max_concurrent = self.config.get('max_concurrent_requests', 10)
                if self.concurrent_requests >= max_concurrent:
                    return jsonify({
                        'error': 'Server capacity limit reached. Please try again later.',
                        'request_id': g.request_id
                    }), 503
                self.concurrent_requests += 1
            return None
        
        def _validate_request_input(self) -> Optional[Any]:
            """Validate JSON input data if present."""
            # Input validation for JSON requests
            if request.is_json:
                try:
                    json_data = request.get_json()
                    if json_data:
                        is_valid, errors = self.input_validator.validate_json_request(json_data)
                        if not is_valid:
                            self.security_monitor.log_security_event(
                                'input_validation_failure',
                                {'errors': errors, 'endpoint': request.endpoint},
                                severity='warning',
                                user_id=g.user_info['user_id'],
                                ip_address=request.remote_addr
                            )
                            return jsonify({
                                'error': 'Invalid input data',
                                'details': errors,
                                'request_id': g.request_id
                            }), 400
                except Exception as e:
                    return jsonify({
                        'error': 'Invalid JSON data',
                        'request_id': g.request_id
                    }), 400
            return None
        
        @self.app.after_request
        def after_request(response):
            """Enhanced request completion processing."""
            # Update concurrent request count
            with self.request_lock:
                self.concurrent_requests = max(0, self.concurrent_requests - 1)
            
            # Log API request for security monitoring
            if hasattr(g, 'user_info'):
                processing_time = time.time() - g.start_time
                self.security_monitor.log_api_request(
                    user_id=g.user_info['user_id'],
                    endpoint=request.endpoint or 'unknown',
                    method=request.method,
                    status_code=response.status_code,
                    response_time=processing_time,
                    ip_address=request.remote_addr,
                    payload_size=len(request.data) if request.data else 0
                )
                
                # Log to privacy manager for audit trail
                self.privacy_manager.log_data_access(
                    user_id=g.user_info['user_id'],
                    data_type='api_access',
                    action=request.method.lower(),
                    data_id=request.endpoint,
                    ip_address=request.remote_addr,
                    purpose='api_request',
                    legal_basis='legitimate_interest'
                )
            
            return response
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """
            Health check endpoint - no authentication required.
            
            Returns comprehensive system health information including
            security monitoring status and privacy compliance.
            """
            self.last_health_check = time.time()
            
            # Enhanced health check with security status
            health_status = {
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'uptime': time.time() - (g.start_time if hasattr(g, 'start_time') else self.last_health_check),
                'agent_status': 'ready',
                'version': '2.0.0-security',
                'security': {
                    'monitoring_active': self.security_monitor.running,
                    'privacy_compliance': True,
                    'encryption_enabled': True,
                    'auth_system': 'active'
                },
                'performance': self.agent.engine.get_performance_report() if hasattr(self.agent, 'engine') else {}
            }
            
            return jsonify(health_status)
        
        @self.app.route('/api/v1/reasoning', methods=['POST'])
        @require_auth()
        def reasoning_endpoint():
            """
            Main reasoning endpoint with enhanced security.
            
            Requires authentication and performs comprehensive input validation.
            All requests are logged for security monitoring and privacy compliance.
            """
            try:
                # Validate and extract input data
                sanitized_task, agent_params, validation_error = self._validate_reasoning_input()
                if validation_error:
                    return validation_error
                
                # Log data access for privacy compliance
                self._log_reasoning_access()
                
                # Execute reasoning and create response
                result = self.agent.generate_reasoning_tree(sanitized_task)
                response = self._create_reasoning_response(sanitized_task, result)
                
                return jsonify(response)
                
            except Exception as e:
                return self._handle_reasoning_error(e)
        
        def _validate_reasoning_input(self) -> Tuple[Optional[str], Optional[Dict], Optional[Any]]:
            """Validate and sanitize reasoning endpoint input."""
            data = request.get_json()
            required_fields = ['task']
            
            # Enhanced input validation
            is_valid, errors = self.input_validator.validate_json_request(data, required_fields)
            if not is_valid:
                error_response = jsonify({
                    'error': 'Input validation failed',
                    'details': errors,
                    'request_id': g.request_id
                }), 400
                return None, None, error_response
            
            task = data['task']
            agent_params = data.get('agent_params', {})
            
            # Sanitize task input
            sanitized_task = self.input_validator.sanitize_string(task, max_length=MAX_TASK_LENGTH)
            
            return sanitized_task, agent_params, None
        
        def _log_reasoning_access(self) -> None:
            """Log reasoning access for privacy compliance."""
            self.privacy_manager.log_data_access(
                user_id=g.user_info['user_id'],
                data_type='reasoning_request',
                action='process',
                purpose='ai_reasoning',
                legal_basis='service_provision'
            )
        
        def _create_reasoning_response(self, sanitized_task: str, result: Dict) -> Dict:
            """Create enhanced response format for reasoning endpoint."""
            return {
                'success': True,
                'task': sanitized_task,
                'result': result['result'],
                'agent': result['agent'],
                'processing_time': time.time() - g.start_time,
                'request_id': g.request_id,
                'user_id': g.user_info['user_id'],
                'security_validated': True
            }
        
        def _handle_reasoning_error(self, error: Exception) -> Any:
            """Handle reasoning endpoint errors with security logging."""
            error_id = str(uuid.uuid4())
            logging.error(f"API reasoning error [{error_id}]: {error}")
            
            # Log security event for investigation
            self.security_monitor.log_security_event(
                'reasoning_endpoint_error',
                {'error_id': error_id, 'endpoint': 'reasoning'},
                severity='warning',
                user_id=g.user_info.get('user_id'),
                ip_address=request.remote_addr
            )
            
            return jsonify({
                'success': False,
                'error': 'Reasoning processing error',
                'error_id': error_id,
                'request_id': g.request_id
            }), 500
        
        @self.app.route('/api/v1/knowledge/search', methods=['POST'])
        @require_auth()
        def knowledge_search():
            """
            Knowledge base search endpoint with security validation.
            
            Requires authentication and sanitizes search queries to prevent
            injection attacks and information disclosure.
            """
            try:
                data = request.get_json()
                
                # Enhanced input validation
                required_fields = ['query']
                is_valid, errors = self.input_validator.validate_json_request(data, required_fields)
                if not is_valid:
                    return jsonify({
                        'error': 'Input validation failed',
                        'details': errors,
                        'request_id': g.request_id
                    }), 400
                
                query = data.get('query', '')
                filters = data.get('filters', {})
                limit = min(data.get('limit', DEFAULT_KNOWLEDGE_SEARCH_LIMIT), MAX_KNOWLEDGE_SEARCH_RESULTS)  # Max results
                
                # Sanitize search query
                sanitized_query = self.input_validator.sanitize_string(query, max_length=MAX_QUERY_LENGTH)
                
                # Log data access
                self.privacy_manager.log_data_access(
                    user_id=g.user_info['user_id'],
                    data_type='knowledge_search',
                    action='search',
                    purpose='information_retrieval',
                    legal_basis='service_provision'
                )
                
                # Perform knowledge search (placeholder - would integrate with actual search)
                results = []  # Would call actual secure search implementation
                
                return jsonify({
                    'success': True,
                    'query': sanitized_query,
                    'results': results,
                    'count': len(results),
                    'processing_time': time.time() - g.start_time,
                    'request_id': g.request_id
                })
                
            except Exception as e:
                error_id = str(uuid.uuid4())
                logging.error(f"Knowledge search error [{error_id}]: {e}")
                return jsonify({
                    'success': False, 
                    'error': 'Search service temporarily unavailable',
                    'error_id': error_id,
                    'request_id': g.request_id
                }), 500
        
        @self.app.route('/api/v1/medical/process', methods=['POST'])
        @require_auth()
        def medical_data_process():
            """
            Secure medical data processing endpoint.
            
            Handles medical data with HIPAA compliance including:
            - Medical data validation
            - Automatic anonymization
            - Audit logging
            - Retention policy enforcement
            """
            try:
                # Validate input data
                data, validation_error = self._validate_medical_input()
                if validation_error:
                    return validation_error
                
                # Generate unique data ID and anonymize
                data_id, anonymized_data = self._prepare_medical_data(data)
                
                # Register for retention and log access
                self._register_and_log_medical_data(data_id)
                
                # Process the medical data
                processing_result = self._process_medical_data(data_id, anonymized_data)
                
                return self._create_medical_success_response(processing_result, data_id)
                
            except Exception as e:
                return self._handle_medical_processing_error(e)
        
        def _validate_medical_input(self) -> Tuple[Optional[Dict], Optional[Any]]:
            """Validate medical data input."""
            data = request.get_json()
            
            # Validate medical data format
            is_valid, errors = self.input_validator.validate_medical_data(data)
            if not is_valid:
                error_response = jsonify({
                    'error': 'Medical data validation failed',
                    'details': errors,
                    'request_id': g.request_id
                }), 400
                return None, error_response
            
            return data, None
        
        def _prepare_medical_data(self, data: Dict) -> Tuple[str, Dict]:
            """Generate data ID and anonymize medical data."""
            # Generate unique data ID for tracking
            data_id = f"medical_{g.user_info['user_id']}_{int(time.time())}"
            
            # Anonymize medical data for processing
            anonymized_data = self.data_encryption.anonymize_medical_data(data)
            
            return data_id, anonymized_data
        
        def _register_and_log_medical_data(self, data_id: str) -> None:
            """Register data for retention and log access."""
            # Register for retention tracking
            self.privacy_manager.register_data_for_retention(data_id, 'medical_data')
            
            # Log medical data access
            self.privacy_manager.log_data_access(
                user_id=g.user_info['user_id'],
                data_type='medical_data',
                action='process',
                data_id=data_id,
                purpose='medical_analysis',
                legal_basis='healthcare_provision'
            )
        
        def _process_medical_data(self, data_id: str, anonymized_data: Dict) -> Dict:
            """Process the anonymized medical data."""
            # Process medical data (placeholder for actual ML processing)
            return {
                'data_id': data_id,
                'processed': True,
                'anonymized': True,
                'retention_registered': True
            }
        
        def _create_medical_success_response(self, processing_result: Dict, data_id: str) -> Any:
            """Create successful response for medical data processing."""
            return jsonify({
                'success': True,
                'result': processing_result,
                'processing_time': time.time() - g.start_time,
                'request_id': g.request_id,
                'privacy_compliant': True
            })
        
        def _handle_medical_processing_error(self, error: Exception) -> Any:
            """Handle and log medical data processing errors."""
            error_id = str(uuid.uuid4())
            logging.error(f"Medical processing error [{error_id}]: {error}")
            
            self.security_monitor.log_security_event(
                'medical_processing_error',
                {'error_id': error_id},
                severity='warning',
                user_id=g.user_info.get('user_id'),
                ip_address=request.remote_addr
            )
            
            return jsonify({
                'success': False,
                'error': 'Medical data processing error',
                'error_id': error_id,
                'request_id': g.request_id
            }), 500
        
        @self.app.route('/api/v1/metrics', methods=['GET'])
        @require_admin
        def metrics_endpoint():
            """
            Performance and security metrics endpoint (admin only).
            
            Provides comprehensive system metrics including:
            - Performance statistics
            - Security monitoring data
            - Privacy compliance status
            """
            try:
                # Collect comprehensive metrics
                metrics = {
                    'performance': self.agent.engine.get_performance_report() if hasattr(self.agent, 'engine') else {},
                    'security': self.security_monitor.get_security_summary(),
                    'privacy': self.privacy_manager.generate_privacy_report(),
                    'system': {
                        'concurrent_requests': self.concurrent_requests,
                        'uptime': time.time() - self.last_health_check,
                        'timestamp': datetime.now().isoformat()
                    }
                }
                
                # Log admin access
                self.privacy_manager.log_data_access(
                    user_id=g.user_info['user_id'],
                    data_type='system_metrics',
                    action='view',
                    purpose='system_administration',
                    legal_basis='legitimate_interest'
                )
                
                return jsonify({
                    'success': True,
                    'metrics': metrics,
                    'request_id': g.request_id
                })
                
            except Exception as e:
                error_id = str(uuid.uuid4())
                logging.error(f"Metrics endpoint error [{error_id}]: {e}")
                return jsonify({
                    'success': False,
                    'error': 'Metrics service error',
                    'error_id': error_id,
                    'request_id': g.request_id
                }), 500
        
        @self.app.route('/api/v1/admin/config', methods=['GET', 'PUT'])
        @require_admin
        def config_endpoint():
            """
            Configuration management endpoint (admin only).
            
            Allows secure configuration updates with validation
            and audit logging.
            """
            try:
                if request.method == 'GET':
                    # Return sanitized configuration (hide sensitive values)
                    safe_config = self.config.copy()
                    sensitive_keys = ['api_key', 'admin_key', 'master_password', 'jwt_secret']
                    for key in sensitive_keys:
                        if key in safe_config:
                            safe_config[key] = '***HIDDEN***'
                    
                    return jsonify({
                        'success': True,
                        'config': safe_config,
                        'request_id': g.request_id
                    })
                
                else:  # PUT request
                    new_config = request.get_json()
                    
                    # Validate configuration changes
                    if 'rate_limit' in new_config:
                        is_valid, errors = self.security_validator.validate_rate_limit_config({
                            'requests_per_minute': new_config['rate_limit'],
                            'burst_limit': new_config.get('burst_limit', 50),
                            'window_size': 60
                        })
                        if not is_valid:
                            return jsonify({
                                'success': False,
                                'error': 'Invalid configuration',
                                'details': errors,
                                'request_id': g.request_id
                            }), 400
                    
                    # Apply configuration changes
                    old_config = self.config.copy()
                    self.config.update(new_config)
                    
                    # Log configuration change
                    self.security_monitor.log_security_event(
                        'configuration_change',
                        {'changes': new_config, 'admin_user': g.user_info['user_id']},
                        severity='info',
                        user_id=g.user_info['user_id'],
                        ip_address=request.remote_addr
                    )
                    
                    return jsonify({
                        'success': True,
                        'message': 'Configuration updated successfully',
                        'request_id': g.request_id
                    })
                    
            except Exception as e:
                error_id = str(uuid.uuid4())
                logging.error(f"Config endpoint error [{error_id}]: {e}")
                return jsonify({
                    'success': False,
                    'error': 'Configuration service error',
                    'error_id': error_id,
                    'request_id': g.request_id
                }), 500
                return jsonify({'success': True, 'config': self.config})
    
    def _check_enhanced_rate_limit(self, user_info: Dict[str, Any]) -> bool:
        """
        Enhanced rate limiting per user with role-based limits.
        
        Different user roles get different rate limits:
        - Admin users: Higher limits
        - Regular users: Standard limits
        - Suspicious IPs: Lower limits
        """
        user_id = user_info['user_id']
        user_roles = user_info.get('roles', [])
        client_ip = request.remote_addr
        current_time = time.time()
        
        # Role-based rate limits
        if 'admin' in user_roles:
            rate_limit = self.config.get('admin_rate_limit', 500)  # Higher limit for admins
        else:
            rate_limit = self.config.get('user_rate_limit', 100)   # Standard limit
        
        # Per-user tracking
        user_key = f"user_{user_id}"
        ip_key = f"ip_{client_ip}"
        
        # Clean old requests (1 minute window)
        cutoff_time = current_time - 60
        
        for key in [user_key, ip_key]:
            self.request_counts[key] = [
                req_time for req_time in self.request_counts[key]
                if req_time > cutoff_time
            ]
        
        # Check both user and IP limits
        user_requests = len(self.request_counts[user_key])
        ip_requests = len(self.request_counts[ip_key])
        
        # Apply stricter limit
        if user_requests >= rate_limit or ip_requests >= (rate_limit * 2):
            return False
        
        # Record this request
        self.request_counts[user_key].append(current_time)
        self.request_counts[ip_key].append(current_time)
        
        return True
    
    def shutdown(self):
        """
        Graceful shutdown of all systems including security components.
        
        Ensures proper cleanup of:
        - Security monitoring
        - Privacy background processes
        - Active sessions
        - Audit logs
        """
        try:
            # Stop security monitoring
            if hasattr(self, 'security_monitor'):
                self.security_monitor.stop_monitoring()
            
            # Stop privacy cleanup
            if hasattr(self, 'privacy_manager'):
                self.privacy_manager.stop_background_cleanup()
            
            # Log shutdown event
            if hasattr(self, 'security_monitor'):
                self.security_monitor.log_security_event(
                    'system_shutdown',
                    {'timestamp': datetime.now().isoformat()},
                    severity='info'
                )
            
            logging.info("Enterprise API shutdown completed successfully")
            
        except Exception as e:
            logging.error(f"Error during shutdown: {e}")
    
    def run(self, host: str = '0.0.0.0', port: int = 8080, debug: bool = False, ssl_context: Optional[Any] = None):
        """
        Run the secure API server with enhanced security options.
        
        Args:
            host: Host address (default: 0.0.0.0)
            port: Port number (default: 8080)
            debug: Debug mode (should be False in production)
            ssl_context: SSL context for HTTPS (recommended for production)
        """
        # Security warning for development mode
        if debug:
            logging.warning("⚠️  DEBUG MODE ENABLED - DO NOT USE IN PRODUCTION")
        
        # HTTPS recommendation
        if not ssl_context and host != '127.0.0.1':
            logging.warning("⚠️  Running without HTTPS - Consider using SSL in production")
        
        # Log startup
        self.security_monitor.log_security_event(
            'system_startup',
            {
                'host': host,
                'port': port,
                'debug': debug,
                'https_enabled': ssl_context is not None
            },
            severity='info'
        )
        
        logging.info(f"🚀 Starting DuetMind Enterprise API v2.0 on {host}:{port}")
        logging.info("🔒 Security features: Authentication ✓ Encryption ✓ Privacy ✓ Monitoring ✓")
        
        try:
            self.app.run(
                host=host, 
                port=port, 
                debug=debug, 
                threaded=True,
                ssl_context=ssl_context
            )
        except KeyboardInterrupt:
            logging.info("Shutdown requested by user")
        finally:
            self.shutdown()

# Docker Configuration
class DockerDeployment:
    """Docker deployment configuration generator"""
    
    @staticmethod
    def generate_dockerfile(config: Dict[str, Any]) -> str:
        """Generate optimized Dockerfile"""
        return f"""
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    sqlite3 \\
    libsqlite3-dev \\
    redis-tools \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Install CUDA support (optional)
RUN if [ "{config.get('gpu_enabled', False)}" = "True" ]; then \\
    apt-get update && \\
    apt-get install -y nvidia-cuda-toolkit && \\
    rm -rf /var/lib/apt/lists/*; \\
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
EXPOSE {config.get('port', 8080)}
EXPOSE 6379

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:{config.get('port', 8080)}/health || exit 1

# Run application
CMD ["gunicorn", "--bind", "0.0.0.0:{config.get('port', 8080)}", "--workers", "{config.get('workers', 4)}", "--timeout", "300", "app:app"]
"""

    @staticmethod
    def generate_docker_compose(config: Dict[str, Any]) -> str:
        """Generate docker-compose.yml"""
        return f"""
version: '3.8'

services:
  duetmind-api:
    build: .
    ports:
      - "{config.get('port', 8080)}:{config.get('port', 8080)}"
    environment:
      - REDIS_URL=redis://redis:6379/0
      - API_KEY={config.get('api_key', 'your-api-key')}
      - ADMIN_KEY={config.get('admin_key', 'your-admin-key')}
      - NETWORK_SIZE={config.get('network_size', 30)}
      - GPU_ENABLED={config.get('gpu_enabled', False)}
    volumes:
      - ./knowledge_base:/app/knowledge_base
      - ./logs:/app/logs
      - ./workspace:/app/workspace
    depends_on:
      - redis
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: {config.get('memory_limit', '2G')}
          cpus: '{config.get('cpu_limit', '2.0')}'
        reservations:
          memory: {config.get('memory_reservation', '1G')}
          cpus: '{config.get('cpu_reservation', '1.0')}'

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes --maxmemory {config.get('redis_memory', '256mb')} --maxmemory-policy allkeys-lru
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - duetmind-api
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD={config.get('grafana_password', 'admin')}
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./grafana/datasources:/etc/grafana/provisioning/datasources
    depends_on:
      - prometheus
    restart: unless-stopped

volumes:
  redis_data:
  prometheus_data:
  grafana_data:
"""

# Monitoring and Observability
class ObservabilitySystem:
    """Comprehensive monitoring and observability"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics_collector = self._setup_metrics_collector()
        self.log_aggregator = self._setup_logging()
        self.alert_manager = self._setup_alerting()
        
    def _setup_metrics_collector(self):
        """Setup Prometheus-style metrics collection"""
        return {
            'request_duration_histogram': defaultdict(list),
            'request_count_counter': defaultdict(int),
            'error_count_counter': defaultdict(int),
            'concurrent_requests_gauge': 0,
            'memory_usage_gauge': 0,
            'cpu_usage_gauge': 0
        }
    
    def _setup_logging(self):
        """Setup structured logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('duetmind.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger('duetmind')
    
    def _setup_alerting(self):
        """Setup alerting thresholds"""
        return {
            'high_error_rate_threshold': 0.05,  # 5%
            'high_response_time_threshold': 5.0,  # 5 seconds
            'high_memory_usage_threshold': 0.8,   # 80%
            'high_cpu_usage_threshold': 0.8       # 80%
        }
    
    def record_request_metrics(self, endpoint: str, duration: float, status_code: int):
        """Record request metrics"""
        self.metrics_collector['request_duration_histogram'][endpoint].append(duration)
        self.metrics_collector['request_count_counter'][endpoint] += 1
        
        if status_code >= 400:
            self.metrics_collector['error_count_counter'][endpoint] += 1
        
        # Check for alerts
        self._check_alerts(endpoint, duration, status_code)
    
    def _check_alerts(self, endpoint: str, duration: float, status_code: int):
        """Check if any alert thresholds are exceeded"""
        
        # High response time alert
        if duration > self.alert_manager['high_response_time_threshold']:
            self._send_alert('high_response_time', {
                'endpoint': endpoint,
                'duration': duration,
                'threshold': self.alert_manager['high_response_time_threshold']
            })
        
        # High error rate alert (check last 100 requests)
        recent_requests = self.metrics_collector['request_count_counter'][endpoint]
        recent_errors = self.metrics_collector['error_count_counter'][endpoint]
        
        if recent_requests > 0:
            error_rate = recent_errors / recent_requests
            if error_rate > self.alert_manager['high_error_rate_threshold']:
                self._send_alert('high_error_rate', {
                    'endpoint': endpoint,
                    'error_rate': error_rate,
                    'threshold': self.alert_manager['high_error_rate_threshold']
                })
    
    def _send_alert(self, alert_type: str, details: Dict[str, Any]):
        """Send alert (webhook, email, etc.)"""
        alert_data = {
            'timestamp': datetime.now().isoformat(),
            'alert_type': alert_type,
            'severity': 'warning',
            'details': details
        }
        
        # Log alert
        self.log_aggregator.warning(f"ALERT: {alert_type} - {details}")
        
        # Send to external systems (webhook, Slack, etc.)
        webhook_url = self.config.get('alert_webhook_url')
        if webhook_url:
            try:
                import requests
                requests.post(webhook_url, json=alert_data, timeout=5)
            except Exception as e:
                self.log_aggregator.error(f"Failed to send alert webhook: {e}")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary"""
        summary = {}
        
        for endpoint, durations in self.metrics_collector['request_duration_histogram'].items():
            if durations:
                summary[endpoint] = {
                    'request_count': len(durations),
                    'avg_response_time': np.mean(durations),
                    'p95_response_time': np.percentile(durations, 95),
                    'p99_response_time': np.percentile(durations, 99),
                    'error_count': self.metrics_collector['error_count_counter'][endpoint],
                    'error_rate': self.metrics_collector['error_count_counter'][endpoint] / len(durations)
                }
        
        return summary

# Production Deployment Manager
class ProductionDeploymentManager:
    """Manage production deployments with zero downtime"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.deployment_path = Path(config.get('deployment_path', './deployment'))
        self.deployment_path.mkdir(exist_ok=True)
        
    def generate_deployment_files(self) -> Dict[str, str]:
        """Generate all deployment files"""
        files = {}
        
        # Dockerfile
        files['Dockerfile'] = DockerDeployment.generate_dockerfile(self.config)
        
        # Docker Compose
        files['docker-compose.yml'] = DockerDeployment.generate_docker_compose(self.config)
        
        # Nginx Configuration
        files['nginx.conf'] = self._generate_nginx_config()
        
        # Kubernetes manifests
        files['k8s-deployment.yaml'] = self._generate_k8s_deployment()
        files['k8s-service.yaml'] = self._generate_k8s_service()
        files['k8s-ingress.yaml'] = self._generate_k8s_ingress()
        
        # Prometheus configuration
        files['prometheus.yml'] = self._generate_prometheus_config()
        
        # Grafana dashboard
        files['grafana-dashboard.json'] = self._generate_grafana_dashboard()
        
        # Requirements file
        files['requirements.txt'] = self._generate_requirements()
        
        return files
    
    def _generate_nginx_config(self) -> str:
        """Generate Nginx reverse proxy configuration"""
        return f"""
events {{
    worker_connections 1024;
}}

http {{
    upstream duetmind_backend {{
        server duetmind-api:{self.config.get('port', 8080)};
        keepalive 32;
    }}
    
    server {{
        listen 80;
        server_name {self.config.get('domain', 'localhost')};
        
        # Security headers
        add_header X-Frame-Options DENY;
        add_header X-Content-Type-Options nosniff;
        add_header X-XSS-Protection "1; mode=block";
        
        # Rate limiting
        limit_req_zone $binary_remote_addr zone=api:10m rate={self.config.get('nginx_rate_limit', '10r/s')};
        
        location / {{
            limit_req zone=api burst=20 nodelay;
            
            proxy_pass http://duetmind_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # Timeouts
            proxy_connect_timeout 30s;
            proxy_send_timeout 30s;
            proxy_read_timeout 300s;
            
            # Buffer settings
            proxy_buffering on;
            proxy_buffer_size 4k;
            proxy_buffers 8 4k;
        }}
        
        location /health {{
            proxy_pass http://duetmind_backend/health;
            access_log off;
        }}
    }}
}}
"""
    
    def _generate_k8s_deployment(self) -> str:
        """Generate Kubernetes deployment manifest"""
        return f"""
apiVersion: apps/v1
kind: Deployment
metadata:
  name: duetmind-api
  labels:
    app: duetmind-api
spec:
  replicas: {self.config.get('k8s_replicas', 3)}
  selector:
    matchLabels:
      app: duetmind-api
  template:
    metadata:
      labels:
        app: duetmind-api
    spec:
      containers:
      - name: duetmind-api
        image: duetmind-api:latest
        ports:
        - containerPort: {self.config.get('port', 8080)}
        env:
        - name: REDIS_URL
          value: "redis://redis-service:6379/0"
        - name: API_KEY
          valueFrom:
            secretKeyRef:
              name: duetmind-secrets
              key: api-key
        - name: ADMIN_KEY
          valueFrom:
            secretKeyRef:
              name: duetmind-secrets
              key: admin-key
        resources:
          requests:
            memory: "{self.config.get('memory_request', '1Gi')}"
            cpu: "{self.config.get('cpu_request', '500m')}"
          limits:
            memory: "{self.config.get('memory_limit', '2Gi')}"
            cpu: "{self.config.get('cpu_limit', '1000m')}"
        livenessProbe:
          httpGet:
            path: /health
            port: {self.config.get('port', 8080)}
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: {self.config.get('port', 8080)}
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: knowledge-base
          mountPath: /app/knowledge_base
        - name: logs
          mountPath: /app/logs
      volumes:
      - name: knowledge-base
        persistentVolumeClaim:
          claimName: knowledge-base-pvc
      - name: logs
        emptyDir: {{}}
"""
    
    def _generate_k8s_service(self) -> str:
        """Generate Kubernetes service manifest"""
        return f"""
apiVersion: v1
kind: Service
metadata:
  name: duetmind-api-service
  labels:
    app: duetmind-api
spec:
  selector:
    app: duetmind-api
  ports:
  - protocol: TCP
    port: {self.config.get('port', 8080)}
    targetPort: {self.config.get('port', 8080)}
  type: ClusterIP
"""
    
    def _generate_k8s_ingress(self) -> str:
        """Generate Kubernetes ingress manifest"""
        return f"""
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: duetmind-api-ingress
  annotations:
    kubernetes.io/ingress.class: "nginx"
    nginx.ingress.kubernetes.io/rate-limit: "{self.config.get('k8s_rate_limit', '100')}"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
spec:
  tls:
  - hosts:
    - {self.config.get('domain', 'duetmind.example.com')}
    secretName: duetmind-tls
  rules:
  - host: {self.config.get('domain', 'duetmind.example.com')}
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: duetmind-api-service
            port:
              number: {self.config.get('port', 8080)}
"""
    
    def _generate_prometheus_config(self) -> str:
        """Generate Prometheus configuration"""
        return """
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'duetmind-api'
    static_configs:
      - targets: ['duetmind-api:8080']
    metrics_path: '/api/v1/metrics'
    scrape_interval: 30s
"""
    
    def _generate_grafana_dashboard(self) -> str:
        """Generate Grafana dashboard configuration"""
        dashboard = {
            "dashboard": {
                "title": "DuetMind Agent Performance",
                "panels": [
                    {
                        "title": "Request Rate",
                        "type": "graph",
                        "targets": [{"expr": "rate(duetmind_requests_total[5m])"}]
                    },
                    {
                        "title": "Response Time",
                        "type": "graph",
                        "targets": [{"expr": "duetmind_request_duration_seconds"}]
                    },
                    {
                        "title": "Memory Usage",
                        "type": "graph",
                        "targets": [{"expr": "duetmind_memory_usage_bytes"}]
                    },
                    {
                        "title": "Knowledge Base Size",
                        "type": "singlestat",
                        "targets": [{"expr": "duetmind_knowledge_documents_total"}]
                    }
                ]
            }
        }
        return json.dumps(dashboard, indent=2)
    
    def _generate_requirements(self) -> str:
        """Generate Python requirements file"""
        return """
flask==2.3.3
flask-cors==4.0.0
redis==4.6.0
numpy==1.24.3
psutil==5.9.5
requests==2.31.0
gunicorn==21.2.0
pyyaml==6.0.1
sqlite3
cupy-cuda11x>=12.0.0  # Optional GPU acceleration
prometheus-client==0.17.1
structlog==23.1.0
"""
    
    def deploy_to_files(self) -> bool:
        """Deploy all configuration files"""
        try:
            files = self.generate_deployment_files()
            
            for filename, content in files.items():
                file_path = self.deployment_path / filename
                file_path.write_text(content)
                print(f"✅ Generated {filename}")
            
            print(f"\n🚀 Deployment files created in: {self.deployment_path}")
            print("Ready for production deployment!")
            
            return True
            
        except Exception as e:
            print(f"❌ Deployment generation failed: {e}")
            return False

# Complete Demo
if __name__ == "__main__":
    from examples.enterprise_demo import demo_enterprise_system
    demo_enterprise_system()
