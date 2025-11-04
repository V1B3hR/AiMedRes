"""
Secure API Server for DuetMind Adaptive

Production-ready Flask server with comprehensive security features:
- Rate limiting and DDoS protection  
- Security headers and HTTPS enforcement
- Authentication and authorization
- Input validation and sanitization
- Audit logging and monitoring
"""

import logging
import ssl
import time
from datetime import datetime, timedelta
from functools import wraps
from typing import Dict, Any, Optional

from flask import Flask, request, jsonify, g
from flask_cors import CORS
import redis
from werkzeug.exceptions import TooManyRequests

from ..core.config import DuetMindConfig
from ..core.agent import DuetMindAgent
from ..core.neural_network import AdaptiveNeuralNetwork
from ..security.auth import SecureAuthManager, require_auth
from ..security.validation import InputValidator
from ..utils.safety import SafetyMonitor

logger = logging.getLogger(__name__)

class RateLimiter:
    """
    Rate limiting with Redis backend for scalability
    """
    
    def __init__(self, redis_client=None):
        self.redis = redis_client
        self.local_cache = {}  # Fallback for when Redis unavailable
        
    def is_rate_limited(self, key: str, limit: int, window: int) -> bool:
        """
        Check if key is rate limited
        
        Args:
            key: Identifier for rate limiting (IP, user ID, etc.)
            limit: Maximum requests allowed
            window: Time window in seconds
            
        Returns:
            True if rate limited
        """
        now = time.time()
        
        try:
            if self.redis:
                # Use Redis sliding window
                pipe = self.redis.pipeline()
                pipe.zremrangebyscore(key, 0, now - window)
                pipe.zcard(key)
                pipe.zadd(key, {str(now): now})
                pipe.expire(key, window)
                results = pipe.execute()
                
                current_requests = results[1]
                return current_requests >= limit
            else:
                # Fallback to local cache
                if key not in self.local_cache:
                    self.local_cache[key] = []
                
                # Clean old requests
                self.local_cache[key] = [
                    req_time for req_time in self.local_cache[key] 
                    if req_time > now - window
                ]
                
                # Check limit
                if len(self.local_cache[key]) >= limit:
                    return True
                
                # Add current request
                self.local_cache[key].append(now)
                return False
                
        except Exception as e:
            logger.error(f"Rate limiting error: {e}")
            return False  # Fail open for availability

def rate_limit(limit: int = 100, window: int = 60, key_func=None):
    """
    Rate limiting decorator
    
    Args:
        limit: Maximum requests per window
        window: Time window in seconds
        key_func: Function to generate rate limit key
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Get rate limiter from app
            rate_limiter = getattr(g, 'rate_limiter', None)
            if not rate_limiter:
                return f(*args, **kwargs)  # Skip if not configured
            
            # Generate rate limit key
            if key_func:
                key = key_func()
            else:
                key = f"rate_limit:{request.remote_addr}"
            
            # Check rate limit
            if rate_limiter.is_rate_limited(key, limit, window):
                logger.warning(f"Rate limit exceeded for {key}")
                raise TooManyRequests(f"Rate limit exceeded: {limit} requests per {window} seconds")
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

class SecureAPIServer:
    """
    Production-ready secure API server for DuetMind
    """
    
    def __init__(self, config: Optional[DuetMindConfig] = None):
        self.config = config or DuetMindConfig()
        self.app = Flask(__name__)
        
        # Security components
        self.auth_manager = SecureAuthManager(self.config)
        self.input_validator = InputValidator()
        self.safety_monitor = SafetyMonitor()
        
        # Rate limiting
        try:
            redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
            redis_client.ping()  # Test connection
            self.rate_limiter = RateLimiter(redis_client)
            logger.info("Redis rate limiting enabled")
        except:
            self.rate_limiter = RateLimiter()  # Local fallback
            logger.warning("Using local rate limiting (Redis unavailable)")
        
        # AI components  
        self.agents: Dict[str, DuetMindAgent] = {}
        self.neural_networks: Dict[str, AdaptiveNeuralNetwork] = {}
        
        # Configure Flask app
        self._configure_app()
        self._setup_security()
        self._register_routes()
        
        logger.info("Secure API server initialized")
    
    def _configure_app(self):
        """Configure Flask application settings"""
        self.app.config.update({
            'SECRET_KEY': self.config.get_secret('FLASK_SECRET_KEY') or 'dev-secret-change-in-production',
            'MAX_CONTENT_LENGTH': self.config.api.max_request_size,
            'JSON_SORT_KEYS': False,
            'JSONIFY_PRETTYPRINT_REGULAR': False
        })
        
        # CORS configuration
        if self.config.api.cors_enabled:
            CORS(self.app, 
                 origins=self.config.security.allowed_origins,
                 methods=['GET', 'POST', 'PUT', 'DELETE'],
                 allow_headers=['Content-Type', 'Authorization'])
    
    def _setup_security(self):
        """Setup comprehensive security middleware"""
        
        @self.app.before_request
        def before_request():
            """Security checks before each request"""
            # Add rate limiter to request context
            g.rate_limiter = self.rate_limiter
            
            # Safety check
            if not self.safety_monitor.is_safe_to_operate():
                logger.error("API unavailable due to safety concerns")
                return jsonify({"error": "Service temporarily unavailable"}), 503
            
            # Request size validation
            if request.content_length and request.content_length > self.config.api.max_request_size:
                return jsonify({"error": "Request too large"}), 413
            
            # Log request for audit
            logger.info(f"API Request: {request.method} {request.path} from {request.remote_addr}")
        
        @self.app.after_request  
        def after_request(response):
            """Add security headers to all responses"""
            response.headers.update({
                'X-Frame-Options': 'DENY',
                'X-Content-Type-Options': 'nosniff',
                'X-XSS-Protection': '1; mode=block',
                'Referrer-Policy': 'strict-origin-when-cross-origin',
                'Content-Security-Policy': "default-src 'self'",
                'Strict-Transport-Security': 'max-age=31536000; includeSubDomains'
            })
            return response
        
        @self.app.errorhandler(400)
        def bad_request(error):
            logger.warning(f"Bad request from {request.remote_addr}: {error}")
            return jsonify({"error": "Bad request"}), 400
        
        @self.app.errorhandler(401)
        def unauthorized(error):
            logger.warning(f"Unauthorized access from {request.remote_addr}")
            return jsonify({"error": "Unauthorized"}), 401
        
        @self.app.errorhandler(403)
        def forbidden(error):
            logger.warning(f"Forbidden access from {request.remote_addr}")
            return jsonify({"error": "Forbidden"}), 403
        
        @self.app.errorhandler(429)
        def rate_limited(error):
            return jsonify({"error": str(error)}), 429
        
        @self.app.errorhandler(500)
        def internal_error(error):
            logger.error(f"Internal error: {error}")
            return jsonify({"error": "Internal server error"}), 500
    
    def _register_routes(self):
        """Register API routes with security"""
        
        # Register P3 blueprints for advanced features
        try:
            from .visualization_routes import visualization_bp
            self.app.register_blueprint(visualization_bp)
            logger.info("Visualization routes registered (P3-1)")
        except Exception as e:
            logger.warning(f"Failed to register visualization routes: {e}")
        
        try:
            from .canary_routes import canary_bp
            self.app.register_blueprint(canary_bp)
            logger.info("Canary deployment routes registered (P3-3)")
        except Exception as e:
            logger.warning(f"Failed to register canary routes: {e}")
        
        try:
            from .quantum_routes import quantum_bp
            self.app.register_blueprint(quantum_bp)
            logger.info("Quantum crypto routes registered (P3-2)")
        except Exception as e:
            logger.warning(f"Failed to register quantum routes: {e}")
        
        @self.app.route('/health', methods=['GET'])
        @rate_limit(limit=30, window=60)  # 30 requests per minute
        def health_check():
            """Health check endpoint"""
            status = {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "version": "1.0.0",
                "safety": self.safety_monitor.get_status() if self.safety_monitor else {"status": "disabled"}
            }
            return jsonify(status)
        
        @self.app.route('/api/v1/predict', methods=['POST'])
        @rate_limit(limit=60, window=60)  # 60 requests per minute
        @require_auth
        def predict():
            """Secure prediction endpoint"""
            try:
                # Validate input
                data = request.get_json()
                if not data or 'input' not in data:
                    return jsonify({"error": "Missing 'input' field"}), 400
                
                # Additional validation
                if not self.input_validator.validate_json(data):
                    return jsonify({"error": "Invalid input data"}), 400
                
                # Get or create neural network
                network_id = data.get('network_id', 'default')
                if network_id not in self.neural_networks:
                    self.neural_networks[network_id] = AdaptiveNeuralNetwork()
                
                network = self.neural_networks[network_id]
                
                # Make prediction
                import numpy as np
                input_array = np.array(data['input'])
                prediction = network.predict(input_array)
                
                # Record operation
                self.safety_monitor.record_operation(
                    operation="prediction",
                    duration=0.1,  # Placeholder
                    success=True
                )
                
                result = {
                    "prediction": prediction.tolist(),
                    "network_id": network_id,
                    "timestamp": datetime.now().isoformat(),
                    "network_health": network.get_network_health()
                }
                
                return jsonify(result)
                
            except Exception as e:
                logger.error(f"Prediction error: {e}")
                
                self.safety_monitor.record_operation(
                    operation="prediction",
                    duration=0.1,
                    success=False,
                    error=str(e)
                )
                
                return jsonify({"error": "Prediction failed"}), 500
        
        @self.app.route('/api/v1/agent/think', methods=['POST'])
        @rate_limit(limit=30, window=60)  # 30 requests per minute
        @require_auth
        def agent_think():
            """Secure agent thinking endpoint"""
            try:
                data = request.get_json()
                if not data or 'input' not in data:
                    return jsonify({"error": "Missing 'input' field"}), 400
                
                # Get or create agent
                agent_id = data.get('agent_id', 'default')
                if agent_id not in self.agents:
                    self.agents[agent_id] = DuetMindAgent(
                        agent_id=agent_id,
                        name=f"Agent-{agent_id}"
                    )
                
                agent = self.agents[agent_id]
                
                # Agent thinking
                import numpy as np
                input_array = np.array(data['input'])
                thought = agent.think(input_array)
                
                return jsonify(thought)
                
            except Exception as e:
                logger.error(f"Agent thinking error: {e}")
                return jsonify({"error": "Agent thinking failed"}), 500
        
        @self.app.route('/api/v1/status', methods=['GET'])
        @rate_limit(limit=20, window=60)  # 20 requests per minute  
        @require_auth
        def system_status():
            """System status endpoint"""
            try:
                status = {
                    "agents": len(self.agents),
                    "neural_networks": len(self.neural_networks),
                    "safety_monitor": self.safety_monitor.get_status(),
                    "timestamp": datetime.now().isoformat()
                }
                
                return jsonify(status)
                
            except Exception as e:
                logger.error(f"Status error: {e}")
                return jsonify({"error": "Status unavailable"}), 500
    
    def start_safety_monitoring(self):
        """Start safety monitoring"""
        self.safety_monitor.start()
        logger.info("API safety monitoring started")
    
    def stop_safety_monitoring(self):
        """Stop safety monitoring"""
        self.safety_monitor.stop()
        logger.info("API safety monitoring stopped")
    
    def run(self, host=None, port=None, debug=None, ssl_context=None):
        """
        Run the secure API server
        
        Args:
            host: Host address
            port: Port number
            debug: Debug mode
            ssl_context: SSL context for HTTPS
        """
        host = host or self.config.api.host
        port = port or self.config.api.port
        debug = debug if debug is not None else self.config.api.debug
        
        # Setup SSL if enabled
        if self.config.api.ssl_enabled and ssl_context is None:
            if self.config.api.ssl_cert_path and self.config.api.ssl_key_path:
                ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
                ssl_context.load_cert_chain(
                    self.config.api.ssl_cert_path,
                    self.config.api.ssl_key_path
                )
                logger.info("HTTPS enabled with SSL certificates")
            else:
                logger.warning("SSL enabled but no certificates configured, using HTTP")
                ssl_context = None
        
        # Start safety monitoring
        self.start_safety_monitoring()
        
        try:
            logger.info(f"Starting secure API server on {host}:{port}")
            self.app.run(
                host=host,
                port=port,
                debug=debug,
                ssl_context=ssl_context,
                threaded=True
            )
        finally:
            self.stop_safety_monitoring()
    
    def shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down API server")
        
        # Shutdown all agents
        for agent in self.agents.values():
            agent.shutdown()
        
        # Shutdown neural networks
        for network in self.neural_networks.values():
            network.shutdown()
        
        # Stop safety monitoring
        self.stop_safety_monitoring()
        
        logger.info("API server shutdown complete")

def create_app(config: Optional[DuetMindConfig] = None) -> Flask:
    """Factory function to create Flask app"""
    server = SecureAPIServer(config)
    return server.app

if __name__ == "__main__":
    # Development server
    server = SecureAPIServer()
    server.run(debug=True)