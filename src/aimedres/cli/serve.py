#!/usr/bin/env python3
"""
Secure Remote Training API Server for DuetMind Adaptive System

Provides a production-ready Flask API server with:
- Comprehensive security and authentication
- Remote model training endpoints
- Real-time training monitoring
- Encrypted model parameter exchange
- System health monitoring
"""

import os
import sys
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
import argparse

# Add project root to path for imports
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import security and training components
from security.auth import SecureAuthManager
from security.encryption import DataEncryption
from remote_training_manager import RemoteTrainingManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('duetmind.api_server')

def create_app(config=None):
    """
    Create and configure Flask application.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Configured Flask application
    """
    app = Flask(__name__)
    
    # Default configuration
    default_config = {
        'SECRET_KEY': os.environ.get('FLASK_SECRET_KEY') or os.urandom(32),
        'JWT_SECRET': os.environ.get('JWT_SECRET') or os.urandom(32),
        'CORS_ORIGINS': ['http://localhost:3000', 'http://127.0.0.1:3000'],
        'MAX_CONTENT_LENGTH': 16 * 1024 * 1024,  # 16MB max request size
        'JSON_SORT_KEYS': False
    }
    
    if config:
        default_config.update(config)
    
    app.config.update(default_config)
    
    # Enable CORS for cross-origin requests
    CORS(app, origins=app.config['CORS_ORIGINS'])
    
    # Initialize security components
    auth_config = {
        'jwt_secret': app.config['JWT_SECRET'],
        'max_failed_attempts': 5,
        'lockout_duration_minutes': 15,
        'token_expiry_hours': 24
    }
    
    app.auth_manager = SecureAuthManager(auth_config)
    app.encryption = DataEncryption()
    app.training_manager = RemoteTrainingManager(app.auth_manager, app.encryption)
    
    # Initialize labyrinth agents if available (optional legacy support)
    try:
        from labyrinth_adaptive import setup_labyrinth_agents
        app.labyrinth_agents = setup_labyrinth_agents()
        logger.info("Labyrinth agents initialized")
    except ImportError:
        logger.warning("Labyrinth agents not available")
        app.labyrinth_agents = []
    
    # Register API routes
    from api.routes import register_routes
    register_routes(app)
    
    logger.info("DuetMind Remote Training API initialized")
    return app

def main(argv=None):
    """Main entry point for the API server."""
    parser = argparse.ArgumentParser(description='DuetMind Remote Training API Server')
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--ssl-cert', help='SSL certificate file for HTTPS')
    parser.add_argument('--ssl-key', help='SSL private key file for HTTPS')
    
    args = parser.parse_args(argv)
    
    # Create application
    app = create_app()
    
    # Show startup information
    print("🧠 DuetMind Adaptive Remote Training API")
    print("=" * 50)
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Debug: {args.debug}")
    print(f"Security: {'HTTPS' if args.ssl_cert else 'HTTP'}")
    print("\n📚 Available Endpoints:")
    print("  Training:")
    print("    POST   /api/v1/training/submit      - Submit training job")
    print("    GET    /api/v1/training/status/<id> - Get job status")
    print("    GET    /api/v1/training/jobs        - List user jobs")
    print("    POST   /api/v1/training/cancel/<id> - Cancel job")
    print("    GET    /api/v1/training/model/<id>  - Download trained model")
    print("  System:")
    print("    GET    /api/v1/health               - Health check")
    print("    GET    /api/v1/training/examples    - Training examples")
    print("    GET    /api/v1/admin/training/status - System status (admin)")
    print("  Legacy:")
    print("    POST   /api/v1/labyrinth/reason     - Labyrinth reasoning")
    print("    GET    /api/v1/labyrinth/state      - Labyrinth state")
    print("\n🔐 Authentication:")
    
    # Show generated API keys
    admin_keys = [k for k, v in app.auth_manager.api_keys.items() if 'admin' in v['roles']]
    user_keys = [k for k, v in app.auth_manager.api_keys.items() if 'admin' not in v['roles']]
    
    if admin_keys:
        print(f"  Admin API Key: {admin_keys[0]}")
    if user_keys:
        print(f"  User API Key:  {user_keys[0]}")
    
    print("\n🔧 Usage Examples:")
    print("  # Health check")
    print(f"  curl http://{args.host}:{args.port}/api/v1/health")
    print("\n  # Submit training job")
    print(f"  curl -X POST http://{args.host}:{args.port}/api/v1/training/submit \\")
    print(f"    -H 'X-API-Key: <API_KEY>' \\")
    print(f"    -H 'Content-Type: application/json' \\")
    print(f"    -d '{{\"model_type\": \"alzheimer_classifier\", \"dataset_source\": \"synthetic_test_data\"}}'")
    print("=" * 50)
    
    # Configure SSL context if provided
    ssl_context = None
    if args.ssl_cert and args.ssl_key:
        ssl_context = (args.ssl_cert, args.ssl_key)
        logger.info("HTTPS enabled with SSL certificates")
    
    try:
        # Start the server
        app.run(
            host=args.host,
            port=args.port,
            debug=args.debug,
            ssl_context=ssl_context,
            threaded=True  # Enable threading for concurrent requests
        )
        return 0
    except KeyboardInterrupt:
        logger.info("Server shutdown requested by user")
        return 0
    except Exception as e:
        logger.error(f"Server startup failed: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())