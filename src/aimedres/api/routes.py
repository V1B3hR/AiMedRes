"""
API Routes Blueprint for DuetMind Adaptive

Modular route definitions for clean separation of concerns.
"""

from flask import Blueprint, request, jsonify
from ..security.auth import require_auth

# Create blueprint for API routes
api_bp = Blueprint('api', __name__, url_prefix='/api/v1')

@api_bp.route('/info', methods=['GET'])
def api_info():
    """API information endpoint"""
    return jsonify({
        "name": "DuetMind Adaptive API",
        "version": "1.0.0",
        "description": "Secure AI framework for adaptive neural networks and cognitive agents",
        "endpoints": [
            "/health - System health check",
            "/api/v1/predict - Neural network prediction",
            "/api/v1/agent/think - Agent cognitive processing", 
            "/api/v1/status - System status"
        ]
    })

# Additional route modules can be imported and registered here
# from .agent_routes import agent_bp
# from .training_routes import training_bp