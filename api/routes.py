#!/usr/bin/env python3
"""
Secure API routes for DuetMind Adaptive System

Provides comprehensive API endpoints for:
- Remote secure model training
- Training job management and monitoring  
- Secure model parameter exchange
- System status and health monitoring
- Legacy labyrinth agent functionality
"""

from flask import request, jsonify, current_app, Blueprint
import logging
from typing import Dict, Any
import traceback
from datetime import datetime

# Import security and training components
from security.auth import require_auth, require_admin
from remote_training_manager import RemoteTrainingManager

logger = logging.getLogger('duetmind.api')

# Create Blueprint for API routes
api_bp = Blueprint('api', __name__, url_prefix='/api/v1')

# ============================================================================
# REMOTE TRAINING API ENDPOINTS
# ============================================================================

@api_bp.route('/training/submit', methods=['POST'])
@require_auth('user')
def submit_training_job():
    """
    Submit a new remote training job.
    
    Required: API key authentication
    Body: Training configuration JSON
    
    Returns: Job ID for tracking
    """
    try:
        training_config = request.get_json()
        if not training_config:
            return jsonify({'error': 'Training configuration required'}), 400
        
        user_id = request.user_info['user_id']
        training_manager = current_app.training_manager
        
        job_id = training_manager.submit_training_job(user_id, training_config)
        
        return jsonify({
            'success': True,
            'job_id': job_id,
            'message': 'Training job submitted successfully'
        }), 201
        
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Training submission failed: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@api_bp.route('/training/status/<job_id>', methods=['GET'])
@require_auth('user')
def get_training_status(job_id: str):
    """
    Get training job status and progress.
    
    Required: API key authentication
    Returns: Job status, progress, and results
    """
    try:
        user_id = request.user_info['user_id']
        training_manager = current_app.training_manager
        
        status = training_manager.get_job_status(job_id, user_id)
        return jsonify(status), 200
        
    except ValueError as e:
        return jsonify({'error': str(e)}), 404
    except PermissionError as e:
        return jsonify({'error': str(e)}), 403
    except Exception as e:
        logger.error(f"Status retrieval failed: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@api_bp.route('/training/jobs', methods=['GET'])
@require_auth('user')
def list_training_jobs():
    """
    List user's training jobs.
    
    Required: API key authentication
    Query params: limit (optional, default 10)
    """
    try:
        user_id = request.user_info['user_id']
        limit = int(request.args.get('limit', 10))
        training_manager = current_app.training_manager
        
        jobs = training_manager.list_user_jobs(user_id, limit)
        return jsonify({'jobs': jobs}), 200
        
    except Exception as e:
        logger.error(f"Job listing failed: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@api_bp.route('/training/cancel/<job_id>', methods=['POST'])
@require_auth('user')
def cancel_training_job(job_id: str):
    """
    Cancel a training job.
    
    Required: API key authentication
    """
    try:
        user_id = request.user_info['user_id']
        training_manager = current_app.training_manager
        
        success = training_manager.cancel_job(job_id, user_id)
        
        if success:
            return jsonify({
                'success': True,
                'message': 'Job cancelled successfully'
            }), 200
        else:
            return jsonify({'error': 'Failed to cancel job'}), 400
            
    except ValueError as e:
        return jsonify({'error': str(e)}), 404
    except PermissionError as e:
        return jsonify({'error': str(e)}), 403
    except Exception as e:
        logger.error(f"Job cancellation failed: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@api_bp.route('/training/model/<job_id>', methods=['GET'])
@require_auth('user')
def download_trained_model(job_id: str):
    """
    Download encrypted trained model.
    
    Required: API key authentication
    Returns: Encrypted model data
    """
    try:
        user_id = request.user_info['user_id']
        training_manager = current_app.training_manager
        
        encrypted_model = training_manager.get_encrypted_model(job_id, user_id)
        
        return jsonify({
            'success': True,
            'encrypted_model': encrypted_model,
            'job_id': job_id
        }), 200
        
    except ValueError as e:
        return jsonify({'error': str(e)}), 404
    except PermissionError as e:
        return jsonify({'error': str(e)}), 403
    except Exception as e:
        logger.error(f"Model download failed: {e}")
        return jsonify({'error': 'Internal server error'}), 500

# ============================================================================
# SYSTEM MONITORING AND ADMIN ENDPOINTS  
# ============================================================================

@api_bp.route('/admin/training/status', methods=['GET'])
@require_admin
def get_system_training_status():
    """
    Get system-wide training status (admin only).
    
    Required: Admin API key
    Returns: System capacity and job statistics
    """
    try:
        training_manager = current_app.training_manager
        status = training_manager.get_system_status()
        return jsonify(status), 200
        
    except Exception as e:
        logger.error(f"System status failed: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@api_bp.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint (no authentication required).
    
    Returns: System health status
    """
    try:
        # Basic health checks
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'version': '1.0.0',
            'components': {
                'api': 'operational',
                'training_manager': 'operational' if hasattr(current_app, 'training_manager') else 'unavailable',
                'security': 'operational' if hasattr(current_app, 'auth_manager') else 'unavailable'
            }
        }
        
        return jsonify(health_status), 200
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

@api_bp.route('/training/examples', methods=['GET'])
def get_training_examples():
    """
    Get example training configurations (no authentication required).
    
    Returns: Sample training configurations for different use cases
    """
    examples = {
        'alzheimer_classifier_synthetic': {
            'model_type': 'alzheimer_classifier',
            'dataset_source': 'synthetic_test_data',
            'description': 'Train Alzheimer\'s disease classifier on synthetic test data'
        },
        'alzheimer_classifier_kaggle': {
            'model_type': 'alzheimer_classifier', 
            'dataset_source': 'kaggle_alzheimer',
            'description': 'Train Alzheimer\'s disease classifier on Kaggle dataset'
        },
        'adaptive_neural_net': {
            'model_type': 'adaptive_neural_net',
            'dataset_source': 'synthetic_test_data',
            'description': 'Train adaptive neural network with biological cycles'
        }
    }
    
    return jsonify({
        'examples': examples,
        'usage': 'Submit one of these configurations to /api/v1/training/submit'
    }), 200

# ============================================================================
# LEGACY LABYRINTH AGENT ENDPOINTS
# ============================================================================

@api_bp.route('/labyrinth/reason', methods=['POST'])
@require_auth('user')
def labyrinth_reason():
    """
    Legacy labyrinth agent reasoning endpoint.
    
    Required: API key authentication
    """
    try:
        data = request.get_json()
        agent_idx = int(data.get("agent_idx", 0))
        task = data.get("task")
        
        if not hasattr(current_app, 'labyrinth_agents') or not current_app.labyrinth_agents:
            return jsonify({'error': 'Labyrinth agents not initialized'}), 503
        
        if agent_idx >= len(current_app.labyrinth_agents):
            return jsonify({'error': 'Invalid agent index'}), 400
        
        agent = current_app.labyrinth_agents[agent_idx]
        result = agent.reason(task)
        return jsonify(result), 200
        
    except (IndexError, ValueError):
        return jsonify({'error': 'Invalid agent index'}), 400
    except Exception as e:
        logger.error(f"Labyrinth reasoning failed: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@api_bp.route('/labyrinth/state', methods=['GET'])
@require_auth('user')
def labyrinth_state():
    """
    Get labyrinth agents state.
    
    Required: API key authentication
    """
    try:
        if not hasattr(current_app, 'labyrinth_agents') or not current_app.labyrinth_agents:
            return jsonify({'error': 'Labyrinth agents not initialized'}), 503
        
        states = [agent.get_state() for agent in current_app.labyrinth_agents]
        return jsonify({"labyrinth_agents": states}), 200
        
    except Exception as e:
        logger.error(f"Labyrinth state failed: {e}")
        return jsonify({'error': 'Internal server error'}), 500

# ============================================================================
# REGISTER BLUEPRINT AND ERROR HANDLERS
# ============================================================================

def register_routes(app):
    """Register API routes and error handlers with Flask app."""
    app.register_blueprint(api_bp)
    
    # Error handlers
    @app.errorhandler(401)
    def unauthorized(error):
        """Handle authentication errors."""
        return jsonify({
            'error': 'Authentication required',
            'message': 'Valid API key required in X-API-Key header or api_key parameter'
        }), 401

    @app.errorhandler(403)
    def forbidden(error):
        """Handle authorization errors."""
        return jsonify({
            'error': 'Insufficient permissions',
            'message': 'This endpoint requires higher privileges'
        }), 403

    @app.errorhandler(404)
    def not_found(error):
        """Handle not found errors."""
        return jsonify({
            'error': 'Endpoint not found',
            'message': 'The requested API endpoint does not exist'
        }), 404

    @app.errorhandler(500)
    def internal_error(error):
        """Handle internal server errors."""
        logger.error(f"Internal server error: {error}")
        return jsonify({
            'error': 'Internal server error',
            'message': 'An unexpected error occurred'
        }), 500
