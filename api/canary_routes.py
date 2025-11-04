#!/usr/bin/env python3
"""
API Routes for Canary Deployment Monitoring

Provides endpoints for:
- Canary deployment monitoring
- Validation test results
- Real-time metrics
- Rollback and promotion actions
"""

from flask import request, jsonify, current_app, Blueprint
import logging
from typing import Dict, Any
from datetime import datetime

from security.auth import require_auth, require_admin

logger = logging.getLogger('aimedres.api.canary')

# Create Blueprint for Canary API routes
canary_bp = Blueprint('canary', __name__, url_prefix='/api/v1/canary')

@canary_bp.route('/deployments', methods=['GET'])
@require_auth('user')
def list_deployments():
    """
    List canary deployments.
    
    Required: API key authentication
    Query params: limit (optional, default 20)
    """
    try:
        limit = int(request.args.get('limit', 20))
        
        # Get canary pipeline from app context
        if not hasattr(current_app, 'canary_pipeline'):
            return jsonify({
                'deployments': [],
                'message': 'Canary pipeline not initialized'
            }), 200
        
        canary_pipeline = current_app.canary_pipeline
        deployments = canary_pipeline.list_deployments(limit=limit)
        
        return jsonify({
            'success': True,
            'deployments': [
                {
                    'deployment_id': d.deployment_id,
                    'model_id': d.model_id,
                    'model_version': d.model_version,
                    'mode': d.mode.value,
                    'status': d.status.value,
                    'traffic_percentage': d.traffic_percentage,
                    'started_at': d.started_at.isoformat(),
                    'completed_at': d.completed_at.isoformat() if d.completed_at else None,
                    'validation_tests': [
                        {
                            'test_id': t.test_id,
                            'test_name': t.test_name,
                            'test_type': t.test_type,
                            'result': t.result.value,
                            'score': t.score,
                            'threshold': t.threshold,
                            'passed': t.passed,
                            'details': t.details,
                            'executed_at': t.executed_at.isoformat() if t.executed_at else None
                        }
                        for t in d.validation_tests
                    ],
                    'performance_metrics': d.performance_metrics,
                    'rollback_triggered': d.rollback_triggered,
                    'rollback_reason': d.rollback_reason
                }
                for d in deployments
            ]
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to list deployments: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@canary_bp.route('/deployments/<deployment_id>', methods=['GET'])
@require_auth('user')
def get_deployment(deployment_id: str):
    """
    Get deployment details.
    
    Required: API key authentication
    """
    try:
        if not hasattr(current_app, 'canary_pipeline'):
            return jsonify({'error': 'Canary pipeline not initialized'}), 503
        
        canary_pipeline = current_app.canary_pipeline
        deployment = canary_pipeline.get_deployment(deployment_id)
        
        if not deployment:
            return jsonify({'error': 'Deployment not found'}), 404
        
        return jsonify({
            'deployment_id': deployment.deployment_id,
            'model_id': deployment.model_id,
            'model_version': deployment.model_version,
            'mode': deployment.mode.value,
            'status': deployment.status.value,
            'traffic_percentage': deployment.traffic_percentage,
            'started_at': deployment.started_at.isoformat(),
            'completed_at': deployment.completed_at.isoformat() if deployment.completed_at else None,
            'validation_tests': [
                {
                    'test_id': t.test_id,
                    'test_name': t.test_name,
                    'test_type': t.test_type,
                    'result': t.result.value,
                    'score': t.score,
                    'threshold': t.threshold,
                    'passed': t.passed,
                    'details': t.details,
                    'executed_at': t.executed_at.isoformat() if t.executed_at else None
                }
                for t in deployment.validation_tests
            ],
            'performance_metrics': deployment.performance_metrics,
            'rollback_triggered': deployment.rollback_triggered,
            'rollback_reason': deployment.rollback_reason
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to get deployment: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@canary_bp.route('/deployments/<deployment_id>/metrics', methods=['GET'])
@require_auth('user')
def get_deployment_metrics(deployment_id: str):
    """
    Get real-time deployment metrics.
    
    Required: API key authentication
    """
    try:
        if not hasattr(current_app, 'canary_pipeline'):
            return jsonify({'error': 'Canary pipeline not initialized'}), 503
        
        canary_pipeline = current_app.canary_pipeline
        metrics = canary_pipeline.get_deployment_metrics(deployment_id)
        
        if not metrics:
            return jsonify({'error': 'Metrics not found'}), 404
        
        return jsonify(metrics), 200
        
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@canary_bp.route('/deployments/<deployment_id>/rollback', methods=['POST'])
@require_admin
def trigger_rollback(deployment_id: str):
    """
    Trigger deployment rollback.
    
    Required: Admin API key
    Body: { "reason": "rollback reason" }
    """
    try:
        data = request.get_json()
        reason = data.get('reason', 'Manual rollback')
        
        if not hasattr(current_app, 'canary_pipeline'):
            return jsonify({'error': 'Canary pipeline not initialized'}), 503
        
        canary_pipeline = current_app.canary_pipeline
        success = canary_pipeline.trigger_rollback(deployment_id, reason)
        
        if success:
            return jsonify({
                'success': True,
                'message': 'Rollback triggered successfully',
                'deployment_id': deployment_id,
                'reason': reason
            }), 200
        else:
            return jsonify({'error': 'Failed to trigger rollback'}), 400
            
    except Exception as e:
        logger.error(f"Failed to trigger rollback: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@canary_bp.route('/deployments/<deployment_id>/promote', methods=['POST'])
@require_admin
def promote_deployment(deployment_id: str):
    """
    Promote canary deployment to stable.
    
    Required: Admin API key
    """
    try:
        if not hasattr(current_app, 'canary_pipeline'):
            return jsonify({'error': 'Canary pipeline not initialized'}), 503
        
        canary_pipeline = current_app.canary_pipeline
        success = canary_pipeline.promote_to_stable(deployment_id)
        
        if success:
            return jsonify({
                'success': True,
                'message': 'Deployment promoted to stable',
                'deployment_id': deployment_id
            }), 200
        else:
            return jsonify({'error': 'Failed to promote deployment'}), 400
            
    except Exception as e:
        logger.error(f"Failed to promote deployment: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@canary_bp.route('/health', methods=['GET'])
def canary_health():
    """
    Canary deployment health check.
    """
    try:
        health = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'pipeline_initialized': hasattr(current_app, 'canary_pipeline')
        }
        
        if hasattr(current_app, 'canary_pipeline'):
            canary_pipeline = current_app.canary_pipeline
            health['active_deployments'] = len(canary_pipeline.list_deployments(limit=100))
        
        return jsonify(health), 200
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 500
