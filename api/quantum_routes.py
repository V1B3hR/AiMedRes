#!/usr/bin/env python3
"""
API Routes for Quantum Key Management

Provides endpoints for:
- Key listing and details
- Key rotation
- Rotation policy management
- Key management statistics
- Rotation history
"""

from flask import request, jsonify, current_app, Blueprint
import logging
from typing import Dict, Any
from datetime import datetime

from security.auth import require_auth, require_admin

logger = logging.getLogger('aimedres.api.quantum')

# Create Blueprint for Quantum Key Management API routes
quantum_bp = Blueprint('quantum', __name__, url_prefix='/api/v1/quantum')

@quantum_bp.route('/keys', methods=['GET'])
@require_admin
def list_keys():
    """
    List cryptographic keys.
    
    Required: Admin API key
    Query params: key_type, status (optional filters)
    """
    try:
        key_type = request.args.get('key_type')
        status = request.args.get('status')
        
        if not hasattr(current_app, 'quantum_key_manager'):
            return jsonify({
                'keys': [],
                'message': 'Quantum key manager not initialized'
            }), 200
        
        key_manager = current_app.quantum_key_manager
        keys = key_manager.list_keys(key_type=key_type, status=status)
        
        return jsonify({
            'success': True,
            'keys': [
                {
                    'key_id': k.key_id,
                    'key_type': k.key_type.value,
                    'status': k.status.value,
                    'created_at': k.created_at.isoformat(),
                    'last_rotated': k.last_rotated.isoformat() if k.last_rotated else None,
                    'expires_at': k.expires_at.isoformat() if k.expires_at else None,
                    'rotation_count': k.rotation_count,
                    'usage_count': k.usage_count,
                    'metadata': k.metadata
                }
                for k in keys
            ]
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to list keys: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@quantum_bp.route('/keys/<key_id>', methods=['GET'])
@require_admin
def get_key(key_id: str):
    """
    Get key details.
    
    Required: Admin API key
    """
    try:
        if not hasattr(current_app, 'quantum_key_manager'):
            return jsonify({'error': 'Quantum key manager not initialized'}), 503
        
        key_manager = current_app.quantum_key_manager
        key = key_manager.get_key(key_id)
        
        if not key:
            return jsonify({'error': 'Key not found'}), 404
        
        return jsonify({
            'key_id': key.key_id,
            'key_type': key.key_type.value,
            'status': key.status.value,
            'created_at': key.created_at.isoformat(),
            'last_rotated': key.last_rotated.isoformat() if key.last_rotated else None,
            'expires_at': key.expires_at.isoformat() if key.expires_at else None,
            'rotation_count': key.rotation_count,
            'usage_count': key.usage_count,
            'metadata': key.metadata
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to get key: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@quantum_bp.route('/stats', methods=['GET'])
@require_admin
def get_stats():
    """
    Get key manager statistics.
    
    Required: Admin API key
    """
    try:
        if not hasattr(current_app, 'quantum_key_manager'):
            return jsonify({'error': 'Quantum key manager not initialized'}), 503
        
        key_manager = current_app.quantum_key_manager
        stats = key_manager.get_stats()
        
        return jsonify(stats), 200
        
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@quantum_bp.route('/policy', methods=['GET'])
@require_admin
def get_rotation_policy():
    """
    Get key rotation policy.
    
    Required: Admin API key
    """
    try:
        if not hasattr(current_app, 'quantum_key_manager'):
            return jsonify({'error': 'Quantum key manager not initialized'}), 503
        
        key_manager = current_app.quantum_key_manager
        policy = key_manager.get_rotation_policy()
        
        return jsonify({
            'enabled': policy.enabled,
            'rotation_interval_days': policy.rotation_interval_days,
            'max_key_age_days': policy.max_key_age_days,
            'grace_period_days': policy.grace_period_days,
            'automatic_rotation': policy.automatic_rotation,
            'notify_before_rotation_days': policy.notify_before_rotation_days,
            'require_manual_approval': policy.require_manual_approval
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to get policy: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@quantum_bp.route('/policy', methods=['PUT'])
@require_admin
def update_rotation_policy():
    """
    Update key rotation policy.
    
    Required: Admin API key
    Body: Partial KeyRotationPolicy object
    """
    try:
        data = request.get_json()
        
        if not hasattr(current_app, 'quantum_key_manager'):
            return jsonify({'error': 'Quantum key manager not initialized'}), 503
        
        key_manager = current_app.quantum_key_manager
        updated_policy = key_manager.update_rotation_policy(data)
        
        return jsonify({
            'success': True,
            'policy': {
                'enabled': updated_policy.enabled,
                'rotation_interval_days': updated_policy.rotation_interval_days,
                'max_key_age_days': updated_policy.max_key_age_days,
                'grace_period_days': updated_policy.grace_period_days,
                'automatic_rotation': updated_policy.automatic_rotation,
                'notify_before_rotation_days': updated_policy.notify_before_rotation_days,
                'require_manual_approval': updated_policy.require_manual_approval
            }
        }), 200
        
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Failed to update policy: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@quantum_bp.route('/keys/<key_id>/rotate', methods=['POST'])
@require_admin
def rotate_key(key_id: str):
    """
    Manually rotate a key.
    
    Required: Admin API key
    """
    try:
        if not hasattr(current_app, 'quantum_key_manager'):
            return jsonify({'error': 'Quantum key manager not initialized'}), 503
        
        key_manager = current_app.quantum_key_manager
        success = key_manager.rotate_key(key_id)
        
        if success:
            return jsonify({
                'success': True,
                'message': 'Key rotated successfully',
                'key_id': key_id
            }), 200
        else:
            return jsonify({'error': 'Failed to rotate key'}), 400
            
    except ValueError as e:
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        logger.error(f"Failed to rotate key: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@quantum_bp.route('/history', methods=['GET'])
@require_admin
def get_rotation_history():
    """
    Get key rotation history.
    
    Required: Admin API key
    Query params: limit (optional, default 50)
    """
    try:
        limit = int(request.args.get('limit', 50))
        
        if not hasattr(current_app, 'quantum_key_manager'):
            return jsonify({
                'events': [],
                'message': 'Quantum key manager not initialized'
            }), 200
        
        key_manager = current_app.quantum_key_manager
        events = key_manager.get_rotation_history(limit=limit)
        
        return jsonify({
            'success': True,
            'events': [
                {
                    'event_id': e.event_id,
                    'key_id': e.key_id,
                    'event_type': e.event_type,
                    'timestamp': e.timestamp.isoformat(),
                    'details': e.details
                }
                for e in events
            ]
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to get history: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@quantum_bp.route('/health', methods=['GET'])
def quantum_health():
    """
    Quantum key management health check.
    """
    try:
        # Check if quantum crypto module is available
        try:
            from security.quantum_crypto import QuantumSafeCryptography
            quantum_available = True
        except ImportError:
            quantum_available = False
        
        health = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'manager_initialized': hasattr(current_app, 'quantum_key_manager'),
            'quantum_crypto_available': quantum_available
        }
        
        if hasattr(current_app, 'quantum_key_manager'):
            key_manager = current_app.quantum_key_manager
            stats = key_manager.get_stats()
            health['total_keys'] = stats.get('total_keys', 0)
            health['active_keys'] = stats.get('active_keys', 0)
        
        return jsonify(health), 200
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 500
