"""
API Routes for Canary Deployment Management (P3-3)

Provides endpoints for:
- Canary deployment creation and management
- Shadow mode deployment
- Automated validation
- Rollback operations
- Deployment monitoring
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from flask import Blueprint, jsonify, request

logger = logging.getLogger("aimedres.api.canary_routes")

# Create blueprint
canary_bp = Blueprint("canary", __name__, url_prefix="/api/v1/canary")

# Import canary deployment components
try:
    import os
    import sys

    mlops_path = os.path.join(os.path.dirname(__file__), "../../../mlops")
    if mlops_path not in sys.path:
        sys.path.insert(0, mlops_path)

    from pipelines.canary_deployment import (
        CanaryDeploymentPipeline,
        DeploymentMode,
        DeploymentStatus,
    )

    CANARY_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Canary deployment module not available: {e}")
    CANARY_AVAILABLE = False

from ..security.auth import require_auth

# Initialize canary pipeline
canary_pipeline = None
if CANARY_AVAILABLE:
    try:
        canary_pipeline = CanaryDeploymentPipeline(
            validation_holdout_size=1000,
            shadow_duration_minutes=60,
            canary_traffic_increments=[5, 10, 25, 50, 100],
        )
    except Exception as e:
        logger.error(f"Failed to initialize canary pipeline: {e}")


# ==================== Deployment Management ====================


@canary_bp.route("/deployments", methods=["POST"])
@require_auth
def create_deployment():
    """
    Create a new canary deployment.

    Request body:
    {
        "model_id": "model-v2",
        "model_version": "2.1.0",
        "mode": "shadow",
        "description": "New improved model with better accuracy"
    }

    Returns:
        Deployment ID and initial status
    """
    try:
        if not CANARY_AVAILABLE or not canary_pipeline:
            return jsonify({"error": "Canary deployment not available"}), 503

        data = request.get_json()
        model_id = data.get("model_id")
        model_version = data.get("model_version")
        mode_str = data.get("mode", "shadow")
        description = data.get("description", "")

        if not model_id or not model_version:
            return jsonify({"error": "model_id and model_version are required"}), 400

        # Validate mode
        try:
            mode = DeploymentMode[mode_str.upper()]
        except KeyError:
            return jsonify({"error": f"Invalid mode: {mode_str}"}), 400

        # Create deployment
        deployment = canary_pipeline.create_deployment(
            model_id=model_id, model_version=model_version, mode=mode
        )

        return (
            jsonify(
                {
                    "success": True,
                    "data": {
                        "deployment_id": deployment.deployment_id,
                        "model_id": deployment.model_id,
                        "model_version": deployment.model_version,
                        "mode": deployment.mode.value,
                        "status": deployment.status.value,
                        "traffic_percentage": deployment.traffic_percentage,
                        "started_at": deployment.started_at.isoformat(),
                    },
                }
            ),
            201,
        )

    except Exception as e:
        logger.error(f"Error creating deployment: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@canary_bp.route("/deployments", methods=["GET"])
@require_auth
def list_deployments():
    """
    List all deployments.

    Query params:
        status: Filter by status (optional)
        mode: Filter by mode (optional)

    Returns:
        List of deployments
    """
    try:
        if not CANARY_AVAILABLE or not canary_pipeline:
            return jsonify({"error": "Canary deployment not available"}), 503

        status_filter = request.args.get("status")
        mode_filter = request.args.get("mode")

        deployments = canary_pipeline.list_deployments()

        # Apply filters
        if status_filter:
            deployments = [d for d in deployments if d.status.value == status_filter.lower()]

        if mode_filter:
            deployments = [d for d in deployments if d.mode.value == mode_filter.lower()]

        return (
            jsonify(
                {
                    "success": True,
                    "data": {
                        "deployments": [
                            {
                                "deployment_id": d.deployment_id,
                                "model_id": d.model_id,
                                "model_version": d.model_version,
                                "mode": d.mode.value,
                                "status": d.status.value,
                                "traffic_percentage": d.traffic_percentage,
                                "started_at": d.started_at.isoformat(),
                                "completed_at": (
                                    d.completed_at.isoformat() if d.completed_at else None
                                ),
                            }
                            for d in deployments
                        ],
                        "total_count": len(deployments),
                    },
                }
            ),
            200,
        )

    except Exception as e:
        logger.error(f"Error listing deployments: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@canary_bp.route("/deployments/<deployment_id>", methods=["GET"])
@require_auth
def get_deployment(deployment_id: str):
    """
    Get deployment details.

    Returns:
        Detailed deployment information including validation tests
    """
    try:
        if not CANARY_AVAILABLE or not canary_pipeline:
            return jsonify({"error": "Canary deployment not available"}), 503

        deployment = canary_pipeline.get_deployment(deployment_id)

        if not deployment:
            return jsonify({"error": "Deployment not found"}), 404

        return (
            jsonify(
                {
                    "success": True,
                    "data": {
                        "deployment_id": deployment.deployment_id,
                        "model_id": deployment.model_id,
                        "model_version": deployment.model_version,
                        "mode": deployment.mode.value,
                        "status": deployment.status.value,
                        "traffic_percentage": deployment.traffic_percentage,
                        "started_at": deployment.started_at.isoformat(),
                        "completed_at": (
                            deployment.completed_at.isoformat() if deployment.completed_at else None
                        ),
                        "validation_tests": [
                            {
                                "test_id": t.test_id,
                                "test_name": t.test_name,
                                "test_type": t.test_type,
                                "result": t.result.value,
                                "score": t.score,
                                "threshold": t.threshold,
                                "passed": t.passed,
                                "executed_at": t.executed_at.isoformat() if t.executed_at else None,
                            }
                            for t in deployment.validation_tests
                        ],
                        "performance_metrics": deployment.performance_metrics,
                        "rollback_triggered": deployment.rollback_triggered,
                        "rollback_reason": deployment.rollback_reason,
                    },
                }
            ),
            200,
        )

    except Exception as e:
        logger.error(f"Error getting deployment: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@canary_bp.route("/deployments/<deployment_id>/promote", methods=["POST"])
@require_auth
def promote_deployment(deployment_id: str):
    """
    Promote canary deployment to next stage.

    Request body:
    {
        "target_traffic": 25  // Optional, for canary mode
    }

    Returns:
        Updated deployment status
    """
    try:
        if not CANARY_AVAILABLE or not canary_pipeline:
            return jsonify({"error": "Canary deployment not available"}), 503

        data = request.get_json() or {}
        target_traffic = data.get("target_traffic")

        deployment = canary_pipeline.promote_deployment(
            deployment_id=deployment_id, target_traffic=target_traffic
        )

        if not deployment:
            return jsonify({"error": "Deployment not found"}), 404

        return (
            jsonify(
                {
                    "success": True,
                    "data": {
                        "deployment_id": deployment.deployment_id,
                        "status": deployment.status.value,
                        "traffic_percentage": deployment.traffic_percentage,
                        "message": "Deployment promoted successfully",
                    },
                }
            ),
            200,
        )

    except Exception as e:
        logger.error(f"Error promoting deployment: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@canary_bp.route("/deployments/<deployment_id>/rollback", methods=["POST"])
@require_auth
def rollback_deployment(deployment_id: str):
    """
    Rollback a deployment.

    Request body:
    {
        "reason": "Performance degradation detected"
    }

    Returns:
        Rollback confirmation
    """
    try:
        if not CANARY_AVAILABLE or not canary_pipeline:
            return jsonify({"error": "Canary deployment not available"}), 503

        data = request.get_json() or {}
        reason = data.get("reason", "Manual rollback")

        deployment = canary_pipeline.rollback_deployment(deployment_id=deployment_id, reason=reason)

        if not deployment:
            return jsonify({"error": "Deployment not found"}), 404

        return (
            jsonify(
                {
                    "success": True,
                    "data": {
                        "deployment_id": deployment.deployment_id,
                        "status": deployment.status.value,
                        "rollback_triggered": deployment.rollback_triggered,
                        "rollback_reason": deployment.rollback_reason,
                        "message": "Deployment rolled back successfully",
                    },
                }
            ),
            200,
        )

    except Exception as e:
        logger.error(f"Error rolling back deployment: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


# ==================== Validation ====================


@canary_bp.route("/deployments/<deployment_id>/validate", methods=["POST"])
@require_auth
def run_validation(deployment_id: str):
    """
    Run validation tests on deployment.

    Request body:
    {
        "test_types": ["accuracy", "fairness", "performance"]
    }

    Returns:
        Validation results
    """
    try:
        if not CANARY_AVAILABLE or not canary_pipeline:
            return jsonify({"error": "Canary deployment not available"}), 503

        data = request.get_json() or {}
        test_types = data.get("test_types", ["accuracy", "fairness", "performance"])

        results = canary_pipeline.run_validation_tests(
            deployment_id=deployment_id, test_types=test_types
        )

        return (
            jsonify(
                {
                    "success": True,
                    "data": {
                        "deployment_id": deployment_id,
                        "validation_results": [
                            {
                                "test_id": r.test_id,
                                "test_name": r.test_name,
                                "test_type": r.test_type,
                                "result": r.result.value,
                                "score": r.score,
                                "threshold": r.threshold,
                                "passed": r.passed,
                                "details": r.details,
                            }
                            for r in results
                        ],
                        "all_passed": all(r.passed for r in results),
                    },
                }
            ),
            200,
        )

    except Exception as e:
        logger.error(f"Error running validation: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


# ==================== Monitoring ====================


@canary_bp.route("/deployments/<deployment_id>/metrics", methods=["GET"])
@require_auth
def get_deployment_metrics(deployment_id: str):
    """
    Get real-time metrics for deployment.

    Returns:
        Performance metrics and health indicators
    """
    try:
        if not CANARY_AVAILABLE or not canary_pipeline:
            return jsonify({"error": "Canary deployment not available"}), 503

        metrics = canary_pipeline.get_deployment_metrics(deployment_id)

        if not metrics:
            return jsonify({"error": "Deployment not found"}), 404

        return (
            jsonify(
                {
                    "success": True,
                    "data": {
                        "deployment_id": deployment_id,
                        "metrics": metrics,
                        "timestamp": datetime.now().isoformat(),
                    },
                }
            ),
            200,
        )

    except Exception as e:
        logger.error(f"Error getting deployment metrics: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@canary_bp.route("/deployments/<deployment_id>/comparison", methods=["GET"])
@require_auth
def compare_with_baseline(deployment_id: str):
    """
    Compare deployment with baseline model.

    Returns:
        Comparison metrics between new and baseline models
    """
    try:
        if not CANARY_AVAILABLE or not canary_pipeline:
            return jsonify({"error": "Canary deployment not available"}), 503

        comparison = canary_pipeline.compare_with_baseline(deployment_id)

        if not comparison:
            return jsonify({"error": "Deployment not found"}), 404

        return jsonify({"success": True, "data": comparison}), 200

    except Exception as e:
        logger.error(f"Error comparing with baseline: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


# ==================== Health Check ====================


@canary_bp.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint for canary deployment service."""
    return (
        jsonify(
            {
                "status": "healthy" if CANARY_AVAILABLE and canary_pipeline else "degraded",
                "service": "canary_deployment",
                "timestamp": datetime.now().isoformat(),
                "canary_pipeline_available": CANARY_AVAILABLE,
            }
        ),
        200,
    )
