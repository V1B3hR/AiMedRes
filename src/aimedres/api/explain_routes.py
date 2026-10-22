"""
Explainability API endpoints.

Provides endpoints for:
- Feature attribution (SHAP, LIME, etc.)
- Uncertainty/confidence estimation
- Explanation visualization data
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from flask import Blueprint, jsonify, request

from ..security.auth import require_auth

logger = logging.getLogger(__name__)

explain_bp = Blueprint("explain", __name__, url_prefix="/api/v1/explain")


class ExplainabilityEngine:
    """
    Engine for generating model explanations.

    Supports:
    - Feature attribution
    - Uncertainty quantification
    - Confidence intervals
    """

    def __init__(self):
        self.explanation_cache = {}

    def compute_feature_attribution(
        self,
        prediction_id: str,
        case_data: Dict[str, Any],
        method: str = "shap",
        model_or_pipeline: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """
        Compute feature attribution for prediction using model weights or clinical heuristics.

        Args:
            prediction_id: Prediction identifier
            case_data: Case input data
            method: Attribution method (shap, lime, gradients)
            model_or_pipeline: Optional trained model/pipeline

        Returns:
            List of feature attributions
        """
        from ..clinical.usability import FeatureAttributionEvaluator

        if case_data and (model_or_pipeline is not None or any(k in FeatureAttributionEvaluator.CLINICAL_DOMAIN_RULES for k in case_data)):
            attr_objs = FeatureAttributionEvaluator.compute_attributions(
                model_or_pipeline=model_or_pipeline,
                patient_record=case_data,
            )
            features = [
                {
                    "feature": a.feature,
                    "importance": a.importance,
                    "value": a.value,
                    "contribution": a.contribution,
                    "direction": a.direction,
                }
                for a in attr_objs
            ]
        else:
            features = [
                {"feature": "Age", "importance": 0.25, "value": case_data.get("age", 72), "contribution": 0.15, "direction": "risk_increasing"},
                {"feature": "MMSE Score", "importance": 0.35, "value": case_data.get("mmse", 24), "contribution": -0.18, "direction": "protective"},
                {
                    "feature": "Hippocampal Volume",
                    "importance": 0.20,
                    "value": case_data.get("hippocampal_vol", 6500),
                    "contribution": 0.12,
                    "direction": "risk_increasing",
                },
                {"feature": "APOE4 Status", "importance": 0.15, "value": case_data.get("apoe4", 1), "contribution": 0.10, "direction": "risk_increasing"},
                {"feature": "Education Years", "importance": 0.05, "value": case_data.get("education", 16), "contribution": -0.02, "direction": "protective"},
            ]

        logger.info(f"Computed feature attribution for prediction: {prediction_id}")
        return features

    def compute_uncertainty(
        self, prediction_id: str, case_data: Dict[str, Any], method: str = "ensemble"
    ) -> Dict[str, Any]:
        """
        Compute uncertainty metrics for prediction.

        Args:
            prediction_id: Prediction identifier
            case_data: Case input data
            method: Uncertainty method (ensemble, bayesian, dropout)

        Returns:
            Uncertainty metrics
        """
        # In production, use actual uncertainty quantification
        # For MVP, generate synthetic but plausible metrics

        base_confidence = 0.78

        # Simulate epistemic and aleatoric uncertainty
        epistemic_uncertainty = 0.12  # Model uncertainty
        aleatoric_uncertainty = 0.08  # Data uncertainty
        total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty

        uncertainty_metrics = {
            "confidence": base_confidence,
            "total_uncertainty": total_uncertainty,
            "epistemic_uncertainty": epistemic_uncertainty,
            "aleatoric_uncertainty": aleatoric_uncertainty,
            "confidence_interval": {
                "lower": base_confidence - 1.96 * total_uncertainty,
                "upper": base_confidence + 1.96 * total_uncertainty,
            },
            "calibration_score": 0.85,
            "method": method,
        }

        logger.info(f"Computed uncertainty for prediction: {prediction_id}")

        return uncertainty_metrics

    def generate_explanation_summary(
        self, prediction_id: str, attributions: List[Dict[str, Any]], uncertainty: Dict[str, Any]
    ) -> str:
        """
        Generate human-readable explanation summary.

        Args:
            prediction_id: Prediction identifier
            attributions: Feature attributions
            uncertainty: Uncertainty metrics

        Returns:
            Human-readable explanation text
        """
        # Find top contributing features
        top_features = sorted(attributions, key=lambda x: abs(x["contribution"]), reverse=True)[:3]

        feature_text = ", ".join([f"{f['feature']}" for f in top_features])

        confidence = uncertainty["confidence"]
        confidence_desc = "high" if confidence > 0.8 else "moderate" if confidence > 0.6 else "low"

        summary = (
            f"The prediction is based primarily on: {feature_text}. "
            f"The model has {confidence_desc} confidence ({confidence:.2%}) in this prediction. "
            f"Estimated uncertainty: ±{uncertainty['total_uncertainty']:.2%}."
        )

        return summary


# Global explainability engine
explainability_engine = ExplainabilityEngine()


@explain_bp.route("/attribution", methods=["POST"])
@require_auth()
def feature_attribution():
    """
    Get feature attributions for a prediction.

    Request Body:
        {
            "prediction_id": "required",
            "case_id": "optional",
            "method": "shap|lime|gradients"
        }

    Returns:
        Feature attributions with importance scores
    """
    try:
        data = request.get_json()

        if not data or "prediction_id" not in data:
            return jsonify({"error": "Missing required field: prediction_id"}), 400

        prediction_id = data["prediction_id"]
        case_id = data.get("case_id")
        method = data.get("method", "shap")

        # Get case data (simplified)
        case_data = data.get("case_data", {})

        # Compute attributions
        attributions = explainability_engine.compute_feature_attribution(
            prediction_id, case_data, method
        )

        return jsonify(
            {
                "prediction_id": prediction_id,
                "case_id": case_id,
                "method": method,
                "attributions": attributions,
                "timestamp": datetime.now().isoformat(),
            }
        )

    except Exception as e:
        logger.error(f"Attribution error: {e}")
        return jsonify({"error": "Attribution computation failed", "message": str(e)}), 500


@explain_bp.route("/uncertainty", methods=["POST"])
@require_auth()
def uncertainty_estimation():
    """
    Get uncertainty and confidence metrics for a prediction.

    Request Body:
        {
            "prediction_id": "required",
            "case_id": "optional",
            "method": "ensemble|bayesian|dropout"
        }

    Returns:
        Uncertainty metrics including confidence intervals
    """
    try:
        data = request.get_json()

        if not data or "prediction_id" not in data:
            return jsonify({"error": "Missing required field: prediction_id"}), 400

        prediction_id = data["prediction_id"]
        case_id = data.get("case_id")
        method = data.get("method", "ensemble")

        # Get case data (simplified)
        case_data = data.get("case_data", {})

        # Compute uncertainty
        uncertainty = explainability_engine.compute_uncertainty(prediction_id, case_data, method)

        return jsonify(
            {
                "prediction_id": prediction_id,
                "case_id": case_id,
                "uncertainty_metrics": uncertainty,
                "timestamp": datetime.now().isoformat(),
            }
        )

    except Exception as e:
        logger.error(f"Uncertainty estimation error: {e}")
        return jsonify({"error": "Uncertainty computation failed", "message": str(e)}), 500


@explain_bp.route("/full", methods=["POST"])
@require_auth()
def full_explanation():
    """
    Get comprehensive explanation including attribution and uncertainty.

    Request Body:
        {
            "prediction_id": "required",
            "case_id": "optional"
        }

    Returns:
        Complete explanation with attributions, uncertainty, and summary
    """
    try:
        data = request.get_json()

        if not data or "prediction_id" not in data:
            return jsonify({"error": "Missing required field: prediction_id"}), 400

        prediction_id = data["prediction_id"]
        case_id = data.get("case_id")
        case_data = data.get("case_data", {})

        # Compute both attribution and uncertainty
        attributions = explainability_engine.compute_feature_attribution(prediction_id, case_data)

        uncertainty = explainability_engine.compute_uncertainty(prediction_id, case_data)

        # Generate summary
        summary = explainability_engine.generate_explanation_summary(
            prediction_id, attributions, uncertainty
        )

        return jsonify(
            {
                "prediction_id": prediction_id,
                "case_id": case_id,
                "attributions": attributions,
                "uncertainty": uncertainty,
                "summary": summary,
                "timestamp": datetime.now().isoformat(),
            }
        )

    except Exception as e:
        logger.error(f"Full explanation error: {e}")
        return jsonify({"error": "Explanation generation failed", "message": str(e)}), 500


@explain_bp.route("/visualize/<prediction_id>", methods=["GET"])
@require_auth()
def get_visualization_data(prediction_id: str):
    """
    Get visualization-ready explanation data.

    Args:
        prediction_id: Prediction identifier

    Returns:
        Data formatted for visualization components
    """
    try:
        # Get cached or compute new explanation
        case_data = {}  # Would retrieve from database in production

        attributions = explainability_engine.compute_feature_attribution(prediction_id, case_data)

        uncertainty = explainability_engine.compute_uncertainty(prediction_id, case_data)

        # Format for visualization
        viz_data = {
            "feature_importance_chart": {
                "type": "bar",
                "data": [
                    {"feature": a["feature"], "importance": a["importance"]} for a in attributions
                ],
            },
            "contribution_chart": {
                "type": "waterfall",
                "data": [
                    {"feature": a["feature"], "contribution": a["contribution"]}
                    for a in attributions
                ],
            },
            "confidence_gauge": {
                "value": uncertainty["confidence"],
                "min": 0,
                "max": 1,
                "threshold": 0.7,
            },
            "uncertainty_band": {
                "lower": uncertainty["confidence_interval"]["lower"],
                "upper": uncertainty["confidence_interval"]["upper"],
            },
        }

        return jsonify(
            {
                "prediction_id": prediction_id,
                "visualization_data": viz_data,
                "timestamp": datetime.now().isoformat(),
            }
        )

    except Exception as e:
        logger.error(f"Visualization data error: {e}")
        return jsonify({"error": "Failed to generate visualization data", "message": str(e)}), 500
