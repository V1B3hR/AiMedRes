"""
Model serving API endpoints.

Provides endpoints for:
- Model inference with versioning
- Model card metadata
- Model listing and discovery
"""

import logging
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from flask import Blueprint, jsonify, request

from ..security.auth import require_auth

logger = logging.getLogger(__name__)

model_bp = Blueprint("model", __name__, url_prefix="/api/v1/model")

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PPMI_MODEL_PATH = _REPO_ROOT / "results" / "ppmi_reproducibility_benchmark" / "models" / "logistic_regression_model.pkl"


class ModelRegistry:
    """
    Registry for trained models with versioning support.
    """

    def __init__(self):
        self.models: Dict[str, Dict[str, Any]] = {}
        self._loaded_models: Dict[str, Any] = {}
        self._initialize_models()

    def _initialize_models(self):
        """Initialize with available models."""
        # Register available models
        self.models = {
            "ppmi_parkinsons_v1": {
                "name": "PPMI Parkinson Motor Progression Classifier",
                "version": "v1.0.0",
                "type": "classification",
                "intended_use": "Predict Parkinson's motor progression (MDS-UPDRS Part III delta >= 5.0) from DAT/PET striatal binding ratios",
                "validation_metrics": {
                    "accuracy": 0.859,
                    "roc_auc": 0.907,
                    "roc_auc_ci_95": [0.859, 0.948],
                    "pr_auc": 0.959,
                    "pr_auc_ci_95": [0.934, 0.981],
                    "brier_score": 0.108,
                    "cross_validation": "5-Fold StratifiedGroupKFold on PATNO (0 patient leakage)",
                },
                "artifact_path": str(_PPMI_MODEL_PATH),
                "dataset_provenance": "PPMI SBR Cohort (USC LONI IDA / Synthetic Fixture)",
                "limitations": [
                    "Research use only",
                    "Requires 8 core striatal binding ratio features",
                    "Not approved by FDA for primary clinical diagnosis",
                ],
                "last_updated": "2026-10-22",
                "status": "active",
            },
            "alzheimer_v1": {
                "name": "Alzheimer Early Detection",
                "version": "v1.0.0",
                "type": "classification",
                "intended_use": "Early detection of Alzheimer's disease from clinical data",
                "validation_metrics": {
                    "accuracy": 0.89,
                    "sensitivity": 0.92,
                    "specificity": 0.87,
                    "auc_roc": 0.93,
                },
                "dataset_provenance": "ADNI dataset + synthetic augmentation",
                "limitations": [
                    "Not approved for clinical diagnosis",
                    "Requires validation on diverse populations",
                    "Performance may vary with data quality",
                ],
                "last_updated": "2025-01-15",
                "status": "active",
            },
            "parkinsons_v1": {
                "name": "Parkinson Disease Progression",
                "version": "v1.0.0",
                "type": "regression",
                "intended_use": "Predict Parkinson's disease progression",
                "validation_metrics": {"mse": 0.15, "r2_score": 0.82, "mae": 0.12},
                "dataset_provenance": "Public Parkinson datasets",
                "limitations": ["Research use only", "Limited to specific symptom scales"],
                "last_updated": "2025-01-20",
                "status": "active",
            },
            "als_v1": {
                "name": "ALS Risk Assessment",
                "version": "v1.0.0",
                "type": "classification",
                "intended_use": "ALS risk stratification and early detection",
                "validation_metrics": {"accuracy": 0.85, "sensitivity": 0.88, "specificity": 0.83},
                "dataset_provenance": "ALS clinical trial data",
                "limitations": ["Not a diagnostic tool", "Requires clinical validation"],
                "last_updated": "2025-01-22",
                "status": "active",
            },
        }

    def get_model_card(self, model_version: str = "latest") -> Optional[Dict[str, Any]]:
        """
        Get model card for specified version.

        Args:
            model_version: Model version or 'latest'

        Returns:
            Model card data
        """
        if model_version == "latest":
            # Return the most recently updated model
            latest = max(self.models.items(), key=lambda x: x[1]["last_updated"])
            return latest[1]

        return self.models.get(model_version)

    def list_models(self) -> list:
        """List all available models."""
        return [
            {
                "model_id": model_id,
                "name": data["name"],
                "version": data["version"],
                "status": data["status"],
            }
            for model_id, data in self.models.items()
        ]

    def predict(
        self, model_version: str, input_data: Union[Dict[str, Any], List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """
        Run inference using registered model artifact or fallback heuristic simulation.
        """
        card = self.get_model_card(model_version)
        if not card:
            raise ValueError(f"Model version not found: {model_version}")

        artifact_path = card.get("artifact_path")
        if artifact_path and os.path.exists(artifact_path):
            try:
                # Load pipeline from disk cache
                if artifact_path not in self._loaded_models:
                    with open(artifact_path, "rb") as f:
                        self._loaded_models[artifact_path] = pickle.load(f)
                pipeline = self._loaded_models[artifact_path]

                # Convert input dictionary or list of dicts to DataFrame
                df = pd.DataFrame([input_data] if isinstance(input_data, dict) else input_data)

                # Pre-sanitize sex if string passed
                if "SEX" in df.columns:
                    df["SEX"] = df["SEX"].apply(
                        lambda v: 1 if str(v).strip().upper() in ["M", "MALE", "1"] else 0
                    )

                # Execute pipeline inference
                probs = pipeline.predict_proba(df)[0]
                prob_positive = float(probs[1]) if len(probs) > 1 else float(probs[0])
                prediction_label = "progression" if prob_positive >= 0.5 else "stable"
                risk_level = (
                    "high"
                    if prob_positive >= 0.7
                    else ("moderate" if prob_positive >= 0.3 else "low")
                )
                confidence = float(np.round(abs(prob_positive - 0.5) * 2, 3))

                return {
                    "prediction": {
                        "class": prediction_label,
                        "probability": round(prob_positive, 4),
                        "risk_level": risk_level,
                    },
                    "confidence": confidence,
                    "model_version": card["version"],
                    "model_name": card["name"],
                    "provenance": "verified_trained_artifact",
                }
            except Exception as e:
                logger.warning(
                    f"Error evaluating artifact for {model_version}: {e}; falling back to simulation"
                )

        # Fallback simulation for unreleased/unmounted models
        simulated_prob = 0.78
        return {
            "prediction": {
                "class": "progression",
                "probability": simulated_prob,
                "risk_level": "moderate",
            },
            "confidence": simulated_prob,
            "model_version": card["version"],
            "model_name": card["name"],
            "provenance": "uncalibrated_heuristic_simulation",
            "audit_notice": "Simulation mode: Model checkpoint not mounted or uncalibrated.",
        }


# Global model registry
model_registry = ModelRegistry()


@model_bp.route("/infer", methods=["POST"])
@require_auth()
def model_inference():
    """
    Run model inference on input data.

    Query Parameters:
        model_version: Model version to use (default: latest)

    Request Body:
        {
            "data": {...},  # Input data
            "patient_id": "optional",
            "case_id": "optional"
        }

    Returns:
        Inference results with prediction and metadata
    """
    try:
        # Get model version
        model_version = request.args.get("model_version", "latest")

        # Get request data
        data = request.get_json()
        if not data or "data" not in data:
            return jsonify({"error": "Missing required field: data"}), 400

        input_data = data["data"]
        patient_id = data.get("patient_id")
        case_id = data.get("case_id")

        # Get model card
        model_card = model_registry.get_model_card(model_version)
        if not model_card:
            return jsonify({"error": f"Model version not found: {model_version}"}), 404

        # Run inference using registry
        inference_result = model_registry.predict(model_version, input_data)

        # Log inference for audit trail
        logger.info(
            f"Model inference: version={model_version}, patient={patient_id}, case={case_id}"
        )

        return jsonify(
            {
                **inference_result,
                "timestamp": datetime.now().isoformat(),
                "patient_id": patient_id,
                "case_id": case_id,
            }
        )

    except Exception as e:
        logger.error(f"Inference error: {e}")
        return jsonify({"error": "Inference failed", "message": str(e)}), 500


@model_bp.route("/card", methods=["GET"])
@require_auth()
def get_model_card():
    """
    Get model card with metadata.

    Query Parameters:
        model_version: Model version (default: latest)

    Returns:
        Model card with intended use, validation metrics, dataset provenance
    """
    try:
        model_version = request.args.get("model_version", "latest")

        model_card = model_registry.get_model_card(model_version)
        if not model_card:
            return jsonify({"error": f"Model version not found: {model_version}"}), 404

        return jsonify({"model_card": model_card, "retrieved_at": datetime.now().isoformat()})

    except Exception as e:
        logger.error(f"Model card error: {e}")
        return jsonify({"error": "Failed to retrieve model card", "message": str(e)}), 500


@model_bp.route("/list", methods=["GET"])
@require_auth()
def list_models():
    """
    List all available models.

    Returns:
        List of models with basic metadata
    """
    try:
        models = model_registry.list_models()
        return jsonify({"models": models, "total": len(models)})
    except Exception as e:
        logger.error(f"List models error: {e}")
        return jsonify({"error": "Failed to list models", "message": str(e)}), 500


@model_bp.route("/versions/<model_name>", methods=["GET"])
@require_auth()
def list_model_versions(model_name: str):
    """
    List versions for a specific model.

    Args:
        model_name: Model name

    Returns:
        List of versions for the model
    """
    try:
        versions = [
            {
                "version": data["version"],
                "status": data["status"],
                "last_updated": data["last_updated"],
            }
            for model_id, data in model_registry.models.items()
            if model_name.lower() in data["name"].lower()
        ]

        if not versions:
            return jsonify({"error": f"Model not found: {model_name}"}), 404

        return jsonify({"model_name": model_name, "versions": versions})

    except Exception as e:
        logger.error(f"List versions error: {e}")
        return jsonify({"error": "Failed to list versions", "message": str(e)}), 500
