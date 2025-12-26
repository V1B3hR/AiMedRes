"""
Mock API Server for Frontend Development.

Provides mock responses for all API endpoints to enable
frontend development without requiring full backend.
"""

import json
import logging
from datetime import datetime

from flask import Flask, jsonify, request

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Mock data stores
mock_cases = {
    "case-001": {
        "case_id": "case-001",
        "patient_id": "patient-001",
        "status": "pending",
        "risk_level": "high",
        "prediction": {"class": "positive", "probability": 0.82},
        "created_at": "2025-01-20T10:00:00Z",
    },
    "case-002": {
        "case_id": "case-002",
        "patient_id": "patient-002",
        "status": "in_review",
        "risk_level": "moderate",
        "prediction": {"class": "positive", "probability": 0.65},
        "created_at": "2025-01-19T14:30:00Z",
    },
}

mock_patients = {
    "patient-001": {
        "id": "patient-001",
        "name": "REDACTED",
        "gender": "male",
        "birthDate": "1950-01-15",
    },
    "patient-002": {
        "id": "patient-002",
        "name": "REDACTED",
        "gender": "female",
        "birthDate": "1948-05-22",
    },
}


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify(
        {"status": "healthy", "timestamp": datetime.now().isoformat(), "version": "1.0.0-mock"}
    )


@app.route("/api/v1/auth/login", methods=["POST"])
def login():
    """Mock login endpoint."""
    data = request.get_json()

    return jsonify(
        {
            "access_token": "mock_token_123456",
            "token_type": "Bearer",
            "expires_in": 3600,
            "roles": ["clinician", "user"],
        }
    )


@app.route("/api/v1/cases", methods=["GET"])
def list_cases():
    """Mock case listing."""
    status = request.args.get("status")

    cases = list(mock_cases.values())
    if status:
        cases = [c for c in cases if c["status"] == status]

    return jsonify({"cases": cases, "total": len(cases), "page": 1, "per_page": 20})


@app.route("/api/v1/cases/<case_id>", methods=["GET"])
def get_case(case_id):
    """Mock case detail."""
    case = mock_cases.get(case_id)

    if not case:
        return jsonify({"error": "Case not found"}), 404

    # Add explainability
    case_with_explain = case.copy()
    case_with_explain["explainability"] = {
        "attributions": [
            {"feature": "Age", "importance": 0.25, "value": 72},
            {"feature": "MMSE Score", "importance": 0.35, "value": 24},
        ],
        "uncertainty": {"confidence": case["prediction"]["probability"], "total_uncertainty": 0.15},
    }

    return jsonify(case_with_explain)


@app.route("/api/v1/cases/<case_id>/approve", methods=["POST"])
def approve_case(case_id):
    """Mock case approval."""
    data = request.get_json()

    if case_id not in mock_cases:
        return jsonify({"error": "Case not found"}), 404

    action = data.get("action")
    rationale = data.get("rationale")

    if not action or not rationale:
        return jsonify({"error": "Missing required fields"}), 400

    # Update mock case
    mock_cases[case_id]["status"] = "completed" if action == "approve" else "rejected"
    mock_cases[case_id]["approved_by"] = "mock-clinician"
    mock_cases[case_id]["approved_at"] = datetime.now().isoformat()

    return jsonify(
        {
            "case_id": case_id,
            "status": mock_cases[case_id]["status"],
            "action": action,
            "approved_by": "mock-clinician",
            "approved_at": mock_cases[case_id]["approved_at"],
        }
    )


@app.route("/api/v1/model/infer", methods=["POST"])
def model_inference():
    """Mock model inference."""
    data = request.get_json()
    model_version = request.args.get("model_version", "latest")

    return jsonify(
        {
            "prediction": {"class": "positive", "probability": 0.78, "risk_level": "moderate"},
            "confidence": 0.78,
            "model_version": model_version,
            "timestamp": datetime.now().isoformat(),
        }
    )


@app.route("/api/v1/model/card", methods=["GET"])
def model_card():
    """Mock model card."""
    return jsonify(
        {
            "model_card": {
                "name": "Alzheimer Early Detection",
                "version": "v1.0.0",
                "intended_use": "Early detection of Alzheimer's disease",
                "validation_metrics": {"accuracy": 0.89, "sensitivity": 0.92, "specificity": 0.87},
                "dataset_provenance": "ADNI dataset",
                "limitations": ["Not approved for clinical diagnosis"],
            }
        }
    )


@app.route("/api/v1/explain/attribution", methods=["POST"])
def feature_attribution():
    """Mock feature attribution."""
    data = request.get_json()

    return jsonify(
        {
            "prediction_id": data.get("prediction_id"),
            "attributions": [
                {"feature": "Age", "importance": 0.25, "contribution": 0.15},
                {"feature": "MMSE Score", "importance": 0.35, "contribution": -0.18},
                {"feature": "Hippocampal Volume", "importance": 0.20, "contribution": 0.12},
            ],
            "timestamp": datetime.now().isoformat(),
        }
    )


@app.route("/api/v1/explain/uncertainty", methods=["POST"])
def uncertainty():
    """Mock uncertainty estimation."""
    data = request.get_json()

    return jsonify(
        {
            "prediction_id": data.get("prediction_id"),
            "uncertainty_metrics": {
                "confidence": 0.78,
                "total_uncertainty": 0.20,
                "epistemic_uncertainty": 0.12,
                "aleatoric_uncertainty": 0.08,
            },
            "timestamp": datetime.now().isoformat(),
        }
    )


@app.route("/api/v1/fhir/patients", methods=["GET"])
def fhir_patients():
    """Mock FHIR patients list."""
    return jsonify(
        {
            "resourceType": "Bundle",
            "type": "searchset",
            "total": len(mock_patients),
            "entry": [{"resource": patient} for patient in mock_patients.values()],
        }
    )


@app.route("/api/v1/fhir/patients/<patient_id>", methods=["GET"])
def fhir_patient(patient_id):
    """Mock FHIR patient detail."""
    patient = mock_patients.get(patient_id)

    if not patient:
        return jsonify({"error": "Patient not found"}), 404

    return jsonify({"resourceType": "Patient", **patient})


@app.route("/api/v1/audit/logs", methods=["GET"])
def audit_logs():
    """Mock audit logs."""
    return jsonify(
        {
            "logs": [
                {
                    "timestamp": "2025-01-24T10:30:00Z",
                    "user_id": "clinician-001",
                    "action": "case_approve",
                    "resource": "case-001",
                    "details": {"rationale": "Clinical review completed"},
                },
                {
                    "timestamp": "2025-01-24T09:15:00Z",
                    "user_id": "system",
                    "action": "model_inference",
                    "resource": "case-002",
                    "details": {"model_version": "v1.0.0"},
                },
            ],
            "total": 2,
        }
    )


@app.route("/api/v1/audit/export", methods=["POST"])
def export_audit():
    """Mock audit export."""
    data = request.get_json()

    return jsonify(
        {
            "export_id": "export-123456",
            "download_url": "/api/v1/audit/exports/export-123456",
            "expires_at": "2025-01-25T12:00:00Z",
        }
    )


if __name__ == "__main__":
    print("=" * 80)
    print("AiMedRes Mock API Server")
    print("=" * 80)
    print("\nStarting server on http://localhost:3001")
    print("\nAvailable endpoints:")
    print("  - GET  /health")
    print("  - POST /api/v1/auth/login")
    print("  - GET  /api/v1/cases")
    print("  - GET  /api/v1/cases/<case_id>")
    print("  - POST /api/v1/cases/<case_id>/approve")
    print("  - POST /api/v1/model/infer")
    print("  - GET  /api/v1/model/card")
    print("  - POST /api/v1/explain/attribution")
    print("  - POST /api/v1/explain/uncertainty")
    print("  - GET  /api/v1/fhir/patients")
    print("  - GET  /api/v1/fhir/patients/<patient_id>")
    print("  - GET  /api/v1/audit/logs")
    print("  - POST /api/v1/audit/export")
    print("\nNo authentication required for mock server")
    print("=" * 80)
    print()

    app.run(host="0.0.0.0", port=3001, debug=True)
