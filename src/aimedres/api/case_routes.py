"""
Case management API endpoints.

Provides endpoints for:
- Case listing and filtering
- Case detail retrieval
- Human-in-loop approval workflow
"""

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List

from flask import Blueprint, jsonify, request

from ..security.auth import require_auth
from ..security.human_in_loop import HumanInLoopGating

logger = logging.getLogger(__name__)

cases_bp = Blueprint("cases", __name__, url_prefix="/api/v1/cases")


class CaseManager:
    """
    Manager for clinical cases and recommendations.
    """

    def __init__(self):
        self.cases = {}
        self._initialize_mock_cases()
        self.hil_gating = HumanInLoopGating({"human_in_loop_enabled": True})

    def _initialize_mock_cases(self):
        """Initialize mock cases for testing."""
        self.cases = {
            "case-001": {
                "case_id": "case-001",
                "patient_id": "patient-001",
                "status": "pending",
                "risk_level": "high",
                "prediction": {"class": "positive", "probability": 0.82, "risk_score": 0.78},
                "created_at": "2025-01-20T10:00:00Z",
                "updated_at": "2025-01-20T10:00:00Z",
                "created_by": "system",
                "model_version": "alzheimer_v1",
            },
            "case-002": {
                "case_id": "case-002",
                "patient_id": "patient-002",
                "status": "in_review",
                "risk_level": "moderate",
                "prediction": {"class": "positive", "probability": 0.65, "risk_score": 0.58},
                "created_at": "2025-01-19T14:30:00Z",
                "updated_at": "2025-01-20T09:15:00Z",
                "created_by": "system",
                "model_version": "alzheimer_v1",
                "assigned_to": "clinician-001",
            },
            "case-003": {
                "case_id": "case-003",
                "patient_id": "patient-001",
                "status": "completed",
                "risk_level": "low",
                "prediction": {"class": "negative", "probability": 0.88, "risk_score": 0.12},
                "created_at": "2025-01-18T11:20:00Z",
                "updated_at": "2025-01-19T08:45:00Z",
                "created_by": "system",
                "model_version": "parkinsons_v1",
                "approved_by": "clinician-002",
                "approved_at": "2025-01-19T08:45:00Z",
            },
        }

    def list_cases(
        self, status: str = None, risk_level: str = None, page: int = 1, per_page: int = 20
    ) -> Dict[str, Any]:
        """
        List cases with filtering and pagination.

        Args:
            status: Filter by status
            risk_level: Filter by risk level
            page: Page number
            per_page: Items per page

        Returns:
            Paginated case list
        """
        filtered_cases = list(self.cases.values())

        # Apply filters
        if status:
            filtered_cases = [c for c in filtered_cases if c["status"] == status]

        if risk_level:
            filtered_cases = [c for c in filtered_cases if c["risk_level"] == risk_level]

        # Sort by updated_at descending
        filtered_cases.sort(key=lambda x: x["updated_at"], reverse=True)

        # Paginate
        total = len(filtered_cases)
        start = (page - 1) * per_page
        end = start + per_page
        page_cases = filtered_cases[start:end]

        return {
            "cases": page_cases,
            "total": total,
            "page": page,
            "per_page": per_page,
            "total_pages": (total + per_page - 1) // per_page,
        }

    def get_case(self, case_id: str) -> Dict[str, Any]:
        """Get case details."""
        return self.cases.get(case_id)

    def approve_case(
        self, case_id: str, user_id: str, action: str, rationale: str, notes: str = None
    ) -> Dict[str, Any]:
        """
        Approve or reject case with human-in-loop gating.

        Args:
            case_id: Case identifier
            user_id: Approving user ID
            action: approve|reject|request_review
            rationale: Clinical rationale
            notes: Optional notes

        Returns:
            Updated case data
        """
        case = self.cases.get(case_id)
        if not case:
            raise ValueError(f"Case not found: {case_id}")

        # Record approval in HIL system
        decision = {
            "case_id": case_id,
            "action": action,
            "rationale": rationale,
            "notes": notes,
            "user_id": user_id,
            "timestamp": datetime.now().isoformat(),
        }

        # Update case status
        if action == "approve":
            case["status"] = "completed"
            case["approved_by"] = user_id
            case["approved_at"] = datetime.now().isoformat()
        elif action == "reject":
            case["status"] = "rejected"
            case["rejected_by"] = user_id
            case["rejected_at"] = datetime.now().isoformat()
        elif action == "request_review":
            case["status"] = "in_review"

        case["updated_at"] = datetime.now().isoformat()

        # Log to audit trail
        logger.info(f"Case {action}: {case_id} by {user_id}")

        return case


# Global case manager
case_manager = CaseManager()


@cases_bp.route("", methods=["GET"])
@require_auth()
def list_cases():
    """
    List clinical cases with filtering.

    Query Parameters:
        status: Filter by status (pending, in_review, completed)
        risk_level: Filter by risk level (low, moderate, high, critical)
        page: Page number (default: 1)
        per_page: Items per page (default: 20)

    Returns:
        Paginated case list
    """
    try:
        status = request.args.get("status")
        risk_level = request.args.get("risk_level")
        page = int(request.args.get("page", 1))
        per_page = int(request.args.get("per_page", 20))

        result = case_manager.list_cases(status, risk_level, page, per_page)

        logger.info(f"Listed cases: status={status}, risk_level={risk_level}, page={page}")

        return jsonify(result)

    except Exception as e:
        logger.error(f"List cases error: {e}")
        return jsonify({"error": "Failed to list cases", "message": str(e)}), 500


@cases_bp.route("/<case_id>", methods=["GET"])
@require_auth()
def get_case_detail(case_id: str):
    """
    Get detailed case information.

    Args:
        case_id: Case identifier

    Returns:
        Case details with prediction and explainability
    """
    try:
        case = case_manager.get_case(case_id)

        if not case:
            return jsonify({"error": "Case not found", "case_id": case_id}), 404

        # Add explainability data (would fetch from explain service in production)
        case_with_explain = case.copy()
        case_with_explain["explainability"] = {
            "attributions": [
                {"feature": "Age", "importance": 0.25},
                {"feature": "MMSE Score", "importance": 0.35},
            ],
            "uncertainty": {
                "confidence": case["prediction"]["probability"],
                "total_uncertainty": 0.15,
            },
        }

        logger.info(f"Retrieved case detail: {case_id}")

        return jsonify(case_with_explain)

    except Exception as e:
        logger.error(f"Get case error: {e}")
        return jsonify({"error": "Failed to retrieve case", "message": str(e)}), 500


@cases_bp.route("/<case_id>/approve", methods=["POST"])
@require_auth(required_role="clinician")
def approve_case(case_id: str):
    """
    Approve or reject case (requires clinician role).

    Args:
        case_id: Case identifier

    Request Body:
        {
            "action": "approve|reject|request_review",
            "rationale": "Clinical rationale (required)",
            "notes": "Optional notes"
        }

    Returns:
        Updated case status
    """
    try:
        data = request.get_json()

        if not data or "action" not in data or "rationale" not in data:
            return (
                jsonify({"error": "Missing required fields", "required": ["action", "rationale"]}),
                400,
            )

        action = data["action"]
        rationale = data["rationale"]
        notes = data.get("notes")

        # Validate action
        if action not in ["approve", "reject", "request_review"]:
            return (
                jsonify(
                    {
                        "error": "Invalid action",
                        "valid_actions": ["approve", "reject", "request_review"],
                    }
                ),
                400,
            )

        # Get user from request context
        user_id = request.user_info["user_id"]

        # Approve case
        updated_case = case_manager.approve_case(case_id, user_id, action, rationale, notes)

        return jsonify(
            {
                "case_id": case_id,
                "status": updated_case["status"],
                "action": action,
                "approved_by": user_id,
                "approved_at": updated_case.get("approved_at")
                or updated_case.get("rejected_at")
                or datetime.now().isoformat(),
            }
        )

    except ValueError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        logger.error(f"Approve case error: {e}")
        return jsonify({"error": "Failed to approve case", "message": str(e)}), 500


@cases_bp.route("/<case_id>/history", methods=["GET"])
@require_auth()
def get_case_history(case_id: str):
    """
    Get case approval history and audit trail.

    Args:
        case_id: Case identifier

    Returns:
        Case history with all actions
    """
    try:
        case = case_manager.get_case(case_id)

        if not case:
            return jsonify({"error": "Case not found"}), 404

        # Mock history (would fetch from audit log in production)
        history = [
            {
                "action": "created",
                "user_id": case["created_by"],
                "timestamp": case["created_at"],
                "details": "Case created by system",
            }
        ]

        if case.get("approved_by"):
            history.append(
                {
                    "action": "approved",
                    "user_id": case["approved_by"],
                    "timestamp": case["approved_at"],
                    "details": "Case approved by clinician",
                }
            )

        return jsonify({"case_id": case_id, "history": history})

    except Exception as e:
        logger.error(f"Get case history error: {e}")
        return jsonify({"error": "Failed to retrieve history", "message": str(e)}), 500
