"""DICOM API routes for CT upload, preprocessing, and analysis."""

import logging
from typing import Any, Dict

import numpy as np
from flask import Blueprint, jsonify, request
from pydicom.errors import InvalidDicomError

from ..integration.dicom_handler import DICOMMetadata, DICOMWorkflowManager
from ..security.auth import require_auth
from .server import rate_limit

logger = logging.getLogger(__name__)

dicom_bp = Blueprint("dicom", __name__, url_prefix="/api/v1/dicom")

dicom_workflow = DICOMWorkflowManager(config={})

# In-memory DICOM study store (ephemeral, per-process)
DICOM_STUDY_STORE: Dict[str, Dict[str, Any]] = {}
MAX_STORED_STUDIES = 50


@dicom_bp.route("/upload", methods=["POST"])
@rate_limit(limit=10, window=60)
@require_auth()
def upload_dicom_series():
    """Upload and process a CT DICOM series."""
    try:
        files = request.files.getlist("files[]")
        if not files:
            return jsonify({"error": "No DICOM files provided in files[]"}), 400

        if len(files) > 500:
            return jsonify({"error": "Maximum 500 DICOM slices per upload"}), 400

        file_bytes_list = []
        for file_obj in files:
            file_bytes = file_obj.read()
            if not file_bytes:
                return jsonify({"error": f"Empty file detected: {file_obj.filename}"}), 400

            is_valid, reason = dicom_workflow.validate_dicom_file(file_bytes)
            if not is_valid:
                return (
                    jsonify({"error": "Invalid DICOM file", "file": file_obj.filename, "reason": reason}),
                    400,
                )
            file_bytes_list.append(file_bytes)

        result = dicom_workflow.process_dicom_upload(file_bytes_list)

        study_uid = result["study_uid"]
        metadata_items = result.get("metadata", [])
        first_meta_dict = metadata_items[0] if metadata_items else None
        first_meta = None
        if first_meta_dict:
            pixel_spacing = first_meta_dict.get("pixel_spacing", [1.0, 1.0])
            if not isinstance(pixel_spacing, (list, tuple)):
                pixel_spacing = [1.0, 1.0]

            first_meta = DICOMMetadata(
                patient_id=str(first_meta_dict.get("patient_id", "")),
                study_instance_uid=str(first_meta_dict.get("study_instance_uid", "")),
                series_instance_uid=str(first_meta_dict.get("series_instance_uid", "")),
                sop_instance_uid=str(first_meta_dict.get("sop_instance_uid", "")),
                modality=str(first_meta_dict.get("modality", "")),
                study_date=str(first_meta_dict.get("study_date", "")),
                study_description=str(first_meta_dict.get("study_description", "")),
                series_description=str(first_meta_dict.get("series_description", "")),
                rows=int(first_meta_dict.get("rows", 0) or 0),
                columns=int(first_meta_dict.get("columns", 0) or 0),
                slice_thickness=float(first_meta_dict.get("slice_thickness", 0.0) or 0.0),
                pixel_spacing=list(pixel_spacing),
                instance_number=int(first_meta_dict.get("instance_number", 0) or 0),
                window_center=float(first_meta_dict.get("window_center", 40.0) or 40.0),
                window_width=float(first_meta_dict.get("window_width", 400.0) or 400.0),
                manufacturer=str(first_meta_dict.get("manufacturer", "")),
                body_part_examined=str(first_meta_dict.get("body_part_examined", "")),
            )

        fhir_resource = None
        if first_meta:
            patient_id = first_meta.patient_id or "anonymous"
            fhir_resource = dicom_workflow.generate_fhir_imaging_study(first_meta, patient_id)

        if study_uid not in DICOM_STUDY_STORE and len(DICOM_STUDY_STORE) >= MAX_STORED_STUDIES:
            oldest_study_uid = next(iter(DICOM_STUDY_STORE))
            del DICOM_STUDY_STORE[oldest_study_uid]

        DICOM_STUDY_STORE[study_uid] = {
            "preprocessed_volume": result["preprocessed_volume"],
            "raw_volume_shape": result["volume_shape"],
            "preview_slices": result["preview_slices"],
            "metadata": result["metadata"],
            "fhir_imaging_study": fhir_resource,
        }

        return (
            jsonify(
                {
                    "status": "success",
                    "study_uid": result["study_uid"],
                    "series_uid": result["series_uid"],
                    "slice_count": result["slice_count"],
                    "volume_shape": result["volume_shape"],
                    "preview_slices": result["preview_slices"],
                    "metadata": result["metadata"],
                }
            ),
            200,
        )

    except InvalidDicomError:
        return jsonify({"error": "Invalid DICOM format"}), 400
    except Exception as exc:
        logger.error("DICOM upload failed: %s", exc, exc_info=True)
        return jsonify({"error": "DICOM upload failed"}), 500


@dicom_bp.route("/analyze", methods=["POST"])
@rate_limit(limit=20, window=60)
@require_auth()
def analyze_dicom_study():
    """Apply analysis windowing and return inference readiness stats."""
    try:
        data = request.get_json() or {}
        study_uid = data.get("study_uid")
        if not study_uid:
            return jsonify({"error": "study_uid is required"}), 400

        study_data = DICOM_STUDY_STORE.get(study_uid)
        if not study_data:
            return jsonify({"error": "Study not found"}), 404

        window_preset = data.get("window_preset", "soft_tissue")
        selected_model = data.get("model", "default_ct_model")

        preprocessed_volume = study_data["preprocessed_volume"]
        center, width = dicom_workflow.processor.get_window_preset(window_preset)
        windowed_volume = dicom_workflow.processor.apply_windowing(preprocessed_volume, center, width)

        volume_stats = {
            "min": float(np.min(windowed_volume)),
            "max": float(np.max(windowed_volume)),
            "mean": float(np.mean(windowed_volume)),
            "std": float(np.std(windowed_volume)),
        }

        return (
            jsonify(
                {
                    "status": "analyzed",
                    "study_uid": study_uid,
                    "window_preset": window_preset,
                    "model": selected_model,
                    "volume_stats": volume_stats,
                    "inference_ready": True,
                }
            ),
            200,
        )

    except Exception as exc:
        logger.error("DICOM analyze failed: %s", exc, exc_info=True)
        return jsonify({"error": "DICOM analysis failed"}), 500


@dicom_bp.route("/preview/<study_uid>", methods=["GET"])
@rate_limit(limit=30, window=60)
@require_auth()
def preview_dicom_study(study_uid: str):
    """Return preview slices for a stored CT study."""
    study_data = DICOM_STUDY_STORE.get(study_uid)
    if not study_data:
        return jsonify({"error": "Study not found"}), 404

    return jsonify({"study_uid": study_uid, "preview_slices": study_data.get("preview_slices", [])}), 200


@dicom_bp.route("/metadata/<study_uid>", methods=["GET"])
@rate_limit(limit=30, window=60)
@require_auth()
def metadata_dicom_study(study_uid: str):
    """Return FHIR ImagingStudy metadata for a stored CT study."""
    study_data = DICOM_STUDY_STORE.get(study_uid)
    if not study_data:
        return jsonify({"error": "Study not found"}), 404

    fhir_imaging_study = study_data.get("fhir_imaging_study")
    if not fhir_imaging_study:
        return jsonify({"error": "ImagingStudy metadata not available"}), 404

    return jsonify(fhir_imaging_study), 200


@dicom_bp.route("/health", methods=["GET"])
@rate_limit(limit=30, window=60)
def dicom_health_check():
    """Health endpoint for DICOM feature support."""
    try:
        import pydicom  # noqa: F401

        pydicom_available = True
    except Exception:
        pydicom_available = False

    return (
        jsonify(
            {
                "status": "ok",
                "dicom_support": True,
                "pydicom_available": pydicom_available,
            }
        ),
        200,
    )
