"""
API Routes for Advanced Multimodal Viewers (P3-1)

Provides endpoints for:
- 3D Brain Visualization
- DICOM Viewer Integration
- Imaging Data Streaming
- Explainability Overlays
"""

from flask import Blueprint, request, jsonify, send_file, Response
from typing import Dict, Any, Optional, List
import logging
import io
import json
from datetime import datetime

from ..dashboards.brain_visualization import (
    BrainVisualizationEngine,
    BrainRegion,
    DiseaseStage,
    TreatmentType,
    VisualizationMode
)
from ..security.auth import require_auth

logger = logging.getLogger('aimedres.api.visualization_routes')

# Create blueprint
visualization_bp = Blueprint('visualization', __name__, url_prefix='/api/v1/visualization')

# Initialize brain visualization engine
brain_engine = BrainVisualizationEngine(enable_real_time=True, cache_size=1000)


# ==================== Brain Visualization Endpoints ====================

@visualization_bp.route('/brain/overlay', methods=['POST'])
@require_auth
def create_brain_overlay():
    """
    Create 3D anatomical brain overlay.
    
    Request body:
    {
        "patient_id": "patient-123",
        "regions_of_interest": ["frontal_lobe", "hippocampus"],
        "highlight_abnormalities": true
    }
    
    Returns:
        Overlay visualization data with markers
    """
    try:
        data = request.get_json()
        patient_id = data.get('patient_id')
        regions = data.get('regions_of_interest', [])
        highlight = data.get('highlight_abnormalities', True)
        
        if not patient_id:
            return jsonify({"error": "patient_id is required"}), 400
        
        # Convert string region names to BrainRegion enums
        region_enums = []
        for r in regions:
            try:
                region_enums.append(BrainRegion[r.upper()])
            except (KeyError, AttributeError):
                logger.warning(f"Invalid region: {r}")
        
        if not region_enums:
            return jsonify({"error": "No valid regions_of_interest provided"}), 400
        
        # Create overlay
        overlay_data = brain_engine.create_anatomical_overlay(
            patient_id=patient_id,
            regions_of_interest=region_enums,
            highlight_abnormalities=highlight
        )
        
        return jsonify({
            "success": True,
            "data": overlay_data
        }), 200
        
    except Exception as e:
        logger.error(f"Error creating brain overlay: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@visualization_bp.route('/brain/disease-map', methods=['POST'])
@require_auth
def create_disease_map():
    """
    Map disease pathology across brain regions.
    
    Request body:
    {
        "patient_id": "patient-123",
        "disease_type": "alzheimers",
        "severity_map": {
            "hippocampus": 0.8,
            "temporal_lobe": 0.6
        }
    }
    
    Returns:
        Disease pathology mapping data
    """
    try:
        data = request.get_json()
        patient_id = data.get('patient_id')
        disease_type = data.get('disease_type')
        severity_map_raw = data.get('severity_map', {})
        
        if not patient_id or not disease_type:
            return jsonify({"error": "patient_id and disease_type are required"}), 400
        
        # Convert severity map to use BrainRegion enums
        severity_map = {}
        for region_str, severity in severity_map_raw.items():
            try:
                region_enum = BrainRegion[region_str.upper()]
                severity_map[region_enum] = float(severity)
            except (KeyError, ValueError, AttributeError):
                logger.warning(f"Invalid region or severity: {region_str}={severity}")
        
        if not severity_map:
            return jsonify({"error": "No valid severity_map provided"}), 400
        
        # Create disease map
        pathology_data = brain_engine.map_disease_pathology(
            patient_id=patient_id,
            disease_type=disease_type,
            severity_map=severity_map
        )
        
        return jsonify({
            "success": True,
            "data": pathology_data
        }), 200
        
    except Exception as e:
        logger.error(f"Error creating disease map: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@visualization_bp.route('/brain/progression-snapshot', methods=['POST'])
@require_auth
def capture_progression_snapshot():
    """
    Capture disease progression snapshot.
    
    Request body:
    {
        "patient_id": "patient-123",
        "stage": "moderate",
        "affected_regions": {
            "hippocampus": 0.7,
            "temporal_lobe": 0.5
        },
        "biomarkers": {"tau": 450, "beta_amyloid": 320},
        "cognitive_scores": {"mmse": 18, "moca": 15}
    }
    
    Returns:
        Progression snapshot data
    """
    try:
        data = request.get_json()
        patient_id = data.get('patient_id')
        stage_str = data.get('stage')
        affected_regions_raw = data.get('affected_regions', {})
        biomarkers = data.get('biomarkers', {})
        cognitive_scores = data.get('cognitive_scores', {})
        
        if not patient_id or not stage_str:
            return jsonify({"error": "patient_id and stage are required"}), 400
        
        # Convert stage to enum
        try:
            stage = DiseaseStage[stage_str.upper()]
        except KeyError:
            return jsonify({"error": f"Invalid stage: {stage_str}"}), 400
        
        # Convert affected regions
        affected_regions = {}
        for region_str, severity in affected_regions_raw.items():
            try:
                region_enum = BrainRegion[region_str.upper()]
                affected_regions[region_enum] = float(severity)
            except (KeyError, ValueError, AttributeError):
                logger.warning(f"Invalid region or severity: {region_str}={severity}")
        
        if not affected_regions:
            return jsonify({"error": "No valid affected_regions provided"}), 400
        
        # Capture snapshot
        snapshot = brain_engine.capture_progression_snapshot(
            patient_id=patient_id,
            stage=stage,
            affected_regions=affected_regions,
            biomarkers=biomarkers,
            cognitive_scores=cognitive_scores
        )
        
        return jsonify({
            "success": True,
            "data": {
                "snapshot_id": snapshot.snapshot_id,
                "patient_id": snapshot.patient_id,
                "timestamp": snapshot.timestamp.isoformat(),
                "stage": snapshot.stage.value,
                "affected_regions": {k.value: v for k, v in snapshot.affected_regions.items()},
                "biomarkers": snapshot.biomarkers,
                "cognitive_scores": snapshot.cognitive_scores,
                "volumetric_data": snapshot.volumetric_data
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error capturing progression snapshot: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@visualization_bp.route('/brain/temporal-progression', methods=['POST'])
@require_auth
def visualize_temporal_progression():
    """
    Visualize disease progression over time.
    
    Request body:
    {
        "patient_id": "patient-123",
        "snapshot_ids": ["snap-1", "snap-2", "snap-3"],
        "time_scale": "months"
    }
    
    Returns:
        Temporal progression visualization data
    """
    try:
        data = request.get_json()
        patient_id = data.get('patient_id')
        snapshot_ids = data.get('snapshot_ids', [])
        time_scale = data.get('time_scale', 'months')
        
        if not patient_id or not snapshot_ids:
            return jsonify({"error": "patient_id and snapshot_ids are required"}), 400
        
        # Create temporal visualization
        viz_data = brain_engine.visualize_temporal_progression(
            patient_id=patient_id,
            snapshots=snapshot_ids,
            time_scale=time_scale
        )
        
        return jsonify({
            "success": True,
            "data": viz_data
        }), 200
        
    except Exception as e:
        logger.error(f"Error visualizing temporal progression: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@visualization_bp.route('/brain/treatment-simulation', methods=['POST'])
@require_auth
def simulate_treatment():
    """
    Simulate treatment impact.
    
    Request body:
    {
        "patient_id": "patient-123",
        "baseline_snapshot_id": "snap-123",
        "treatment_type": "medication",
        "duration_days": 180,
        "efficacy_rate": 0.7
    }
    
    Returns:
        Treatment simulation results
    """
    try:
        data = request.get_json()
        patient_id = data.get('patient_id')
        baseline_snapshot_id = data.get('baseline_snapshot_id')
        treatment_type_str = data.get('treatment_type')
        duration_days = data.get('duration_days', 180)
        efficacy_rate = data.get('efficacy_rate', 0.7)
        
        if not patient_id or not baseline_snapshot_id or not treatment_type_str:
            return jsonify({"error": "patient_id, baseline_snapshot_id, and treatment_type are required"}), 400
        
        # Convert treatment type to enum
        try:
            treatment_type = TreatmentType[treatment_type_str.upper()]
        except KeyError:
            return jsonify({"error": f"Invalid treatment_type: {treatment_type_str}"}), 400
        
        # Run simulation
        simulation = brain_engine.simulate_treatment_impact(
            patient_id=patient_id,
            baseline_snapshot_id=baseline_snapshot_id,
            treatment_type=treatment_type,
            duration_days=int(duration_days),
            efficacy_rate=float(efficacy_rate)
        )
        
        return jsonify({
            "success": True,
            "data": {
                "simulation_id": simulation.simulation_id,
                "patient_id": simulation.patient_id,
                "treatment_type": simulation.treatment_type.value,
                "duration_days": simulation.duration_days,
                "baseline_snapshot": simulation.baseline_snapshot,
                "projected_outcomes": simulation.projected_outcomes,
                "confidence_interval": simulation.confidence_interval,
                "success_probability": simulation.success_probability,
                "side_effects": simulation.side_effects
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error simulating treatment: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@visualization_bp.route('/brain/compare-treatments', methods=['POST'])
@require_auth
def compare_treatments():
    """
    Compare multiple treatment scenarios.
    
    Request body:
    {
        "patient_id": "patient-123",
        "simulation_ids": ["sim-1", "sim-2", "sim-3"]
    }
    
    Returns:
        Treatment comparison analysis
    """
    try:
        data = request.get_json()
        patient_id = data.get('patient_id')
        simulation_ids = data.get('simulation_ids', [])
        
        if not patient_id or not simulation_ids:
            return jsonify({"error": "patient_id and simulation_ids are required"}), 400
        
        # Compare scenarios
        comparison = brain_engine.compare_treatment_scenarios(
            patient_id=patient_id,
            simulation_ids=simulation_ids
        )
        
        return jsonify({
            "success": True,
            "data": comparison
        }), 200
        
    except Exception as e:
        logger.error(f"Error comparing treatments: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@visualization_bp.route('/brain/educational-module', methods=['POST'])
@require_auth
def create_educational_module():
    """
    Create educational training module.
    
    Request body:
    {
        "title": "Introduction to Brain Anatomy",
        "description": "Learn basic brain structures",
        "difficulty_level": "beginner",
        "target_audience": "student",
        "brain_regions": ["frontal_lobe", "hippocampus"],
        "learning_objectives": ["Identify major brain regions", "Understand functions"],
        "completion_time_minutes": 30
    }
    
    Returns:
        Educational module data
    """
    try:
        data = request.get_json()
        title = data.get('title')
        description = data.get('description')
        difficulty_level = data.get('difficulty_level', 'beginner')
        target_audience = data.get('target_audience', 'student')
        regions_raw = data.get('brain_regions', [])
        learning_objectives = data.get('learning_objectives', [])
        completion_time = data.get('completion_time_minutes', 30)
        
        if not title or not description:
            return jsonify({"error": "title and description are required"}), 400
        
        # Convert regions
        brain_regions = []
        for r in regions_raw:
            try:
                brain_regions.append(BrainRegion[r.upper()])
            except KeyError:
                logger.warning(f"Invalid region: {r}")
        
        if not brain_regions:
            return jsonify({"error": "No valid brain_regions provided"}), 400
        
        # Create module
        module = brain_engine.create_educational_module(
            title=title,
            description=description,
            difficulty_level=difficulty_level,
            target_audience=target_audience,
            brain_regions=brain_regions,
            learning_objectives=learning_objectives,
            completion_time_minutes=int(completion_time)
        )
        
        return jsonify({
            "success": True,
            "data": {
                "module_id": module.module_id,
                "title": module.title,
                "description": module.description,
                "difficulty_level": module.difficulty_level,
                "target_audience": module.target_audience,
                "brain_regions": [r.value for r in module.brain_regions],
                "learning_objectives": module.learning_objectives,
                "interactive_elements": module.interactive_elements,
                "completion_time_minutes": module.completion_time_minutes,
                "assessment_questions": module.assessment_questions
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error creating educational module: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@visualization_bp.route('/brain/statistics', methods=['GET'])
@require_auth
def get_brain_statistics():
    """
    Get brain visualization engine statistics.
    
    Returns:
        Engine statistics and metrics
    """
    try:
        stats = brain_engine.get_statistics()
        
        return jsonify({
            "success": True,
            "data": stats
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting statistics: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


# ==================== DICOM Viewer Endpoints ====================

@visualization_bp.route('/dicom/series', methods=['GET'])
@require_auth
def list_dicom_series():
    """
    List available DICOM series for a patient.
    
    Query params:
        patient_id: Patient identifier
    
    Returns:
        List of DICOM series metadata
    """
    try:
        patient_id = request.args.get('patient_id')
        
        if not patient_id:
            return jsonify({"error": "patient_id query parameter is required"}), 400
        
        # Mock data - in production, this would query a PACS or imaging database
        series_list = [
            {
                "series_id": "series-001",
                "patient_id": patient_id,
                "modality": "MR",
                "series_description": "T1 MPRAGE",
                "study_date": "2024-01-15",
                "num_instances": 176,
                "series_number": 3
            },
            {
                "series_id": "series-002",
                "patient_id": patient_id,
                "modality": "MR",
                "series_description": "T2 FLAIR",
                "study_date": "2024-01-15",
                "num_instances": 48,
                "series_number": 5
            }
        ]
        
        return jsonify({
            "success": True,
            "data": {
                "patient_id": patient_id,
                "series": series_list,
                "total_count": len(series_list)
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error listing DICOM series: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@visualization_bp.route('/dicom/series/<series_id>/metadata', methods=['GET'])
@require_auth
def get_dicom_series_metadata(series_id: str):
    """
    Get detailed metadata for a DICOM series.
    
    Returns:
        DICOM series metadata including tags and acquisition parameters
    """
    try:
        # Mock metadata - in production, this would query DICOM tags
        metadata = {
            "series_id": series_id,
            "study_instance_uid": "1.2.840.113619.2.408.5769116.6102871.29345.1705334400",
            "series_instance_uid": f"1.2.840.113619.2.408.5769116.6102871.29345.{series_id}",
            "modality": "MR",
            "series_description": "T1 MPRAGE",
            "series_number": 3,
            "manufacturer": "SIEMENS",
            "manufacturer_model_name": "Prisma",
            "magnetic_field_strength": 3.0,
            "repetition_time": 2300.0,
            "echo_time": 2.98,
            "flip_angle": 9.0,
            "slice_thickness": 1.0,
            "pixel_spacing": [1.0, 1.0],
            "image_orientation_patient": [1, 0, 0, 0, 1, 0],
            "rows": 256,
            "columns": 256,
            "num_slices": 176,
            "acquisition_date": "2024-01-15",
            "acquisition_time": "14:30:25"
        }
        
        return jsonify({
            "success": True,
            "data": metadata
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting DICOM metadata: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@visualization_bp.route('/dicom/series/<series_id>/thumbnail', methods=['GET'])
@require_auth
def get_dicom_thumbnail(series_id: str):
    """
    Get thumbnail image for a DICOM series.
    
    Returns:
        PNG thumbnail image
    """
    try:
        # Mock thumbnail - in production, this would render a middle slice
        # For now, return a simple response indicating thumbnail endpoint
        return jsonify({
            "success": True,
            "message": "Thumbnail endpoint - in production, returns PNG image",
            "series_id": series_id,
            "note": "Implement actual DICOM rendering with pydicom/PIL"
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting DICOM thumbnail: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@visualization_bp.route('/dicom/series/<series_id>/slice/<int:slice_number>', methods=['GET'])
@require_auth
def get_dicom_slice(series_id: str, slice_number: int):
    """
    Get a specific slice from a DICOM series.
    
    Query params:
        window_center: Window center for display (optional)
        window_width: Window width for display (optional)
    
    Returns:
        PNG image of the slice
    """
    try:
        window_center = request.args.get('window_center', type=int)
        window_width = request.args.get('window_width', type=int)
        
        # Mock response - in production, render actual DICOM slice
        return jsonify({
            "success": True,
            "message": "DICOM slice endpoint - in production, returns PNG image",
            "series_id": series_id,
            "slice_number": slice_number,
            "window_center": window_center,
            "window_width": window_width,
            "note": "Implement actual DICOM rendering with pydicom/PIL"
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting DICOM slice: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@visualization_bp.route('/dicom/series/<series_id>/stream', methods=['GET'])
@require_auth
def stream_dicom_series(series_id: str):
    """
    Stream DICOM series for progressive loading.
    
    Returns:
        Server-sent events stream with slice data
    """
    def generate():
        """Generator for streaming DICOM slices."""
        # Mock streaming - in production, yield actual slice data
        for i in range(10):  # Mock 10 slices
            data = {
                "slice_number": i,
                "series_id": series_id,
                "message": f"Slice {i} data would be here"
            }
            yield f"data: {json.dumps(data)}\n\n"
    
    try:
        return Response(generate(), mimetype='text/event-stream')
        
    except Exception as e:
        logger.error(f"Error streaming DICOM series: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@visualization_bp.route('/dicom/series/<series_id>/explainability', methods=['POST'])
@require_auth
def get_dicom_explainability():
    """
    Get explainability overlay for DICOM series.
    
    Request body:
    {
        "model_prediction": "lesion_detected",
        "slice_number": 45
    }
    
    Returns:
        Explainability overlay data (heatmap, regions of interest)
    """
    try:
        data = request.get_json()
        slice_number = data.get('slice_number')
        model_prediction = data.get('model_prediction')
        
        # Mock explainability data
        overlay_data = {
            "series_id": series_id,
            "slice_number": slice_number,
            "model_prediction": model_prediction,
            "heatmap": {
                "regions": [
                    {"x": 120, "y": 150, "width": 30, "height": 25, "intensity": 0.85},
                    {"x": 80, "y": 100, "width": 20, "height": 15, "intensity": 0.65}
                ]
            },
            "attention_scores": [0.85, 0.65, 0.45, 0.32],
            "confidence": 0.92
        }
        
        return jsonify({
            "success": True,
            "data": overlay_data
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting DICOM explainability: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


# ==================== Health Check ====================

@visualization_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for visualization services."""
    return jsonify({
        "status": "healthy",
        "service": "visualization",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "brain_engine": "operational",
            "dicom_viewer": "operational"
        }
    }), 200
