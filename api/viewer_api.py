"""
Advanced Multimodal Viewer API (P3-1)

Provides REST endpoints for:
- DICOM streaming and visualization
- 3D brain visualization with explainability overlays
- Real-time viewer updates
- Medical imaging explainability integration
"""

from flask import Flask, jsonify, request, send_file, Response
from flask_cors import CORS
from typing import Dict, Any, List, Optional, BinaryIO
import logging
import json
import io
import os
from datetime import datetime
from pathlib import Path
import numpy as np

logger = logging.getLogger('aimedres.api.viewer')

try:
    from src.aimedres.dashboards.brain_visualization import (
        BrainVisualizationEngine,
        BrainRegion,
        DiseaseStage,
        VisualizationMode
    )
    BRAIN_VIZ_AVAILABLE = True
except ImportError:
    logger.warning("Brain visualization module not available")
    BRAIN_VIZ_AVAILABLE = False

try:
    from mlops.imaging.converters.dicom_to_nifti import (
        AdvancedDICOMToNIfTIConverter,
        ConverterConfig
    )
    DICOM_CONVERTER_AVAILABLE = True
except ImportError:
    logger.warning("DICOM converter not available")
    DICOM_CONVERTER_AVAILABLE = False


class AdvancedViewerAPI:
    """
    Advanced Multimodal Viewer API for medical imaging.
    
    Features:
    - Smooth streaming viewer for DICOM and NIfTI
    - 3D brain visualization with disease progression
    - Explainability overlays for AI predictions
    - Real-time updates and annotations
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Advanced Viewer API.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.app = Flask(__name__)
        CORS(self.app)
        
        # Initialize engines
        self.brain_engine = None
        if BRAIN_VIZ_AVAILABLE:
            self.brain_engine = BrainVisualizationEngine(
                enable_real_time=True,
                cache_size=1000
            )
        
        self.dicom_converter = None
        if DICOM_CONVERTER_AVAILABLE:
            converter_config = ConverterConfig(
                output_dir=Path(config.get('temp_dir', '/tmp/dicom_temp')),
                compute_quality=True
            )
            self.dicom_converter = AdvancedDICOMToNIfTIConverter(converter_config)
        
        # Cache for active viewers
        self.active_viewers = {}
        self.viewer_sessions = {}
        
        # Setup routes
        self._setup_routes()
        
        logger.info("Advanced Viewer API initialized")
    
    def _setup_routes(self):
        """Setup API routes for viewers."""
        
        @self.app.route('/api/viewer/health')
        def health_check():
            """Health check endpoint."""
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'brain_viz_available': BRAIN_VIZ_AVAILABLE,
                'dicom_converter_available': DICOM_CONVERTER_AVAILABLE,
                'version': '1.0.0'
            })
        
        # ==================== DICOM Viewer APIs ====================
        
        @self.app.route('/api/viewer/dicom/upload', methods=['POST'])
        def upload_dicom():
            """
            Upload DICOM file for visualization.
            
            Returns viewer session with initial metadata.
            """
            try:
                if 'file' not in request.files:
                    return jsonify({'error': 'No file provided'}), 400
                
                file = request.files['file']
                if file.filename == '':
                    return jsonify({'error': 'Empty filename'}), 400
                
                # Save temporarily
                session_id = self._generate_session_id()
                temp_dir = Path(self.config.get('temp_dir', '/tmp/dicom_temp')) / session_id
                temp_dir.mkdir(parents=True, exist_ok=True)
                
                file_path = temp_dir / file.filename
                file.save(str(file_path))
                
                # Extract DICOM metadata
                metadata = self._extract_dicom_metadata(file_path)
                
                # Store session
                self.viewer_sessions[session_id] = {
                    'session_id': session_id,
                    'file_path': str(file_path),
                    'metadata': metadata,
                    'created_at': datetime.now().isoformat(),
                    'type': 'dicom'
                }
                
                return jsonify({
                    'success': True,
                    'session_id': session_id,
                    'metadata': metadata
                }), 201
                
            except Exception as e:
                logger.error(f"DICOM upload failed: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/viewer/dicom/<session_id>/image')
        def get_dicom_image(session_id: str):
            """
            Get DICOM image data for streaming viewer.
            
            Supports slice selection and windowing.
            """
            try:
                if session_id not in self.viewer_sessions:
                    return jsonify({'error': 'Session not found'}), 404
                
                session = self.viewer_sessions[session_id]
                slice_idx = request.args.get('slice', 0, type=int)
                window_center = request.args.get('window_center', type=float)
                window_width = request.args.get('window_width', type=float)
                
                # Generate image data (placeholder for actual DICOM rendering)
                image_data = self._render_dicom_slice(
                    session['file_path'],
                    slice_idx,
                    window_center,
                    window_width
                )
                
                return Response(
                    image_data,
                    mimetype='image/png',
                    headers={'Cache-Control': 'no-cache'}
                )
                
            except Exception as e:
                logger.error(f"DICOM image retrieval failed: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/viewer/dicom/<session_id>/metadata')
        def get_dicom_metadata(session_id: str):
            """Get detailed DICOM metadata."""
            try:
                if session_id not in self.viewer_sessions:
                    return jsonify({'error': 'Session not found'}), 404
                
                session = self.viewer_sessions[session_id]
                return jsonify(session['metadata']), 200
                
            except Exception as e:
                logger.error(f"Metadata retrieval failed: {e}")
                return jsonify({'error': str(e)}), 500
        
        # ==================== 3D Brain Viewer APIs ====================
        
        @self.app.route('/api/viewer/brain/create', methods=['POST'])
        def create_brain_viewer():
            """
            Create 3D brain visualization session.
            
            Body: {
                "patient_id": str,
                "regions": [str],  # List of brain regions
                "disease_type": str (optional),
                "highlight_abnormalities": bool (optional)
            }
            """
            try:
                if not BRAIN_VIZ_AVAILABLE:
                    return jsonify({'error': 'Brain visualization not available'}), 503
                
                data = request.get_json() or {}
                patient_id = data.get('patient_id', 'unknown')
                regions_str = data.get('regions', [])
                disease_type = data.get('disease_type')
                highlight = data.get('highlight_abnormalities', True)
                
                # Convert region strings to enums
                regions = []
                for r in regions_str:
                    try:
                        regions.append(BrainRegion(r.lower()))
                    except ValueError:
                        logger.warning(f"Invalid brain region: {r}")
                
                if not regions:
                    # Default regions
                    regions = [
                        BrainRegion.FRONTAL_LOBE,
                        BrainRegion.HIPPOCAMPUS,
                        BrainRegion.TEMPORAL_LOBE
                    ]
                
                # Create anatomical overlay
                overlay = self.brain_engine.create_anatomical_overlay(
                    patient_id=patient_id,
                    regions_of_interest=regions,
                    highlight_abnormalities=highlight
                )
                
                # Create session
                session_id = overlay['overlay_id']
                self.viewer_sessions[session_id] = {
                    'session_id': session_id,
                    'patient_id': patient_id,
                    'type': 'brain_3d',
                    'overlay_data': overlay,
                    'created_at': datetime.now().isoformat()
                }
                
                return jsonify({
                    'success': True,
                    'session_id': session_id,
                    'visualization': overlay
                }), 201
                
            except Exception as e:
                logger.error(f"Brain viewer creation failed: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/viewer/brain/<session_id>/progression', methods=['POST'])
        def add_disease_progression(session_id: str):
            """
            Add disease progression visualization to brain viewer.
            
            Body: {
                "stage": str,
                "affected_regions": {region: severity},
                "biomarkers": {marker: value} (optional),
                "cognitive_scores": {test: score} (optional)
            }
            """
            try:
                if not BRAIN_VIZ_AVAILABLE:
                    return jsonify({'error': 'Brain visualization not available'}), 503
                
                if session_id not in self.viewer_sessions:
                    return jsonify({'error': 'Session not found'}), 404
                
                session = self.viewer_sessions[session_id]
                if session['type'] != 'brain_3d':
                    return jsonify({'error': 'Invalid session type'}), 400
                
                data = request.get_json() or {}
                stage_str = data.get('stage', 'mild')
                affected_regions_data = data.get('affected_regions', {})
                
                # Convert to proper format
                try:
                    stage = DiseaseStage(stage_str.lower())
                except ValueError:
                    stage = DiseaseStage.MILD
                
                affected_regions = {}
                for region_str, severity in affected_regions_data.items():
                    try:
                        region = BrainRegion(region_str.lower())
                        affected_regions[region] = float(severity)
                    except (ValueError, TypeError):
                        logger.warning(f"Invalid region or severity: {region_str}")
                
                # Capture snapshot
                snapshot = self.brain_engine.capture_progression_snapshot(
                    patient_id=session['patient_id'],
                    stage=stage,
                    affected_regions=affected_regions,
                    biomarkers=data.get('biomarkers'),
                    cognitive_scores=data.get('cognitive_scores')
                )
                
                # Store snapshot reference
                if 'snapshots' not in session:
                    session['snapshots'] = []
                session['snapshots'].append(snapshot.snapshot_id)
                
                return jsonify({
                    'success': True,
                    'snapshot_id': snapshot.snapshot_id,
                    'stage': stage.value,
                    'volumetric_data': snapshot.volumetric_data
                }), 201
                
            except Exception as e:
                logger.error(f"Progression addition failed: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/viewer/brain/<session_id>/explainability', methods=['POST'])
        def add_explainability_overlay(session_id: str):
            """
            Add AI explainability overlay to brain visualization.
            
            Body: {
                "prediction_type": str,
                "region_importance": {region: importance_score},
                "confidence": float,
                "features": [{name, value, contribution}]
            }
            """
            try:
                if session_id not in self.viewer_sessions:
                    return jsonify({'error': 'Session not found'}), 404
                
                session = self.viewer_sessions[session_id]
                data = request.get_json() or {}
                
                # Store explainability data
                explainability = {
                    'prediction_type': data.get('prediction_type', 'risk_assessment'),
                    'region_importance': data.get('region_importance', {}),
                    'confidence': data.get('confidence', 0.0),
                    'features': data.get('features', []),
                    'timestamp': datetime.now().isoformat()
                }
                
                if 'explainability' not in session:
                    session['explainability'] = []
                session['explainability'].append(explainability)
                
                return jsonify({
                    'success': True,
                    'explainability_id': len(session['explainability']) - 1,
                    'data': explainability
                }), 201
                
            except Exception as e:
                logger.error(f"Explainability overlay failed: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/viewer/brain/<session_id>/data')
        def get_brain_viewer_data(session_id: str):
            """Get complete brain viewer data for rendering."""
            try:
                if session_id not in self.viewer_sessions:
                    return jsonify({'error': 'Session not found'}), 404
                
                session = self.viewer_sessions[session_id]
                return jsonify(session), 200
                
            except Exception as e:
                logger.error(f"Viewer data retrieval failed: {e}")
                return jsonify({'error': str(e)}), 500
        
        # ==================== Treatment Simulation APIs ====================
        
        @self.app.route('/api/viewer/brain/simulate-treatment', methods=['POST'])
        def simulate_treatment():
            """
            Simulate treatment impact on brain.
            
            Body: {
                "patient_id": str,
                "baseline_snapshot_id": str,
                "treatment_type": str,
                "duration_days": int,
                "efficacy_rate": float (optional)
            }
            """
            try:
                if not BRAIN_VIZ_AVAILABLE:
                    return jsonify({'error': 'Brain visualization not available'}), 503
                
                data = request.get_json() or {}
                patient_id = data.get('patient_id')
                baseline_snapshot_id = data.get('baseline_snapshot_id')
                treatment_type_str = data.get('treatment_type', 'medication')
                duration_days = data.get('duration_days', 90)
                efficacy_rate = data.get('efficacy_rate', 0.7)
                
                if not patient_id or not baseline_snapshot_id:
                    return jsonify({'error': 'patient_id and baseline_snapshot_id required'}), 400
                
                # Import treatment type enum
                from src.aimedres.dashboards.brain_visualization import TreatmentType
                
                try:
                    treatment_type = TreatmentType(treatment_type_str.lower())
                except ValueError:
                    treatment_type = TreatmentType.MEDICATION
                
                # Run simulation
                simulation = self.brain_engine.simulate_treatment_impact(
                    patient_id=patient_id,
                    baseline_snapshot_id=baseline_snapshot_id,
                    treatment_type=treatment_type,
                    duration_days=duration_days,
                    efficacy_rate=efficacy_rate
                )
                
                return jsonify({
                    'success': True,
                    'simulation_id': simulation.simulation_id,
                    'treatment_type': simulation.treatment_type.value,
                    'projected_outcomes': simulation.projected_outcomes,
                    'confidence_interval': simulation.confidence_interval,
                    'success_probability': simulation.success_probability,
                    'side_effects': simulation.side_effects
                }), 201
                
            except Exception as e:
                logger.error(f"Treatment simulation failed: {e}")
                return jsonify({'error': str(e)}), 500
        
        # ==================== Utility APIs ====================
        
        @self.app.route('/api/viewer/sessions')
        def list_sessions():
            """List active viewer sessions."""
            try:
                sessions = []
                for session_id, session in self.viewer_sessions.items():
                    sessions.append({
                        'session_id': session_id,
                        'type': session['type'],
                        'created_at': session['created_at'],
                        'patient_id': session.get('patient_id', 'unknown')
                    })
                
                return jsonify({
                    'sessions': sessions,
                    'total': len(sessions)
                }), 200
                
            except Exception as e:
                logger.error(f"Session listing failed: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/viewer/session/<session_id>', methods=['DELETE'])
        def close_session(session_id: str):
            """Close viewer session and cleanup resources."""
            try:
                if session_id in self.viewer_sessions:
                    session = self.viewer_sessions.pop(session_id)
                    
                    # Cleanup temp files if DICOM
                    if session['type'] == 'dicom':
                        file_path = Path(session.get('file_path', ''))
                        if file_path.exists():
                            file_path.unlink()
                    
                    return jsonify({'success': True}), 200
                else:
                    return jsonify({'error': 'Session not found'}), 404
                    
            except Exception as e:
                logger.error(f"Session close failed: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/viewer/statistics')
        def get_statistics():
            """Get viewer API statistics."""
            try:
                stats = {
                    'active_sessions': len(self.viewer_sessions),
                    'session_types': {},
                    'timestamp': datetime.now().isoformat()
                }
                
                for session in self.viewer_sessions.values():
                    session_type = session['type']
                    stats['session_types'][session_type] = \
                        stats['session_types'].get(session_type, 0) + 1
                
                if BRAIN_VIZ_AVAILABLE and self.brain_engine:
                    stats['brain_engine'] = self.brain_engine.get_statistics()
                
                return jsonify(stats), 200
                
            except Exception as e:
                logger.error(f"Statistics retrieval failed: {e}")
                return jsonify({'error': str(e)}), 500
    
    # ==================== Helper Methods ====================
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID."""
        import uuid
        return str(uuid.uuid4())
    
    def _extract_dicom_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract metadata from DICOM file."""
        try:
            import pydicom
            ds = pydicom.dcmread(str(file_path))
            
            metadata = {
                'patient_id': getattr(ds, 'PatientID', 'unknown'),
                'study_date': getattr(ds, 'StudyDate', ''),
                'modality': getattr(ds, 'Modality', 'unknown'),
                'series_description': getattr(ds, 'SeriesDescription', ''),
                'rows': getattr(ds, 'Rows', 0),
                'columns': getattr(ds, 'Columns', 0),
                'bits_allocated': getattr(ds, 'BitsAllocated', 0)
            }
            
            return metadata
            
        except Exception as e:
            logger.warning(f"DICOM metadata extraction failed: {e}")
            return {'error': 'Could not extract metadata'}
    
    def _render_dicom_slice(
        self,
        file_path: str,
        slice_idx: int,
        window_center: Optional[float],
        window_width: Optional[float]
    ) -> bytes:
        """
        Render DICOM slice to PNG.
        
        This is a placeholder - real implementation would use proper
        DICOM rendering library.
        """
        try:
            # Placeholder: return a simple PNG
            from PIL import Image
            import io
            
            # Create dummy image for now
            img = Image.new('L', (512, 512), color=128)
            
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            buffer.seek(0)
            
            return buffer.read()
            
        except Exception as e:
            logger.error(f"DICOM rendering failed: {e}")
            # Return minimal valid PNG
            return b'\x89PNG\r\n\x1a\n'
    
    def run(self, host: str = '0.0.0.0', port: int = 5002, debug: bool = False):
        """Run the viewer API server."""
        self.app.run(host=host, port=port, debug=debug)


def create_viewer_api(config: Dict[str, Any]) -> AdvancedViewerAPI:
    """Factory function to create viewer API."""
    return AdvancedViewerAPI(config)


if __name__ == '__main__':
    # Run standalone for testing
    config = {
        'temp_dir': '/tmp/aimedres_viewer'
    }
    api = create_viewer_api(config)
    api.run(debug=True)
