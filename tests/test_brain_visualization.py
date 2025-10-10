"""
Tests for 3D Brain Visualization Platform (P15)
"""

import pytest
import sys
import os
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from aimedres.dashboards.brain_visualization import (
    create_brain_visualization_engine,
    BrainVisualizationEngine,
    BrainRegion,
    DiseaseStage,
    VisualizationMode,
    TreatmentType
)


class TestBrainVisualizationEngine:
    """Test suite for Brain Visualization Engine."""
    
    @pytest.fixture
    def engine(self):
        """Create a test brain visualization engine."""
        return create_brain_visualization_engine(enable_real_time=True, cache_size=100)
    
    def test_engine_initialization(self, engine):
        """Test engine initialization."""
        assert engine is not None
        assert engine.enable_real_time is True
        assert engine.cache_size == 100
        assert len(engine.brain_atlas) > 0
    
    def test_create_anatomical_overlay(self, engine):
        """Test creating anatomical overlay."""
        regions = [BrainRegion.HIPPOCAMPUS, BrainRegion.FRONTAL_LOBE]
        overlay = engine.create_anatomical_overlay(
            patient_id="patient_001",
            regions_of_interest=regions,
            highlight_abnormalities=True
        )
        
        assert overlay is not None
        assert 'overlay_id' in overlay
        assert 'markers' in overlay
        assert len(overlay['markers']) == 2
        assert overlay['render_time_ms'] >= 0
    
    def test_map_disease_pathology(self, engine):
        """Test disease pathology mapping."""
        severity_map = {
            BrainRegion.HIPPOCAMPUS: 0.8,
            BrainRegion.TEMPORAL_LOBE: 0.6,
            BrainRegion.FRONTAL_LOBE: 0.4
        }
        
        mapping = engine.map_disease_pathology(
            patient_id="patient_001",
            disease_type="alzheimers",
            severity_map=severity_map
        )
        
        assert mapping is not None
        assert 'mapping_id' in mapping
        assert 'overall_burden' in mapping
        assert mapping['disease_type'] == "alzheimers"
        assert len(mapping['most_affected_regions']) > 0
    
    def test_capture_progression_snapshot(self, engine):
        """Test capturing disease progression snapshot."""
        affected_regions = {
            BrainRegion.HIPPOCAMPUS: 0.7,
            BrainRegion.TEMPORAL_LOBE: 0.5
        }
        
        snapshot = engine.capture_progression_snapshot(
            patient_id="patient_001",
            stage=DiseaseStage.MODERATE,
            affected_regions=affected_regions,
            biomarkers={'amyloid_beta': 0.8},
            cognitive_scores={'mmse': 22}
        )
        
        assert snapshot is not None
        assert snapshot.stage == DiseaseStage.MODERATE
        assert len(snapshot.volumetric_data) > 0
    
    def test_visualize_temporal_progression(self, engine):
        """Test temporal progression visualization."""
        # Create multiple snapshots
        snapshot_ids = []
        for i in range(3):
            affected_regions = {
                BrainRegion.HIPPOCAMPUS: 0.3 + i * 0.2
            }
            snapshot = engine.capture_progression_snapshot(
                patient_id="patient_001",
                stage=DiseaseStage.MILD if i < 2 else DiseaseStage.MODERATE,
                affected_regions=affected_regions
            )
            snapshot_ids.append(snapshot.snapshot_id)
        
        visualization = engine.visualize_temporal_progression(
            patient_id="patient_001",
            snapshots=snapshot_ids,
            time_scale="months"
        )
        
        assert visualization is not None
        assert 'visualization_id' in visualization
        assert visualization['num_snapshots'] == 3
        assert 'severity_timeline' in visualization
    
    def test_simulate_treatment_impact(self, engine):
        """Test treatment impact simulation."""
        # Create baseline snapshot
        affected_regions = {BrainRegion.HIPPOCAMPUS: 0.7}
        baseline = engine.capture_progression_snapshot(
            patient_id="patient_001",
            stage=DiseaseStage.MODERATE,
            affected_regions=affected_regions
        )
        
        simulation = engine.simulate_treatment_impact(
            patient_id="patient_001",
            baseline_snapshot_id=baseline.snapshot_id,
            treatment_type=TreatmentType.MEDICATION,
            duration_days=180,
            efficacy_rate=0.7
        )
        
        assert simulation is not None
        assert simulation.treatment_type == TreatmentType.MEDICATION
        assert len(simulation.projected_outcomes) > 0
        assert simulation.success_probability == 0.7
    
    def test_compare_treatment_scenarios(self, engine):
        """Test comparing multiple treatment scenarios."""
        # Create baseline
        affected_regions = {BrainRegion.HIPPOCAMPUS: 0.7}
        baseline = engine.capture_progression_snapshot(
            patient_id="patient_001",
            stage=DiseaseStage.MODERATE,
            affected_regions=affected_regions
        )
        
        # Create multiple simulations
        sim_ids = []
        for treatment_type in [TreatmentType.MEDICATION, TreatmentType.COGNITIVE_THERAPY]:
            sim = engine.simulate_treatment_impact(
                patient_id="patient_001",
                baseline_snapshot_id=baseline.snapshot_id,
                treatment_type=treatment_type,
                duration_days=180,
                efficacy_rate=0.7
            )
            sim_ids.append(sim.simulation_id)
        
        comparison = engine.compare_treatment_scenarios(
            patient_id="patient_001",
            simulation_ids=sim_ids
        )
        
        assert comparison is not None
        assert comparison['num_scenarios'] == 2
        assert 'recommended' in comparison
    
    def test_create_educational_module(self, engine):
        """Test creating educational module."""
        module = engine.create_educational_module(
            title="Introduction to Brain Anatomy",
            description="Basic brain anatomy for medical students",
            difficulty_level="beginner",
            target_audience="student",
            brain_regions=[BrainRegion.FRONTAL_LOBE, BrainRegion.TEMPORAL_LOBE],
            learning_objectives=["Identify major brain regions", "Understand basic functions"],
            completion_time_minutes=30
        )
        
        assert module is not None
        assert module.title == "Introduction to Brain Anatomy"
        assert len(module.assessment_questions) > 0
    
    def test_complete_module(self, engine):
        """Test completing educational module."""
        module = engine.create_educational_module(
            title="Test Module",
            description="Test",
            difficulty_level="beginner",
            target_audience="student",
            brain_regions=[BrainRegion.FRONTAL_LOBE],
            learning_objectives=["Test"],
            completion_time_minutes=10
        )
        
        completion = engine.complete_module(
            module_id=module.module_id,
            user_id="user_001",
            assessment_score=85.0
        )
        
        assert completion is not None
        assert completion['passed'] is True
        assert completion['assessment_score'] == 85.0
    
    def test_get_statistics(self, engine):
        """Test getting engine statistics."""
        # Generate some activity
        engine.create_anatomical_overlay("patient_001", [BrainRegion.HIPPOCAMPUS])
        
        stats = engine.get_statistics()
        
        assert stats is not None
        assert 'visualizations_generated' in stats
        assert stats['visualizations_generated'] >= 1


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
