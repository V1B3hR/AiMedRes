"""
Tests for Multi-Modal AI Integration (P16)
"""

import pytest
import sys
import os
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from aimedres.core.multimodal_integration import (
    create_multimodal_integration_engine,
    MultiModalIntegrationEngine,
    ImagingModality,
    GeneticVariantType,
    BiomarkerType,
    SpeechFeatureType
)


class TestMultiModalIntegrationEngine:
    """Test suite for Multi-Modal Integration Engine."""
    
    @pytest.fixture
    def engine(self):
        """Create a test multi-modal integration engine."""
        return create_multimodal_integration_engine(enable_gpu=False, fusion_strategy='weighted')
    
    def test_engine_initialization(self, engine):
        """Test engine initialization."""
        assert engine is not None
        assert engine.enable_gpu is False
        assert engine.fusion_strategy == 'weighted'
        assert len(engine.modality_weights) > 0
    
    def test_ingest_dicom_image(self, engine):
        """Test DICOM image ingestion."""
        image = engine.ingest_dicom_image(
            patient_id="patient_001",
            modality=ImagingModality.MRI,
            image_shape=(256, 256, 128),
            voxel_spacing=(1.0, 1.0, 1.0)
        )
        
        assert image is not None
        assert image.modality == ImagingModality.MRI
        assert len(image.extracted_features) > 0
        assert engine.images_processed == 1
    
    def test_fuse_imaging_studies(self, engine):
        """Test fusing multiple imaging studies."""
        # Ingest multiple images
        img1 = engine.ingest_dicom_image("patient_001", ImagingModality.MRI, (256, 256, 128), (1.0, 1.0, 1.0))
        img2 = engine.ingest_dicom_image("patient_001", ImagingModality.CT, (512, 512, 100), (0.5, 0.5, 1.0))
        
        fusion = engine.fuse_imaging_studies(
            patient_id="patient_001",
            image_ids=[img1.image_id, img2.image_id]
        )
        
        assert fusion is not None
        assert 'fusion_id' in fusion
        assert fusion['num_images'] == 2
        assert len(fusion['modalities']) == 2
    
    def test_analyze_genetic_variant(self, engine):
        """Test genetic variant analysis."""
        variant = engine.analyze_genetic_variant(
            patient_id="patient_001",
            variant_type=GeneticVariantType.SNP,
            chromosome="19",
            position=45411941,
            reference_allele="C",
            alternate_allele="T",
            gene_name="APOE"
        )
        
        assert variant is not None
        assert variant.gene_name == "APOE"
        assert variant.clinical_significance in ['pathogenic', 'likely_pathogenic', 'uncertain_significance', 'benign', 'likely_benign']
        assert 0.0 <= variant.risk_score <= 1.0
    
    def test_create_genetic_risk_profile(self, engine):
        """Test creating genetic risk profile."""
        # Add some variants
        engine.analyze_genetic_variant("patient_001", GeneticVariantType.SNP, "19", 45411941, "C", "T", "APOE")
        engine.analyze_genetic_variant("patient_001", GeneticVariantType.SNP, "21", 25891796, "A", "G", "APP")
        
        profile = engine.create_genetic_risk_profile("patient_001")
        
        assert profile is not None
        assert 'profile_id' in profile
        assert profile['total_variants'] == 2
        assert 'overall_genetic_risk' in profile
    
    def test_record_biomarker_measurement(self, engine):
        """Test recording biomarker measurement."""
        measurement = engine.record_biomarker_measurement(
            patient_id="patient_001",
            biomarker_type=BiomarkerType.PROTEIN,
            biomarker_name="amyloid_beta_42",
            value=200.0,
            unit="pg/mL",
            reference_range=(300.0, 800.0)
        )
        
        assert measurement is not None
        assert measurement.is_abnormal is True  # Below reference range
        assert engine.biomarkers_measured == 1
    
    def test_analyze_biomarker_patterns(self, engine):
        """Test biomarker pattern analysis."""
        # Add multiple biomarkers
        engine.record_biomarker_measurement("patient_001", BiomarkerType.PROTEIN, "amyloid_beta_42", 200.0, "pg/mL", (300.0, 800.0))
        engine.record_biomarker_measurement("patient_001", BiomarkerType.PROTEIN, "tau", 500.0, "pg/mL", (100.0, 300.0))
        
        analysis = engine.analyze_biomarker_patterns("patient_001")
        
        assert analysis is not None
        assert 'analysis_id' in analysis
        assert analysis['total_biomarkers'] == 2
        assert analysis['abnormal_biomarkers'] >= 1
    
    def test_perform_speech_assessment(self, engine):
        """Test speech cognitive assessment."""
        assessment = engine.perform_speech_assessment(
            patient_id="patient_001",
            recording_duration_seconds=60.0
        )
        
        assert assessment is not None
        assert len(assessment.extracted_features) > 0
        assert len(assessment.cognitive_scores) > 0
        assert engine.speech_assessments_completed == 1
    
    def test_fuse_multimodal_data(self, engine):
        """Test comprehensive multi-modal data fusion."""
        # Add imaging data
        img = engine.ingest_dicom_image("patient_001", ImagingModality.MRI, (256, 256, 128), (1.0, 1.0, 1.0))
        
        # Add genetic data
        engine.analyze_genetic_variant("patient_001", GeneticVariantType.SNP, "19", 45411941, "C", "T", "APOE")
        
        # Add biomarker data
        engine.record_biomarker_measurement("patient_001", BiomarkerType.PROTEIN, "amyloid_beta_42", 200.0, "pg/mL", (300.0, 800.0))
        
        # Add speech data
        engine.perform_speech_assessment("patient_001", 60.0)
        
        # Fuse all data
        fusion = engine.fuse_multimodal_data(
            patient_id="patient_001",
            image_ids=[img.image_id],
            include_genetic=True,
            include_biomarker=True,
            include_speech=True
        )
        
        assert fusion is not None
        assert len(fusion.modalities_included) >= 3
        assert 0.0 <= fusion.integrated_risk_score <= 1.0
        assert 0.0 <= fusion.confidence <= 1.0
    
    def test_get_statistics(self, engine):
        """Test getting engine statistics."""
        # Generate some activity
        engine.ingest_dicom_image("patient_001", ImagingModality.MRI, (256, 256, 128), (1.0, 1.0, 1.0))
        
        stats = engine.get_statistics()
        
        assert stats is not None
        assert 'images_processed' in stats
        assert stats['images_processed'] >= 1


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
