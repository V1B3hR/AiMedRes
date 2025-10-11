"""
Test suite for Rare Disease Research Extension (P19)

Tests orphan disease detection, federated learning, patient advocacy,
and precision medicine analytics capabilities.
"""

import pytest
import time
from datetime import datetime, timedelta
import sys
import os
import numpy as np

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from aimedres.clinical.rare_disease_research import (
    RareDiseaseResearchSystem,
    OrphanDiseaseDetectionEngine,
    FederatedLearningCollaborator,
    PatientAdvocacyProgram,
    PrecisionMedicineAnalytics,
    OrphanDiseaseCategory,
    LearningMethod,
    FederatedRole,
    AdvocacyPartnerType,
    VariantPathogenicity,
    OrphanDiseaseProfile,
    FewShotExample,
    DetectionModel,
    FederatedNode,
    FederatedRound,
    AdvocacyPartner,
    PatientRegistry,
    GeneticVariant,
    PhenotypeProfile,
    create_rare_disease_research_system
)


class TestOrphanDiseaseDetectionEngine:
    """Tests for orphan disease detection."""
    
    def test_engine_initialization(self):
        """Test detection engine initialization."""
        engine = OrphanDiseaseDetectionEngine()
        assert len(engine.disease_profiles) == 0
        assert len(engine.detection_models) == 0
        assert len(engine.detection_history) == 0
    
    def test_register_disease(self):
        """Test registering orphan disease."""
        engine = OrphanDiseaseDetectionEngine()
        
        profile = OrphanDiseaseProfile(
            disease_id="disease_001",
            disease_name="Rare Syndrome X",
            category=OrphanDiseaseCategory.GENETIC,
            prevalence=1.5,  # per 100,000
            known_cases=500,
            genetic_basis="GENE_X mutation",
            phenotype_features=["symptom_a", "symptom_b"],
            diagnostic_criteria=["criterion_1", "criterion_2"],
            available_treatments=["treatment_a"],
            research_priority=5
        )
        
        engine.register_disease(profile)
        
        assert "disease_001" in engine.disease_profiles
        assert engine.disease_profiles["disease_001"].disease_name == "Rare Syndrome X"
    
    def test_add_few_shot_example(self):
        """Test adding few-shot learning examples."""
        engine = OrphanDiseaseDetectionEngine()
        
        # Register disease first
        profile = OrphanDiseaseProfile(
            disease_id="disease_002",
            disease_name="Rare Disease Y",
            category=OrphanDiseaseCategory.METABOLIC,
            prevalence=0.8,
            known_cases=200,
            genetic_basis=None,
            phenotype_features=["feature_x"],
            diagnostic_criteria=["test_positive"],
            available_treatments=[],
            research_priority=4
        )
        engine.register_disease(profile)
        
        # Add few-shot example
        example = FewShotExample(
            example_id="example_001",
            disease_id="disease_002",
            features={"biomarker_1": 0.8, "biomarker_2": 0.6},
            phenotype={"symptom": "severe"},
            genotype={"variant": "positive"},
            diagnosis_confirmed=True,
            confidence=0.95
        )
        
        engine.add_few_shot_example(example)
        
        assert len(engine.few_shot_examples["disease_002"]) == 1
    
    def test_train_detection_model(self):
        """Test training detection model."""
        engine = OrphanDiseaseDetectionEngine()
        
        # Register disease
        profile = OrphanDiseaseProfile(
            disease_id="disease_003",
            disease_name="Orphan Disease Z",
            category=OrphanDiseaseCategory.NEURODEGENERATIVE,
            prevalence=2.0,
            known_cases=1000,
            genetic_basis="multiple genes",
            phenotype_features=["cognitive_decline"],
            diagnostic_criteria=["mri_pattern"],
            available_treatments=["experimental_drug"],
            research_priority=5
        )
        engine.register_disease(profile)
        
        # Train model
        model = engine.train_detection_model(
            disease_id="disease_003",
            method=LearningMethod.FEW_SHOT,
            training_samples=50
        )
        
        assert model.disease_id == "disease_003"
        assert model.learning_method == LearningMethod.FEW_SHOT
        assert model.training_samples == 50
        assert 0.80 <= model.validation_accuracy <= 0.95
        assert model.model_id in engine.detection_models
    
    def test_detect_disease(self):
        """Test disease detection."""
        engine = OrphanDiseaseDetectionEngine()
        
        # Setup diseases and models
        for i in range(3):
            profile = OrphanDiseaseProfile(
                disease_id=f"disease_detect_{i}",
                disease_name=f"Disease {i}",
                category=OrphanDiseaseCategory.GENETIC,
                prevalence=1.0,
                known_cases=100,
                genetic_basis=None,
                phenotype_features=[],
                diagnostic_criteria=[],
                available_treatments=[],
                research_priority=3
            )
            engine.register_disease(profile)
            engine.train_detection_model(f"disease_detect_{i}", LearningMethod.FEW_SHOT, 20)
        
        # Detect
        patient_features = {"feature_1": 0.7, "feature_2": 0.8}
        result = engine.detect_disease(patient_features)
        
        assert 'detection_id' in result
        assert 'predictions' in result
        assert len(result['predictions']) <= 5  # Top 5
        assert 'detection_time_ms' in result
        assert result['features_analyzed'] == 2
    
    def test_transfer_learning(self):
        """Test transfer learning between diseases."""
        engine = OrphanDiseaseDetectionEngine()
        
        # Register source and target diseases
        for suffix in ['source', 'target']:
            profile = OrphanDiseaseProfile(
                disease_id=f"disease_{suffix}",
                disease_name=f"Disease {suffix}",
                category=OrphanDiseaseCategory.GENETIC,
                prevalence=1.0,
                known_cases=100,
                genetic_basis=None,
                phenotype_features=[],
                diagnostic_criteria=[],
                available_treatments=[],
                research_priority=3
            )
            engine.register_disease(profile)
        
        # Train source model
        source_model = engine.train_detection_model("disease_source", LearningMethod.FEW_SHOT, 100)
        
        # Transfer to target
        transferred_model = engine.transfer_model("disease_source", "disease_target")
        
        assert transferred_model.disease_id == "disease_target"
        assert transferred_model.learning_method == LearningMethod.TRANSFER_LEARNING
        assert transferred_model.metadata['source_disease_id'] == "disease_source"
        assert transferred_model.validation_accuracy < source_model.validation_accuracy


class TestFederatedLearningCollaborator:
    """Tests for federated learning."""
    
    def test_collaborator_initialization(self):
        """Test federated collaborator initialization."""
        collaborator = FederatedLearningCollaborator()
        assert len(collaborator.nodes) == 0
        assert len(collaborator.rounds) == 0
    
    def test_register_node(self):
        """Test registering federated node."""
        collaborator = FederatedLearningCollaborator()
        
        node = FederatedNode(
            node_id="node_001",
            institution_name="Hospital A",
            role=FederatedRole.CONTRIBUTOR,
            data_samples=500,
            computational_capacity="high",
            privacy_level="strict",
            last_sync=datetime.now(),
            active=True
        )
        
        collaborator.register_node(node)
        
        assert "node_001" in collaborator.nodes
        assert collaborator.collaboration_stats['total_data_samples'] == 500
    
    def test_start_federated_round(self):
        """Test starting federated learning round."""
        collaborator = FederatedLearningCollaborator()
        
        # Register multiple nodes
        for i in range(3):
            node = FederatedNode(
                node_id=f"node_{i}",
                institution_name=f"Hospital {i}",
                role=FederatedRole.CONTRIBUTOR,
                data_samples=200 + i * 50,
                computational_capacity="medium",
                privacy_level="standard",
                last_sync=datetime.now(),
                active=True
            )
            collaborator.register_node(node)
        
        # Start round
        federated_round = collaborator.start_federated_round(
            participating_node_ids=["node_0", "node_1", "node_2"],
            model_version="v1.0",
            aggregation_method="fedavg"
        )
        
        assert federated_round.round_number == 1
        assert len(federated_round.participating_nodes) == 3
        assert 0.70 <= federated_round.global_accuracy <= 0.95
        assert federated_round.convergence_metric > 0
    
    def test_submit_model_update(self):
        """Test submitting model update from node."""
        collaborator = FederatedLearningCollaborator()
        
        # Register node
        node = FederatedNode(
            node_id="node_update",
            institution_name="Test Hospital",
            role=FederatedRole.CONTRIBUTOR,
            data_samples=300,
            computational_capacity="high",
            privacy_level="strict",
            last_sync=datetime.now(),
            active=True
        )
        collaborator.register_node(node)
        
        # Start round
        round_obj = collaborator.start_federated_round(
            ["node_update", "node_update"],  # Same node twice for test
            "v1.0"
        )
        
        # Submit update
        update = collaborator.submit_model_update(
            node_id="node_update",
            round_id=round_obj.round_id,
            local_accuracy=0.88,
            samples_used=250
        )
        
        assert 'update_id' in update
        assert update['node_id'] == "node_update"
        assert update['local_accuracy'] == 0.88
    
    def test_collaboration_summary(self):
        """Test getting collaboration summary."""
        collaborator = FederatedLearningCollaborator()
        
        # Register nodes and run rounds
        for i in range(2):
            node = FederatedNode(
                node_id=f"node_summary_{i}",
                institution_name=f"Hospital {i}",
                role=FederatedRole.CONTRIBUTOR,
                data_samples=100,
                computational_capacity="medium",
                privacy_level="standard",
                last_sync=datetime.now(),
                active=True
            )
            collaborator.register_node(node)
        
        collaborator.start_federated_round(
            [f"node_summary_{i}" for i in range(2)],
            "v1.0"
        )
        
        summary = collaborator.get_collaboration_summary()
        
        assert summary['total_nodes'] == 2
        assert summary['active_nodes'] == 2
        assert summary['completed_rounds'] == 1
        assert 'latest_global_accuracy' in summary


class TestPatientAdvocacyProgram:
    """Tests for patient advocacy program."""
    
    def test_program_initialization(self):
        """Test advocacy program initialization."""
        program = PatientAdvocacyProgram()
        assert len(program.partners) == 0
        assert len(program.engagement_events) == 0
    
    def test_register_partner(self):
        """Test registering advocacy partner."""
        program = PatientAdvocacyProgram()
        
        partner = AdvocacyPartner(
            partner_id="partner_001",
            organization_name="Rare Disease Foundation",
            partner_type=AdvocacyPartnerType.RESEARCH_FOUNDATION,
            focus_diseases=["disease_a", "disease_b"],
            patient_reach=5000,
            collaboration_level="strategic",
            contact_info={"email": "contact@foundation.org"},
            partnership_date=datetime.now(),
            active=True
        )
        
        program.register_partner(partner)
        
        assert "partner_001" in program.partners
        assert program.partners["partner_001"].patient_reach == 5000
    
    def test_enroll_patient(self):
        """Test enrolling patient in registry."""
        program = PatientAdvocacyProgram()
        
        registry = PatientRegistry(
            registry_id="reg_001",
            patient_id="patient_anon_001",
            disease_id="disease_rare",
            enrollment_date=datetime.now(),
            phenotype_data={"symptoms": ["symptom_x"]},
            genotype_data={"variant": "positive"},
            clinical_outcomes=[],
            consent_level="full"
        )
        
        program.enroll_patient(registry)
        
        assert len(program.patient_registries["disease_rare"]) == 1
    
    def test_create_engagement_event(self):
        """Test creating patient engagement event."""
        program = PatientAdvocacyProgram()
        
        # Register partner first
        partner = AdvocacyPartner(
            partner_id="partner_event",
            organization_name="Patient Group",
            partner_type=AdvocacyPartnerType.SUPPORT_GROUP,
            focus_diseases=["disease_x"],
            patient_reach=1000,
            collaboration_level="active",
            contact_info={},
            partnership_date=datetime.now(),
            active=True
        )
        program.register_partner(partner)
        
        # Create event
        event = program.create_engagement_event(
            partner_id="partner_event",
            event_type="education_workshop",
            disease_id="disease_x",
            participants=50
        )
        
        assert 'event_id' in event
        assert event['participants'] == 50
        assert event['event_type'] == "education_workshop"
    
    def test_track_outcome(self):
        """Test tracking patient outcomes."""
        program = PatientAdvocacyProgram()
        
        outcome = program.track_outcome(
            disease_id="disease_outcome",
            outcome_type="quality_of_life",
            improvement_metric=1.5,
            patients_affected=100
        )
        
        assert 'outcome_id' in outcome
        assert outcome['improvement_metric'] == 1.5
        assert len(program.outcomes) == 1
    
    def test_program_impact(self):
        """Test getting program impact metrics."""
        program = PatientAdvocacyProgram()
        
        # Add some data
        partner = AdvocacyPartner(
            partner_id="partner_impact",
            organization_name="Impact Org",
            partner_type=AdvocacyPartnerType.PATIENT_ORGANIZATION,
            focus_diseases=["disease_impact"],
            patient_reach=2000,
            collaboration_level="strategic",
            contact_info={},
            partnership_date=datetime.now(),
            active=True
        )
        program.register_partner(partner)
        
        impact = program.get_program_impact()
        
        assert impact['active_partners'] == 1
        assert impact['patient_reach'] == 2000


class TestPrecisionMedicineAnalytics:
    """Tests for precision medicine analytics."""
    
    def test_analytics_initialization(self):
        """Test analytics initialization."""
        analytics = PrecisionMedicineAnalytics()
        assert len(analytics.genetic_variants) == 0
        assert len(analytics.phenotype_profiles) == 0
    
    def test_add_genetic_variant(self):
        """Test adding genetic variant."""
        analytics = PrecisionMedicineAnalytics()
        
        variant = GeneticVariant(
            variant_id="var_001",
            gene_symbol="BRCA1",
            chromosome="17",
            position=41234567,
            reference_allele="G",
            alternate_allele="A",
            pathogenicity=VariantPathogenicity.PATHOGENIC,
            disease_associations=["breast_cancer", "ovarian_cancer"],
            population_frequency=0.001,
            functional_impact="high"
        )
        
        analytics.add_genetic_variant(variant)
        
        assert "var_001" in analytics.genetic_variants
        assert analytics.genetic_variants["var_001"].gene_symbol == "BRCA1"
    
    def test_add_phenotype_profile(self):
        """Test adding phenotype profile."""
        analytics = PrecisionMedicineAnalytics()
        
        profile = PhenotypeProfile(
            profile_id="pheno_001",
            patient_id="patient_001",
            hpo_terms=["HP:0001250", "HP:0002376"],
            clinical_features={"seizures": "present", "ataxia": "present"},
            severity_scores={"overall": 0.7},
            age_of_onset=10,
            progression_rate="slow"
        )
        
        analytics.add_phenotype_profile(profile)
        
        assert "pheno_001" in analytics.phenotype_profiles
    
    def test_correlate_variant_phenotype(self):
        """Test correlating variant with phenotype."""
        analytics = PrecisionMedicineAnalytics()
        
        # Add variant
        variant = GeneticVariant(
            variant_id="var_corr",
            gene_symbol="SCN1A",
            chromosome="2",
            position=166848600,
            reference_allele="C",
            alternate_allele="T",
            pathogenicity=VariantPathogenicity.PATHOGENIC,
            disease_associations=["epilepsy"],
            population_frequency=0.0001,
            functional_impact="severe"
        )
        analytics.add_genetic_variant(variant)
        
        # Add phenotype
        profile = PhenotypeProfile(
            profile_id="pheno_corr",
            patient_id="patient_corr",
            hpo_terms=["HP:0001250"],
            clinical_features={"seizures": "severe"},
            severity_scores={"seizure": 0.9},
            age_of_onset=5,
            progression_rate="rapid"
        )
        analytics.add_phenotype_profile(profile)
        
        # Correlate
        correlation = analytics.correlate_variant_phenotype(
            variant_id="var_corr",
            phenotype_id="pheno_corr",
            correlation_strength=0.92
        )
        
        assert 'correlation_id' in correlation
        assert correlation['gene_symbol'] == "SCN1A"
        assert correlation['correlation_strength'] == 0.92
    
    def test_analyze_patient(self):
        """Test patient analysis."""
        analytics = PrecisionMedicineAnalytics()
        
        # Add pathogenic variant
        variant = GeneticVariant(
            variant_id="var_analysis",
            gene_symbol="CFTR",
            chromosome="7",
            position=117199646,
            reference_allele="C",
            alternate_allele="T",
            pathogenicity=VariantPathogenicity.PATHOGENIC,
            disease_associations=["cystic_fibrosis"],
            population_frequency=0.02,
            functional_impact="severe"
        )
        analytics.add_genetic_variant(variant)
        
        # Add phenotype
        profile = PhenotypeProfile(
            profile_id="pheno_analysis",
            patient_id="patient_analysis",
            hpo_terms=["HP:0006528", "HP:0002837"],
            clinical_features={},
            severity_scores={},
            age_of_onset=None,
            progression_rate=None
        )
        analytics.add_phenotype_profile(profile)
        
        # Analyze
        result = analytics.analyze_patient(
            patient_variants=["var_analysis"],
            patient_phenotypes=["pheno_analysis"]
        )
        
        assert 'analysis_id' in result
        assert result['pathogenic_variants_found'] >= 1
        assert 'integrated_risk_score' in result
        assert result['risk_level'] in ['low', 'moderate', 'high']
    
    def test_recommend_treatment(self):
        """Test treatment recommendation."""
        analytics = PrecisionMedicineAnalytics()
        
        # Create patient analysis with high risk
        patient_analysis = {
            'analysis_id': 'analysis_001',
            'integrated_risk_score': 0.8,
            'pathogenic_variants_found': 2
        }
        
        treatment = analytics.recommend_treatment(patient_analysis)
        
        assert 'recommendation_id' in treatment
        assert 'recommendations' in treatment
        assert len(treatment['recommendations']) > 0
        assert treatment['personalized'] is True


class TestRareDiseaseResearchSystem:
    """Tests for integrated system."""
    
    def test_system_initialization(self):
        """Test system initialization."""
        system = create_rare_disease_research_system()
        
        assert system.detection is not None
        assert system.federated is not None
        assert system.advocacy is not None
        assert system.precision is not None
    
    def test_system_statistics(self):
        """Test comprehensive system statistics."""
        system = create_rare_disease_research_system()
        
        stats = system.get_system_statistics()
        
        assert 'detection' in stats
        assert 'federated_learning' in stats
        assert 'advocacy' in stats
        assert 'precision_medicine' in stats
        
        # Verify statistics structure
        assert 'registered_diseases' in stats['detection']
        assert 'total_nodes' in stats['federated_learning']
        assert 'total_partners' in stats['advocacy']
        assert 'total_variants' in stats['precision_medicine']


class TestIntegrationScenarios:
    """Integration tests for complete workflows."""
    
    def test_rare_disease_research_workflow(self):
        """Test complete rare disease research workflow."""
        system = create_rare_disease_research_system()
        
        # 1. Register disease
        profile = OrphanDiseaseProfile(
            disease_id="workflow_disease",
            disease_name="Test Syndrome",
            category=OrphanDiseaseCategory.GENETIC,
            prevalence=1.0,
            known_cases=100,
            genetic_basis="GENE_TEST",
            phenotype_features=["feature_1"],
            diagnostic_criteria=["criteria_1"],
            available_treatments=[],
            research_priority=5
        )
        system.detection.register_disease(profile)
        
        # 2. Train detection model
        model = system.detection.train_detection_model(
            "workflow_disease",
            LearningMethod.FEW_SHOT,
            30
        )
        
        # 3. Setup federated learning
        node = FederatedNode(
            node_id="workflow_node",
            institution_name="Research Center",
            role=FederatedRole.CONTRIBUTOR,
            data_samples=200,
            computational_capacity="high",
            privacy_level="strict",
            last_sync=datetime.now(),
            active=True
        )
        system.federated.register_node(node)
        
        # 4. Add advocacy partner
        partner = AdvocacyPartner(
            partner_id="workflow_partner",
            organization_name="Patient Alliance",
            partner_type=AdvocacyPartnerType.PATIENT_ORGANIZATION,
            focus_diseases=["workflow_disease"],
            patient_reach=500,
            collaboration_level="active",
            contact_info={},
            partnership_date=datetime.now(),
            active=True
        )
        system.advocacy.register_partner(partner)
        
        # Verify all components working
        assert model.validation_accuracy > 0
        assert "workflow_node" in system.federated.nodes
        assert "workflow_partner" in system.advocacy.partners


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
