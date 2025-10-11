"""
Test suite for International Healthcare Systems (P18)

Tests multilingual interface, regional guidelines, constrained deployment,
and global data governance capabilities.
"""

import pytest
import time
from datetime import datetime, timedelta
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from aimedres.clinical.international_systems import (
    InternationalHealthcareSystem,
    MultilingualInterface,
    RegionalGuidelineEngine,
    ConstrainedDeploymentManager,
    GlobalDataGovernanceFramework,
    Language,
    Region,
    TerminologyStandard,
    DeploymentMode,
    DataGovernanceLevel,
    TerminologyMapping,
    ClinicalGuideline,
    TranslationEntry,
    DeploymentConfiguration,
    DataCollaborationPolicy,
    create_international_healthcare_system
)


class TestMultilingualInterface:
    """Tests for multilingual interface capabilities."""
    
    def test_interface_initialization(self):
        """Test multilingual interface initialization."""
        interface = MultilingualInterface(Language.ENGLISH)
        assert interface.default_language == Language.ENGLISH
        assert len(interface.translations) == 0
        assert len(interface.terminology_mappings) == 0
    
    def test_add_translation(self):
        """Test adding translations."""
        interface = MultilingualInterface(Language.ENGLISH)
        
        translation = TranslationEntry(
            entry_id="trans_001",
            key="dashboard.title",
            source_language=Language.ENGLISH,
            target_language=Language.SPANISH,
            source_text="Medical Dashboard",
            translated_text="Panel Médico",
            context="main_ui",
            translation_quality=0.95,
            verified_by_medical_expert=True
        )
        
        interface.add_translation(translation)
        
        assert "dashboard.title" in interface.translations
        assert Language.SPANISH in interface.supported_languages
        assert interface.translate("dashboard.title", Language.SPANISH) == "Panel Médico"
    
    def test_translation_fallback(self):
        """Test translation fallback to default language."""
        interface = MultilingualInterface(Language.ENGLISH)
        
        # Add English translation only
        translation = TranslationEntry(
            entry_id="trans_002",
            key="alert.warning",
            source_language=Language.ENGLISH,
            target_language=Language.ENGLISH,
            source_text="Warning",
            translated_text="Warning",
            context="alerts",
            translation_quality=1.0
        )
        
        interface.add_translation(translation)
        
        # Request non-existent French translation, should fallback to English
        result = interface.translate("alert.warning", Language.FRENCH)
        assert result == "Warning"
    
    def test_terminology_mapping(self):
        """Test medical terminology mapping."""
        interface = MultilingualInterface()
        
        mapping = TerminologyMapping(
            mapping_id="map_001",
            source_standard=TerminologyStandard.ICD10,
            target_standard=TerminologyStandard.SNOMED_CT,
            source_code="I50.9",
            target_code="84114007",
            source_term="Heart failure, unspecified",
            target_term="Heart failure",
            mapping_confidence=0.92,
            bidirectional=True
        )
        
        interface.add_terminology_mapping(mapping)
        
        result = interface.map_terminology("I50.9", TerminologyStandard.ICD10, TerminologyStandard.SNOMED_CT)
        assert result is not None
        assert result.target_code == "84114007"
        assert result.mapping_confidence == 0.92
    
    def test_interface_statistics(self):
        """Test interface statistics."""
        interface = MultilingualInterface()
        
        # Add some data
        for i, lang in enumerate([Language.SPANISH, Language.FRENCH, Language.GERMAN]):
            translation = TranslationEntry(
                entry_id=f"trans_{i}",
                key=f"key_{i}",
                source_language=Language.ENGLISH,
                target_language=lang,
                source_text=f"Text {i}",
                translated_text=f"Translated {i}",
                context="test",
                translation_quality=0.9
            )
            interface.add_translation(translation)
        
        stats = interface.get_statistics()
        assert stats['supported_languages'] == 3
        assert stats['translation_keys'] == 3


class TestRegionalGuidelineEngine:
    """Tests for regional guideline adaptation."""
    
    def test_engine_initialization(self):
        """Test guideline engine initialization."""
        engine = RegionalGuidelineEngine()
        assert len(engine.guidelines) == 0
        assert len(engine.condition_index) == 0
    
    def test_add_guideline(self):
        """Test adding regional guidelines."""
        engine = RegionalGuidelineEngine()
        
        guideline = ClinicalGuideline(
            guideline_id="guide_001",
            region=Region.NORTH_AMERICA,
            condition="Diabetes",
            guideline_name="ADA Diabetes Management",
            version="2025.1",
            effective_date=datetime.now(),
            recommendations=[
                {"drug": "metformin", "dosage": "500mg", "frequency": "twice daily"}
            ],
            evidence_level="A"
        )
        
        engine.add_guideline(guideline)
        
        assert len(engine.guidelines[Region.NORTH_AMERICA]) == 1
        assert len(engine.condition_index["diabetes"]) == 1
    
    def test_get_guidelines_by_region(self):
        """Test retrieving guidelines by region."""
        engine = RegionalGuidelineEngine()
        
        # Add guidelines for different regions
        for region in [Region.NORTH_AMERICA, Region.EUROPE, Region.ASIA_PACIFIC]:
            guideline = ClinicalGuideline(
                guideline_id=f"guide_{region.value}",
                region=region,
                condition="Hypertension",
                guideline_name=f"{region.value} HTN Guidelines",
                version="1.0",
                effective_date=datetime.now(),
                recommendations=[{"drug": "lisinopril"}],
                evidence_level="A"
            )
            engine.add_guideline(guideline)
        
        na_guidelines = engine.get_guidelines_for_region(Region.NORTH_AMERICA)
        assert len(na_guidelines) == 1
        assert na_guidelines[0].region == Region.NORTH_AMERICA
    
    def test_adapt_guideline(self):
        """Test adapting guideline for different region."""
        engine = RegionalGuidelineEngine()
        
        source_guideline = ClinicalGuideline(
            guideline_id="guide_source",
            region=Region.NORTH_AMERICA,
            condition="Diabetes",
            guideline_name="NA Diabetes Guidelines",
            version="1.0",
            effective_date=datetime.now(),
            recommendations=[{"treatment": "intensive care"}],
            evidence_level="A"
        )
        
        adapted = engine.adapt_guideline(source_guideline, Region.ASIA_PACIFIC)
        
        assert adapted.region == Region.ASIA_PACIFIC
        assert adapted.adapted_from == source_guideline.guideline_id
        assert "Adapted" in adapted.guideline_name
        assert adapted.evidence_level == source_guideline.evidence_level


class TestConstrainedDeploymentManager:
    """Tests for constrained deployment management."""
    
    def test_manager_initialization(self):
        """Test deployment manager initialization."""
        manager = ConstrainedDeploymentManager()
        assert len(manager.configurations) >= 4  # Default configs
        assert DeploymentMode.FULL_CLOUD in manager.configurations
        assert DeploymentMode.LOW_BANDWIDTH in manager.configurations
    
    def test_get_configuration(self):
        """Test getting deployment configuration."""
        manager = ConstrainedDeploymentManager()
        
        config = manager.get_configuration(DeploymentMode.LOW_BANDWIDTH)
        assert config is not None
        assert config.mode == DeploymentMode.LOW_BANDWIDTH
        assert config.compression_enabled is True
        assert config.max_bandwidth_mbps <= 1.0
    
    def test_optimize_for_bandwidth(self):
        """Test bandwidth-based optimization."""
        manager = ConstrainedDeploymentManager()
        
        # Test different bandwidth scenarios
        high_bw_config = manager.optimize_for_bandwidth(200.0)
        assert high_bw_config.mode == DeploymentMode.FULL_CLOUD
        
        medium_bw_config = manager.optimize_for_bandwidth(15.0)
        assert medium_bw_config.mode == DeploymentMode.EDGE_COMPUTING
        
        low_bw_config = manager.optimize_for_bandwidth(0.8)
        assert low_bw_config.mode == DeploymentMode.LOW_BANDWIDTH
        
        offline_config = manager.optimize_for_bandwidth(0.0)
        assert offline_config.mode == DeploymentMode.OFFLINE
    
    def test_deployment(self):
        """Test deploying with configuration."""
        manager = ConstrainedDeploymentManager()
        
        config = manager.get_configuration(DeploymentMode.EDGE_COMPUTING)
        result = manager.deploy("deploy_001", config)
        
        assert result['deployment_id'] == "deploy_001"
        assert result['mode'] == DeploymentMode.EDGE_COMPUTING.value
        assert 'deployment_time_ms' in result
        assert "deploy_001" in manager.active_deployments
    
    def test_performance_measurement(self):
        """Test deployment performance measurement."""
        manager = ConstrainedDeploymentManager()
        
        config = manager.get_configuration(DeploymentMode.EDGE_COMPUTING)
        manager.deploy("deploy_002", config)
        
        manager.measure_performance("deploy_002", latency_ms=45.0, throughput_mbps=8.5)
        
        assert "deploy_002" in manager.performance_metrics
        assert manager.performance_metrics["deploy_002"]['latency_ms'] == 45.0
        assert manager.performance_metrics["deploy_002"]['throughput_mbps'] == 8.5


class TestGlobalDataGovernanceFramework:
    """Tests for global data governance."""
    
    def test_framework_initialization(self):
        """Test governance framework initialization."""
        framework = GlobalDataGovernanceFramework()
        assert len(framework.policies) >= 3  # Default policies
        assert Region.NORTH_AMERICA in framework.policies
        assert Region.EUROPE in framework.policies
    
    def test_get_policy(self):
        """Test retrieving regional policy."""
        framework = GlobalDataGovernanceFramework()
        
        na_policy = framework.get_policy(Region.NORTH_AMERICA)
        assert na_policy is not None
        assert na_policy.region == Region.NORTH_AMERICA
        assert 'HIPAA' in na_policy.compliance_frameworks
        
        eu_policy = framework.get_policy(Region.EUROPE)
        assert eu_policy is not None
        assert 'GDPR' in eu_policy.compliance_frameworks
    
    def test_collaboration_request_approved(self):
        """Test approved collaboration request."""
        framework = GlobalDataGovernanceFramework()
        
        # North America allows sharing within North America
        request = framework.request_collaboration(
            source_region=Region.NORTH_AMERICA,
            target_region=Region.NORTH_AMERICA,
            data_type="clinical_data",
            purpose="research"
        )
        
        assert request['approved'] is True
        assert request['source_region'] == Region.NORTH_AMERICA.value
        assert request['target_region'] == Region.NORTH_AMERICA.value
    
    def test_collaboration_request_denied(self):
        """Test denied collaboration request."""
        framework = GlobalDataGovernanceFramework()
        
        # Europe typically doesn't allow sharing to other regions (GDPR)
        request = framework.request_collaboration(
            source_region=Region.EUROPE,
            target_region=Region.NORTH_AMERICA,
            data_type="patient_data",
            purpose="analysis"
        )
        
        # Check if denied based on policy
        assert 'approved' in request
    
    def test_verify_compliance(self):
        """Test compliance verification."""
        framework = GlobalDataGovernanceFramework()
        
        result = framework.verify_compliance(Region.NORTH_AMERICA, ['HIPAA'])
        assert result['compliant'] is True
        assert 'HIPAA' in result['supported_frameworks']
        
        result2 = framework.verify_compliance(Region.EUROPE, ['GDPR', 'HIPAA'])
        assert 'GDPR' in result2['supported_frameworks']
    
    def test_audit_log(self):
        """Test audit logging."""
        framework = GlobalDataGovernanceFramework()
        
        # Trigger some events
        framework.request_collaboration(
            Region.NORTH_AMERICA, Region.NORTH_AMERICA,
            "test_data", "testing"
        )
        framework.verify_compliance(Region.EUROPE, ['GDPR'])
        
        audit_log = framework.get_audit_log()
        assert len(audit_log) >= 2
        assert all('event_type' in entry for entry in audit_log)


class TestInternationalHealthcareSystem:
    """Tests for integrated international healthcare system."""
    
    def test_system_initialization(self):
        """Test system initialization."""
        system = create_international_healthcare_system(
            default_language=Language.ENGLISH,
            default_region=Region.NORTH_AMERICA
        )
        
        assert system.multilingual is not None
        assert system.guidelines is not None
        assert system.deployment is not None
        assert system.governance is not None
        assert system.default_region == Region.NORTH_AMERICA
    
    def test_system_statistics(self):
        """Test comprehensive system statistics."""
        system = create_international_healthcare_system()
        
        stats = system.get_system_statistics()
        
        assert 'multilingual' in stats
        assert 'guidelines' in stats
        assert 'deployment' in stats
        assert 'governance' in stats
        assert 'default_region' in stats
        
        # Verify statistics structure
        assert 'supported_languages' in stats['multilingual']
        assert 'total_guidelines' in stats['guidelines']
        assert 'total_configurations' in stats['deployment']
        assert 'total_policies' in stats['governance']


class TestIntegrationScenarios:
    """Integration tests for complete workflows."""
    
    def test_multilingual_clinical_workflow(self):
        """Test complete multilingual clinical workflow."""
        system = create_international_healthcare_system(
            Language.ENGLISH,
            Region.NORTH_AMERICA
        )
        
        # Add translations
        translation = TranslationEntry(
            entry_id="trans_clinical",
            key="diagnosis.label",
            source_language=Language.ENGLISH,
            target_language=Language.SPANISH,
            source_text="Diagnosis",
            translated_text="Diagnóstico",
            context="clinical",
            translation_quality=0.98,
            verified_by_medical_expert=True
        )
        system.multilingual.add_translation(translation)
        
        # Translate for Spanish-speaking patient
        result = system.multilingual.translate("diagnosis.label", Language.SPANISH)
        assert result == "Diagnóstico"
    
    def test_regional_deployment_workflow(self):
        """Test regional deployment workflow."""
        system = create_international_healthcare_system()
        
        # Add regional guideline
        guideline = ClinicalGuideline(
            guideline_id="guide_workflow",
            region=Region.ASIA_PACIFIC,
            condition="COVID-19",
            guideline_name="APAC COVID Guidelines",
            version="2.0",
            effective_date=datetime.now(),
            recommendations=[{"treatment": "protocol_a"}],
            evidence_level="A"
        )
        system.guidelines.add_guideline(guideline)
        
        # Deploy in low bandwidth region
        config = system.deployment.optimize_for_bandwidth(2.0)
        deployment = system.deployment.deploy("deploy_apac", config)
        
        assert deployment['mode'] == DeploymentMode.LOW_BANDWIDTH.value
        assert len(system.guidelines.get_guidelines_for_region(Region.ASIA_PACIFIC)) == 1
    
    def test_global_collaboration_workflow(self):
        """Test global data collaboration workflow."""
        system = create_international_healthcare_system()
        
        # Request collaboration
        request = system.governance.request_collaboration(
            source_region=Region.ASIA_PACIFIC,
            target_region=Region.NORTH_AMERICA,
            data_type="anonymized_research_data",
            purpose="rare_disease_research"
        )
        
        assert 'request_id' in request
        assert 'approved' in request
        assert 'encryption_required' in request


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
