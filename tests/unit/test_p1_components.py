"""
Tests for P1 API components.
"""

import pytest
from src.aimedres.security.oidc_auth import OIDCAuthProvider, RoleMapper, MFAManager
from src.aimedres.api.model_routes import ModelRegistry
from src.aimedres.api.explain_routes import ExplainabilityEngine
from src.aimedres.api.fhir_routes import FHIRMockServer
from src.aimedres.api.case_routes import CaseManager
from src.aimedres.security.audit_export import AuditLogExporter


class TestOIDCAuth:
    """Test OIDC authentication components."""
    
    def test_oidc_provider_initialization(self):
        """Test OIDC provider can be initialized."""
        config = {
            'provider_url': 'https://auth.example.com',
            'client_id': 'test-client',
            'client_secret': 'test-secret',
            'redirect_uri': 'http://localhost/callback'
        }
        provider = OIDCAuthProvider(config)
        
        assert provider.provider_url == 'https://auth.example.com'
        assert provider.client_id == 'test-client'
    
    def test_authorization_url_generation(self):
        """Test authorization URL generation."""
        config = {
            'provider_url': 'https://auth.example.com',
            'client_id': 'test-client',
            'client_secret': 'test-secret',
            'redirect_uri': 'http://localhost/callback'
        }
        provider = OIDCAuthProvider(config)
        
        url = provider.get_authorization_url('test-state-123')
        
        assert 'client_id=test-client' in url
        assert 'state=test-state-123' in url
        assert 'response_type=code' in url
    
    def test_role_mapper(self):
        """Test role mapping."""
        oidc_roles = ['clinician', 'admin']
        app_roles = RoleMapper.map_roles(oidc_roles)
        
        assert 'clinician' in app_roles
        assert 'admin' in app_roles
        assert 'user' in app_roles  # Should include base user role
    
    def test_mfa_manager(self):
        """Test MFA manager."""
        config = {'mfa_enabled': True}
        mfa = MFAManager(config)
        
        # MFA required for admin
        assert mfa.is_mfa_required('user-1', ['admin'])
        
        # MFA required for clinician
        assert mfa.is_mfa_required('user-2', ['clinician'])
        
        # MFA not required for researcher
        assert not mfa.is_mfa_required('user-3', ['researcher'])
    
    def test_mfa_challenge_verification(self):
        """Test MFA challenge creation and verification."""
        config = {'mfa_enabled': True}
        mfa = MFAManager(config)
        
        # Create challenge
        challenge = mfa.create_challenge('user-1', 'totp')
        assert 'challenge_id' in challenge
        assert challenge['method'] == 'totp'
        
        # Verify with correct code
        assert mfa.verify_challenge(challenge['challenge_id'], '123456')


class TestModelRegistry:
    """Test model serving components."""
    
    def test_model_registry_initialization(self):
        """Test model registry initializes with models."""
        registry = ModelRegistry()
        
        assert len(registry.models) > 0
        assert 'alzheimer_v1' in registry.models
    
    def test_get_model_card(self):
        """Test retrieving model card."""
        registry = ModelRegistry()
        
        card = registry.get_model_card('alzheimer_v1')
        
        assert card is not None
        assert 'name' in card
        assert 'version' in card
        assert 'validation_metrics' in card
        assert 'intended_use' in card
    
    def test_get_latest_model(self):
        """Test getting latest model."""
        registry = ModelRegistry()
        
        card = registry.get_model_card('latest')
        
        assert card is not None
        assert 'version' in card
    
    def test_list_models(self):
        """Test listing all models."""
        registry = ModelRegistry()
        
        models = registry.list_models()
        
        assert len(models) > 0
        assert all('model_id' in m for m in models)
        assert all('name' in m for m in models)


class TestExplainabilityEngine:
    """Test explainability components."""
    
    def test_feature_attribution(self):
        """Test feature attribution computation."""
        engine = ExplainabilityEngine()
        
        attributions = engine.compute_feature_attribution(
            'pred-123',
            {'age': 72, 'mmse': 24}
        )
        
        assert len(attributions) > 0
        assert all('feature' in a for a in attributions)
        assert all('importance' in a for a in attributions)
    
    def test_uncertainty_estimation(self):
        """Test uncertainty estimation."""
        engine = ExplainabilityEngine()
        
        uncertainty = engine.compute_uncertainty(
            'pred-123',
            {'age': 72}
        )
        
        assert 'confidence' in uncertainty
        assert 'total_uncertainty' in uncertainty
        assert 'epistemic_uncertainty' in uncertainty
        assert 'aleatoric_uncertainty' in uncertainty
    
    def test_explanation_summary(self):
        """Test explanation summary generation."""
        engine = ExplainabilityEngine()
        
        attributions = [
            {'feature': 'Age', 'importance': 0.3, 'contribution': 0.15},
            {'feature': 'MMSE', 'importance': 0.4, 'contribution': -0.18}
        ]
        uncertainty = {
            'confidence': 0.82,
            'total_uncertainty': 0.15
        }
        
        summary = engine.generate_explanation_summary(
            'pred-123',
            attributions,
            uncertainty
        )
        
        assert isinstance(summary, str)
        assert len(summary) > 0
        assert 'Age' in summary or 'MMSE' in summary


class TestFHIRMockServer:
    """Test FHIR integration components."""
    
    def test_fhir_server_initialization(self):
        """Test FHIR mock server initialization."""
        server = FHIRMockServer()
        
        assert len(server.patients) > 0
        assert len(server.consent_records) > 0
    
    def test_consent_check(self):
        """Test consent checking."""
        server = FHIRMockServer()
        
        # Patient with consent
        assert server.check_consent('patient-001', 'research')
        
        # Patient without consent
        assert not server.check_consent('patient-003', 'research')
    
    def test_get_patient(self):
        """Test patient retrieval."""
        server = FHIRMockServer()
        
        patient = server.get_patient('patient-001')
        
        assert patient is not None
        assert patient['resourceType'] == 'Patient'
        assert patient['id'] == 'patient-001'
    
    def test_list_patients(self):
        """Test patient listing."""
        server = FHIRMockServer()
        
        patients = server.list_patients()
        
        assert len(patients) > 0
        assert all(p['resourceType'] == 'Patient' for p in patients)


class TestCaseManager:
    """Test case management components."""
    
    def test_case_manager_initialization(self):
        """Test case manager initialization."""
        manager = CaseManager()
        
        assert len(manager.cases) > 0
    
    def test_list_cases(self):
        """Test case listing."""
        manager = CaseManager()
        
        result = manager.list_cases()
        
        assert 'cases' in result
        assert 'total' in result
        assert len(result['cases']) > 0
    
    def test_list_cases_with_filter(self):
        """Test case listing with status filter."""
        manager = CaseManager()
        
        result = manager.list_cases(status='pending')
        
        assert all(c['status'] == 'pending' for c in result['cases'])
    
    def test_get_case(self):
        """Test case retrieval."""
        manager = CaseManager()
        
        case = manager.get_case('case-001')
        
        assert case is not None
        assert case['case_id'] == 'case-001'
    
    def test_approve_case(self):
        """Test case approval."""
        manager = CaseManager()
        
        updated_case = manager.approve_case(
            'case-001',
            'clinician-001',
            'approve',
            'Clinical review completed'
        )
        
        assert updated_case['status'] == 'completed'
        assert updated_case['approved_by'] == 'clinician-001'


class TestAuditExport:
    """Test audit export components."""
    
    def test_audit_exporter_json(self):
        """Test JSON export."""
        from security.blockchain_records import BlockchainMedicalRecords
        
        blockchain = BlockchainMedicalRecords({'blockchain_enabled': True})
        exporter = AuditLogExporter(blockchain)
        
        # Add test audit record
        blockchain.record_audit_trail(
            'access',
            'patient-001',
            'user-001',
            'view',
            {'ip_address': '127.0.0.1'}
        )
        
        # Export as JSON
        json_export = exporter.export_audit_logs(format='json')
        
        assert isinstance(json_export, str)
        assert 'export_timestamp' in json_export
        assert 'records' in json_export
    
    def test_compliance_report(self):
        """Test compliance report generation."""
        from security.blockchain_records import BlockchainMedicalRecords
        
        blockchain = BlockchainMedicalRecords({'blockchain_enabled': True})
        exporter = AuditLogExporter(blockchain)
        
        # Add test records
        blockchain.record_audit_trail(
            'access',
            'patient-001',
            'user-001',
            'view',
            {}
        )
        
        # Generate report
        report = exporter.generate_compliance_report()
        
        assert 'report_generated' in report
        assert 'summary' in report
        assert 'total_audit_entries' in report['summary']
        assert 'compliance_status' in report
