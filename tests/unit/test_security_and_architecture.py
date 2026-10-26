"""
Unit tests for Domain-Driven Architecture Decoupling (Phase 3.1)
and Enterprise Security Hardening with Database Store & Fail-Closed Gates (Phase 3.3).
"""

import os
import pytest
from flask import Flask, jsonify

from aimedres.security.auth import (
    AuthDatabaseStore,
    SecureAuthManager,
    SecurityConfigurationError,
    require_auth,
)
from aimedres.security.secure_medical_processor import SecureMedicalDataProcessor
from aimedres.clinical.medical_processor import SecureMedicalDataProcessor as LegacyMedicalDataProcessor
from aimedres.service import SecureAPIServer, ModelRegistry, create_app
from aimedres.simulation import (
    SpecializedMedicalAgent,
    RadiologistAgent,
    NeurologistAgent,
    ConsensusManager,
    assert_simulation_allowed,
)
from aimedres.research import (
    ParkinsonsResearchPipeline,
    LocalExperimentTracker,
    OASISDataset,
    PROACTALSDataset,
    PTBXLECGDataset,
)


class TestDomainDecoupling:
    """Verify clean architectural boundaries between research, service, and simulation (Phase 3.1)."""

    def test_research_package_exports(self):
        assert ParkinsonsResearchPipeline is not None
        assert LocalExperimentTracker is not None
        assert OASISDataset is not None
        assert PROACTALSDataset is not None
        assert PTBXLECGDataset is not None

    def test_service_package_exports(self):
        assert SecureAPIServer is not None
        assert ModelRegistry is not None
        assert callable(create_app)

    def test_simulation_package_quarantine(self):
        assert SpecializedMedicalAgent is not None
        assert RadiologistAgent is not None
        assert NeurologistAgent is not None
        assert ConsensusManager is not None

        # Verify simulation allowed in dev/test mode
        assert_simulation_allowed()

    def test_simulation_quarantine_enforcement_in_production(self, monkeypatch):
        monkeypatch.setenv("AIMEDRES_ENV", "production")
        monkeypatch.setenv("STRICT_CLINICAL_ISOLATION", "true")
        with pytest.raises(RuntimeError, match="CRITICAL ARCHITECTURAL VIOLATION"):
            assert_simulation_allowed()


class TestEnterpriseSecurityHardening:
    """Verify database-backed token management, revocation, and fail-closed security (Phase 3.3)."""

    def test_auth_database_store_lifecycle(self, tmp_path):
        db_file = tmp_path / "auth_test.db"
        store = AuthDatabaseStore(db_path=db_file)

        # Generate key
        salt = os.urandom(32)
        key_hash = b"sample_hash_bytes_for_testing"
        store.store_api_key("dmk_test_key_123", "user_1", ["user", "clinician"], salt, key_hash)

        # Revocation
        assert store.revoke_api_key("dmk_test_key_123") is True
        # Second attempt should return false (already revoked)
        assert store.revoke_api_key("dmk_non_existent") is False

        # Audit log verification
        logs = store.get_audit_logs(limit=10)
        assert len(logs) >= 2
        event_types = [log["event_type"] for log in logs]
        assert "KEY_GENERATED" in event_types
        assert "KEY_REVOKED" in event_types

    def test_secure_auth_manager_key_and_session_revocation(self):
        auth = SecureAuthManager()
        # Generate new user key
        api_key = auth._generate_api_key("dr_smith", ["clinician"])

        # Validate active key
        valid, info = auth.validate_api_key(api_key)
        assert valid is True
        assert info["user_id"] == "dr_smith"

        # Revoke key
        revoked = auth.revoke_api_key(api_key)
        assert revoked is True

        # Validation after revocation must fail
        valid_after, info_after = auth.validate_api_key(api_key)
        assert valid_after is False
        assert info_after is None

        # Test JWT token generation and revocation
        token = auth.generate_jwt_token({"user_id": "dr_smith", "roles": ["clinician"]})
        valid_jwt, payload = auth.validate_jwt_token(token)
        assert valid_jwt is True
        assert payload["user_id"] == "dr_smith"

        # Revoke JWT session
        auth.revoke_session(token)
        valid_jwt_after, _ = auth.validate_jwt_token(token)
        assert valid_jwt_after is False

    def test_authentication_accepts_header_but_rejects_query_string_key(self, tmp_path):
        app = Flask(__name__)
        app.auth_manager = SecureAuthManager({"auth_db_path": tmp_path / "auth.db"})
        api_key = app.auth_manager._generate_api_key("researcher", ["user"])

        @app.get("/protected")
        @require_auth
        def protected():
            return jsonify({"ok": True})

        client = app.test_client()
        assert client.get(f"/protected?api_key={api_key}").status_code == 401
        assert client.get("/protected", headers={"X-API-Key": api_key}).status_code == 200

    def test_fail_closed_validation_in_production_mode(self, monkeypatch):
        # Set production environment without required 32-char secret
        monkeypatch.setenv("AIMEDRES_ENV", "production")
        monkeypatch.delenv("JWT_SECRET", raising=False)

        with pytest.raises(SecurityConfigurationError, match="CRITICAL SECURITY CONFIGURATION ERROR"):
            SecureAuthManager(config={"jwt_secret": "short-secret"})

    def test_server_fail_closed_on_missing_flask_secret_in_production(self, monkeypatch):
        monkeypatch.setenv("AIMEDRES_ENV", "production")
        monkeypatch.delenv("FLASK_SECRET_KEY", raising=False)

        with pytest.raises(SecurityConfigurationError, match="CRITICAL SECURITY CONFIGURATION ERROR"):
            SecureAPIServer()

    def test_prediction_endpoint_disables_unapproved_model_serving(self):
        server = SecureAPIServer()
        api_key = next(iter(server.auth_manager.api_keys))
        client = server.app.test_client()

        missing_model = client.post(
            "/api/v1/predict",
            headers={"X-API-Key": api_key},
            json={"input": {"AGE": 60}},
        )
        assert missing_model.status_code == 400

        unavailable_model = client.post(
            "/api/v1/predict",
            headers={"X-API-Key": api_key},
            json={
                "input": {"AGE": 60},
                "model_id": "ppmi_synthetic_fixture_v1",
            },
        )
        assert unavailable_model.status_code == 503

        unavailable_agent = client.post(
            "/api/v1/agent/think",
            headers={"X-API-Key": api_key},
            json={"input": [1.0, 2.0]},
        )
        assert unavailable_agent.status_code == 503

    def test_maintained_api_does_not_expose_mock_or_prototype_routes(self):
        server = SecureAPIServer()
        client = server.app.test_client()

        for path in (
            "/api/v1/fhir/metadata",
            "/api/v1/cases",
            "/api/v1/explain/health",
            "/api/v1/dicom/health",
            "/api/v1/visualization/health",
            "/api/v1/canary/health",
            "/api/v1/quantum/health",
        ):
            assert client.get(path).status_code == 404

    @pytest.mark.parametrize(
        "processor_type", (SecureMedicalDataProcessor, LegacyMedicalDataProcessor)
    )
    def test_legacy_medical_processors_refuse_pickle_backed_inference(self, processor_type):
        processor = object.__new__(processor_type)

        with pytest.raises(RuntimeError, match="inference is disabled"):
            processor.secure_inference("untrusted-model", {"AGE": 60}, "researcher")
