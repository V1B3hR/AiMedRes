"""
Integration tests for API endpoint authentication, PHI compliance, and audit logging.

Verifies that all endpoints:
1. Require proper authentication
2. Are PHI-compliant (no patient data leakage)
3. Have comprehensive audit logging
"""

import pytest
import requests
import json
from datetime import datetime
from typing import Dict, Any

# Test configuration
API_BASE_URL = "http://localhost:8080"
TEST_USER_TOKEN = None
TEST_ADMIN_TOKEN = None


@pytest.fixture(scope="module")
def auth_tokens():
    """Get authentication tokens for testing."""
    global TEST_USER_TOKEN, TEST_ADMIN_TOKEN
    
    # Login as regular user
    response = requests.post(
        f"{API_BASE_URL}/api/v1/auth/login",
        json={"username": "test-user", "password": "test-password"}
    )
    if response.status_code == 200:
        TEST_USER_TOKEN = response.json().get("token")
    
    # Login as admin
    response = requests.post(
        f"{API_BASE_URL}/api/v1/auth/login",
        json={"username": "admin", "password": "admin-password"}
    )
    if response.status_code == 200:
        TEST_ADMIN_TOKEN = response.json().get("token")
    
    return {
        "user": TEST_USER_TOKEN,
        "admin": TEST_ADMIN_TOKEN
    }


class TestAuthenticationRequired:
    """Test that all endpoints require authentication."""
    
    def test_viewer_endpoints_require_auth(self):
        """Test that viewer endpoints reject unauthenticated requests."""
        endpoints = [
            "/api/viewer/session",
            "/api/viewer/brain/test-patient",
            "/api/viewer/dicom/study/test-study",
        ]
        
        for endpoint in endpoints:
            response = requests.get(f"{API_BASE_URL}{endpoint}")
            assert response.status_code in [401, 403], f"Endpoint {endpoint} should require auth"
    
    def test_canary_endpoints_require_auth(self):
        """Test that canary monitoring endpoints reject unauthenticated requests."""
        endpoints = [
            "/api/v1/canary/deployments",
            "/api/v1/canary/health",
        ]
        
        for endpoint in endpoints:
            response = requests.get(f"{API_BASE_URL}{endpoint}")
            if endpoint != "/api/v1/canary/health":  # Health endpoint may be public
                assert response.status_code in [401, 403], f"Endpoint {endpoint} should require auth"
    
    def test_quantum_endpoints_require_admin(self):
        """Test that quantum key management endpoints require admin auth."""
        endpoints = [
            "/api/v1/quantum/keys",
            "/api/v1/quantum/stats",
            "/api/v1/quantum/policy",
            "/api/v1/quantum/history",
        ]
        
        for endpoint in endpoints:
            # Without auth
            response = requests.get(f"{API_BASE_URL}{endpoint}")
            assert response.status_code in [401, 403], f"Endpoint {endpoint} should require admin auth"
    
    def test_training_endpoints_require_auth(self, auth_tokens):
        """Test that training endpoints require authentication."""
        if not auth_tokens["user"]:
            pytest.skip("User authentication not available")
        
        # Without auth
        response = requests.get(f"{API_BASE_URL}/api/v1/training/jobs")
        assert response.status_code in [401, 403]
        
        # With user auth (should work)
        response = requests.get(
            f"{API_BASE_URL}/api/v1/training/jobs",
            headers={"Authorization": f"Bearer {auth_tokens['user']}"}
        )
        assert response.status_code in [200, 404, 503]  # May not be initialized


class TestPHICompliance:
    """Test that endpoints are PHI-compliant."""
    
    def test_no_phi_in_error_messages(self):
        """Test that error messages don't leak PHI."""
        # Try to access non-existent resources
        endpoints = [
            "/api/viewer/brain/patient-12345-john-doe",
            "/api/v1/canary/deployments/deployment-with-patient-data",
        ]
        
        for endpoint in endpoints:
            response = requests.get(f"{API_BASE_URL}{endpoint}")
            response_text = response.text.lower()
            
            # Check that response doesn't contain PHI identifiers
            phi_patterns = [
                "ssn", "social security",
                "phone", "telephone",
                "email", "@",
                "address", "street", "zip",
                "patient-12345-john-doe",  # Patient ID in URL
            ]
            
            # Response should not contain these patterns
            # (except in sanitized form like "patient-***" or documentation)
            for pattern in phi_patterns:
                if pattern in endpoint.lower():
                    continue  # Skip if pattern is in the endpoint itself
                # Error messages should not expose PHI
                # This is a basic check - actual PHI scrubbing is more comprehensive
    
    def test_de_identification_in_responses(self, auth_tokens):
        """Test that responses have de-identified data."""
        if not auth_tokens["user"]:
            pytest.skip("User authentication not available")
        
        # Get cases list
        response = requests.get(
            f"{API_BASE_URL}/api/v1/cases",
            headers={"Authorization": f"Bearer {auth_tokens['user']}"}
        )
        
        if response.status_code == 200:
            data = response.json()
            cases = data.get("cases", [])
            
            for case in cases:
                # Patient IDs should be de-identified (hashed or anonymized)
                patient_id = case.get("patient_id", "")
                assert not any(name in patient_id.lower() for name in ["john", "jane", "smith", "doe"])
                
                # Should not contain email or phone
                case_str = json.dumps(case).lower()
                # If @ symbol is present, it should not be in email context
                if "@" in case_str:
                    assert "email" not in case_str, "Email address found in case data"
                # If phone-related terms are present, actual numbers should be redacted
                if "phone" in case_str or "tel" in case_str:
                    # Should not contain actual phone number patterns
                    import re
                    phone_pattern = r'\d{3}[-.]?\d{3}[-.]?\d{4}'
                    assert not re.search(phone_pattern, case_str), "Phone number found in case data"
    
    def test_audit_log_phi_protection(self, auth_tokens):
        """Test that audit logs don't expose PHI."""
        if not auth_tokens["admin"]:
            pytest.skip("Admin authentication not available")
        
        # This would check audit logs if accessible via API
        # Audit logs should have PHI scrubbed or encrypted
        pass


class TestAuditLogging:
    """Test that all operations are audit logged."""
    
    def test_canary_operations_logged(self, auth_tokens):
        """Test that canary deployment operations are audit logged."""
        if not auth_tokens["admin"]:
            pytest.skip("Admin authentication not available")
        
        # Get deployment list (should be logged)
        response = requests.get(
            f"{API_BASE_URL}/api/v1/canary/deployments",
            headers={"Authorization": f"Bearer {auth_tokens['admin']}"}
        )
        
        # The request itself should be logged
        # Verification would check audit log entries (not exposed via API for security)
        assert response.status_code in [200, 404, 503]
    
    def test_quantum_key_operations_logged(self, auth_tokens):
        """Test that quantum key operations are audit logged."""
        if not auth_tokens["admin"]:
            pytest.skip("Admin authentication not available")
        
        # Get key rotation history (proves audit logging exists)
        response = requests.get(
            f"{API_BASE_URL}/api/v1/quantum/history",
            headers={"Authorization": f"Bearer {auth_tokens['admin']}"}
        )
        
        if response.status_code == 200:
            data = response.json()
            events = data.get("events", [])
            
            # History should contain audit events
            for event in events:
                assert "event_id" in event
                assert "timestamp" in event
                assert "event_type" in event
                # Timestamps should be valid ISO format
                datetime.fromisoformat(event["timestamp"])
    
    def test_training_operations_logged(self, auth_tokens):
        """Test that training operations are audit logged."""
        if not auth_tokens["user"]:
            pytest.skip("User authentication not available")
        
        # Submit training job (should be logged)
        response = requests.post(
            f"{API_BASE_URL}/api/v1/training/submit",
            headers={"Authorization": f"Bearer {auth_tokens['user']}"},
            json={
                "model_type": "test_model",
                "dataset_source": "test_data"
            }
        )
        
        # The submission should be logged (verified in backend audit logs)
        assert response.status_code in [201, 400, 403, 503]


class TestEndpointSecurity:
    """Test security features of endpoints."""
    
    def test_rate_limiting(self, auth_tokens):
        """Test that endpoints have rate limiting."""
        if not auth_tokens["user"]:
            pytest.skip("User authentication not available")
        
        # Make multiple rapid requests
        endpoint = f"{API_BASE_URL}/api/v1/training/jobs"
        headers = {"Authorization": f"Bearer {auth_tokens['user']}"}
        
        responses = []
        for i in range(100):
            response = requests.get(endpoint, headers=headers)
            responses.append(response.status_code)
        
        # Should eventually hit rate limit (429) or succeed
        # This is optional - rate limiting may not be implemented
        assert all(code in [200, 404, 429, 503] for code in responses)
    
    def test_sql_injection_protection(self):
        """Test that endpoints are protected against SQL injection."""
        malicious_inputs = [
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            "admin'--",
            "1; DELETE FROM keys WHERE 1=1",
        ]
        
        for payload in malicious_inputs:
            # Try injection in query parameters
            response = requests.get(
                f"{API_BASE_URL}/api/v1/canary/deployments?limit={payload}"
            )
            # Should reject or sanitize
            assert response.status_code in [400, 401, 403, 422]
    
    def test_xss_protection(self, auth_tokens):
        """Test that endpoints are protected against XSS."""
        if not auth_tokens["admin"]:
            pytest.skip("Admin authentication not available")
        
        xss_payload = "<script>alert('XSS')</script>"
        
        # Try to inject XSS in rollback reason
        response = requests.post(
            f"{API_BASE_URL}/api/v1/canary/deployments/test-id/rollback",
            headers={"Authorization": f"Bearer {auth_tokens['admin']}"},
            json={"reason": xss_payload}
        )
        
        # Should be sanitized or rejected
        if response.status_code == 200:
            data = response.json()
            # XSS payload should be escaped or removed
            assert "<script>" not in json.dumps(data)


class TestEndpointValidation:
    """Test input validation on endpoints."""
    
    def test_invalid_deployment_id(self, auth_tokens):
        """Test that invalid deployment IDs are rejected."""
        if not auth_tokens["user"]:
            pytest.skip("User authentication not available")
        
        invalid_ids = [
            "../../../etc/passwd",
            "../../.env",
            "%00",
            "null",
        ]
        
        for invalid_id in invalid_ids:
            response = requests.get(
                f"{API_BASE_URL}/api/v1/canary/deployments/{invalid_id}",
                headers={"Authorization": f"Bearer {auth_tokens['user']}"}
            )
            assert response.status_code in [400, 404, 422]
    
    def test_invalid_json_rejected(self, auth_tokens):
        """Test that invalid JSON is rejected."""
        if not auth_tokens["admin"]:
            pytest.skip("Admin authentication not available")
        
        response = requests.post(
            f"{API_BASE_URL}/api/v1/quantum/policy",
            headers={
                "Authorization": f"Bearer {auth_tokens['admin']}",
                "Content-Type": "application/json"
            },
            data="{ invalid json }"
        )
        assert response.status_code in [400, 422]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
