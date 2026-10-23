"""Unit coverage for security controls maintained in ``aimedres.security``."""

import pytest

from aimedres.security.encryption import DataEncryption
from aimedres.security.phi_scrubber import PHIScrubber, enforce_phi_free_ingestion
from aimedres.security.validation import InputValidator


def test_sensitive_medical_fields_are_encrypted_and_recoverable():
    encryption = DataEncryption(master_password="unit-test-master-password")
    record = {
        "patient_id": "patient-001",
        "name": "Test Patient",
        "diagnosis": "research-only",
    }

    encrypted = encryption.encrypt_medical_data(record)

    assert encrypted["patient_id"] != record["patient_id"]
    assert encrypted["name"] != record["name"]
    assert encryption.decrypt_data(encrypted["patient_id"]) == record["patient_id"]
    assert encrypted["diagnosis"] == record["diagnosis"]


def test_phi_scrubber_detects_and_blocks_sensitive_input():
    scrubber = PHIScrubber(aggressive=True)
    sensitive_text = "Contact patient@example.com or call 555-123-4567."

    result = scrubber.detect_phi(sensitive_text)

    assert result.has_phi
    assert {"email", "phone"}.issubset(result.phi_types_found)
    with pytest.raises(ValueError, match="PHI detected"):
        enforce_phi_free_ingestion(sensitive_text, "unit-test-input")


def test_input_validator_rejects_injection_and_accepts_valid_medical_values():
    validator = InputValidator()

    assert not validator.validate_json({"query": "1; DROP TABLE patients"})
    is_valid, errors = validator.validate_medical_data({"Age": 68, "MMSE": 27})
    assert is_valid
    assert errors == []
