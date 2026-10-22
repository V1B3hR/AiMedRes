"""Focused smoke tests for maintained security primitives."""

from aimedres.security.privacy import DataRetentionPolicy
from aimedres.security.validation import InputValidator


def test_medical_data_validation_rejects_out_of_range_values():
    is_valid, errors = InputValidator().validate_medical_data({"Age": 121, "MMSE": 31})

    assert not is_valid
    assert len(errors) == 2


def test_retention_policy_exposes_healthcare_defaults():
    policy = DataRetentionPolicy()

    assert policy.medical_data_retention_days == 2555
    assert policy.audit_logs_retention_days == 2555
    assert policy.gdpr_enabled
