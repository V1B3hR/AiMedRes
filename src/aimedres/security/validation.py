"""
Input validation and security validation utilities.

Provides comprehensive input sanitization and validation for:
- API request data validation
- SQL injection prevention
- XSS attack prevention
- Medical data validation
- Parameter boundary checking
"""

import html
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import quote

security_logger = logging.getLogger("duetmind.security")


class InputValidator:
    """
    Comprehensive input validation and sanitization.

    Security Features:
    - SQL injection prevention
    - XSS attack prevention
    - Input type validation
    - Medical data format validation
    - Length and boundary checking
    - Malicious payload detection
    """

    # Security patterns
    SQL_INJECTION_PATTERNS = [
        r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION|SCRIPT)\b)",
        r"(\b(OR|AND)\s+\d+\s*[=><]\s*\d+)",
        r"(['\";])",
        r"(--|\#|\/\*|\*\/)",
    ]

    XSS_PATTERNS = [
        r"<script[^>]*>.*?</script>",
        r"javascript:",
        r"on\w+\s*=",
        r"<iframe",
        r"<object",
        r"<embed",
    ]

    def __init__(self):
        self.sql_regex = re.compile("|".join(self.SQL_INJECTION_PATTERNS), re.IGNORECASE)
        self.xss_regex = re.compile("|".join(self.XSS_PATTERNS), re.IGNORECASE)

    def sanitize_string(self, value: str, max_length: int = 1000) -> str:
        """
        Sanitize string input against common attacks.

        Args:
            value: Input string to sanitize
            max_length: Maximum allowed length

        Returns:
            Sanitized string
        """
        if not isinstance(value, str):
            raise ValueError("Input must be a string")

        # Check length
        if len(value) > max_length:
            raise ValueError(f"Input exceeds maximum length of {max_length}")

        # HTML escape to prevent XSS
        sanitized = html.escape(value)

        # URL encode special characters
        sanitized = quote(
            sanitized, safe="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.~ "
        )

        return sanitized

    def validate_sql_injection(self, value: str) -> bool:
        """Check for SQL injection patterns."""
        if self.sql_regex.search(value):
            security_logger.warning(f"Potential SQL injection detected: {value[:50]}...")
            return False
        return True

    def validate_xss(self, value: str) -> bool:
        """Check for XSS patterns."""
        if self.xss_regex.search(value):
            security_logger.warning(f"Potential XSS attack detected: {value[:50]}...")
            return False
        return True

    def validate_medical_data(self, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate medical data format and values.

        Args:
            data: Medical data dictionary

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        # Define valid ranges for medical parameters
        medical_ranges = {
            "Age": (0, 120),
            "BMI": (10.0, 70.0),
            "MMSE": (0, 30),
            "CDR": (0.0, 3.0),
            "FunctionalAssessment": (0.0, 10.0),
            "MemoryComplaints": (0, 1),
            "FamilyHistoryAlzheimers": (0, 1),
            "Depression": (0, 1),
            "Gender": (0, 1),
        }

        for field, (min_val, max_val) in medical_ranges.items():
            if field in data:
                value = data[field]

                # Type validation
                if not isinstance(value, (int, float)):
                    try:
                        value = float(value)
                    except (ValueError, TypeError):
                        errors.append(f"Invalid type for {field}: must be numeric")
                        continue

                # Range validation
                if not (min_val <= value <= max_val):
                    errors.append(
                        f"Invalid range for {field}: must be between {min_val} and {max_val}"
                    )

        return len(errors) == 0, errors

    def validate_json_request(
        self, json_data: Any, required_fields: List[str] = None
    ) -> Tuple[bool, List[str]]:
        """
        Validate JSON request data.

        Args:
            json_data: JSON data to validate
            required_fields: List of required field names

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        if not isinstance(json_data, dict):
            errors.append("Request must be a JSON object")
            return False, errors

        # Check required fields
        if required_fields:
            for field in required_fields:
                if field not in json_data:
                    errors.append(f"Missing required field: {field}")

        # Validate string fields for security
        for key, value in json_data.items():
            if isinstance(value, str):
                if not self.validate_sql_injection(value):
                    errors.append(f"Potential SQL injection in field: {key}")

                if not self.validate_xss(value):
                    errors.append(f"Potential XSS attack in field: {key}")

        return len(errors) == 0, errors

    def validate_array(self, arr, name: str = "array"):
        """Validate numpy array for neural network input"""
        import numpy as np

        if not isinstance(arr, np.ndarray):
            raise ValueError(f"{name} must be numpy array")

        if not np.isfinite(arr).all():
            raise ValueError(f"{name} contains invalid values (inf/nan)")

        if arr.size == 0:
            raise ValueError(f"{name} cannot be empty")

        # Check for reasonable value ranges
        if np.any(np.abs(arr) > 1e6):
            raise ValueError(f"{name} contains extremely large values")

        security_logger.debug(f"Array validation passed for {name}: shape {arr.shape}")


class SecurityValidator:
    """
    Advanced security validation for enterprise features.

    Validates:
    - API rate limiting parameters
    - Security configuration
    - Access control policies
    - Data privacy compliance
    """

    def __init__(self):
        self.input_validator = InputValidator()

    def validate_rate_limit_config(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate rate limiting configuration."""
        errors = []

        required_fields = ["requests_per_minute", "burst_limit", "window_size"]
        for field in required_fields:
            if field not in config:
                errors.append(f"Missing rate limit config field: {field}")

        # Validate ranges
        if "requests_per_minute" in config:
            rpm = config["requests_per_minute"]
            if not isinstance(rpm, int) or rpm < 1 or rpm > 10000:
                errors.append("requests_per_minute must be between 1 and 10000")

        if "burst_limit" in config:
            burst = config["burst_limit"]
            if not isinstance(burst, int) or burst < 1 or burst > 1000:
                errors.append("burst_limit must be between 1 and 1000")

        return len(errors) == 0, errors

    def validate_security_headers(self, headers: Dict[str, str]) -> Tuple[bool, List[str]]:
        """Validate security headers are present."""
        errors = []

        required_headers = [
            "X-Frame-Options",
            "X-Content-Type-Options",
            "X-XSS-Protection",
            "Strict-Transport-Security",
        ]

        for header in required_headers:
            if header not in headers:
                errors.append(f"Missing security header: {header}")

        return len(errors) == 0, errors

    def validate_privacy_compliance(self, data_config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate data privacy compliance configuration."""
        errors = []

        # Check for required privacy settings
        privacy_requirements = [
            "data_retention_days",
            "anonymization_enabled",
            "audit_logging_enabled",
            "encryption_at_rest",
            "data_minimization",
        ]

        for requirement in privacy_requirements:
            if requirement not in data_config:
                errors.append(f"Missing privacy requirement: {requirement}")

        # Validate retention period (GDPR: right to erasure)
        if "data_retention_days" in data_config:
            retention = data_config["data_retention_days"]
            if not isinstance(retention, int) or retention < 1 or retention > 2555:  # ~7 years max
                errors.append("data_retention_days must be between 1 and 2555 days")

        return len(errors) == 0, errors
