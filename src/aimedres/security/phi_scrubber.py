#!/usr/bin/env python3
"""
PHI De-identification and Enforcement Module

This module provides comprehensive PHI (Protected Health Information) detection,
de-identification, and enforcement for the AiMedRes platform.

Implements HIPAA Safe Harbor method and additional safeguards.

P0-3 Requirement: PHI de-identification & ingestion enforcement
"""

import re
import hashlib
import logging
from typing import Dict, List, Set, Tuple, Any, Optional
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class PHIDetectionResult:
    """Results from PHI detection scan"""
    has_phi: bool
    phi_types_found: Set[str]
    phi_locations: List[Tuple[str, int, int]]  # (type, start, end)
    original_text: str
    sanitized_text: str
    confidence_score: float


class PHIScrubber:
    """
    Comprehensive PHI detection and de-identification system.
    
    Implements HIPAA Safe Harbor method requiring removal/generalization of:
    1. Names
    2. Geographic subdivisions smaller than state
    3. Dates (except year)
    4. Telephone numbers
    5. Fax numbers
    6. Email addresses
    7. Social security numbers
    8. Medical record numbers
    9. Health plan beneficiary numbers
    10. Account numbers
    11. Certificate/license numbers
    12. Vehicle identifiers
    13. Device identifiers
    14. Web URLs
    15. IP addresses
    16. Biometric identifiers
    17. Full-face photos (N/A for text)
    18. Other unique identifying numbers
    """
    
    # HIPAA Safe Harbor PHI patterns
    PHI_PATTERNS = {
        # Names - pattern matches capitalized words potentially forming names
        'name': re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b'),
        
        # Geographic - addresses, cities with street numbers, zip codes
        'address': re.compile(r'\b\d{1,5}\s+[A-Z][a-zA-Z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Lane|Ln|Boulevard|Blvd|Drive|Dr|Court|Ct|Way)\b', re.IGNORECASE),
        'zip_code': re.compile(r'\b\d{5}(?:-\d{4})?\b'),
        
        # Dates - various formats
        'date_iso': re.compile(r'\b\d{4}-\d{2}-\d{2}\b'),
        'date_us': re.compile(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'),
        'date_text': re.compile(r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b', re.IGNORECASE),
        
        # Contact information
        'phone': re.compile(r'\b(?:\+?\d{1,3}[\s\-\.]?)?(?:\(?\d{3}\)?[\s\-\.]?)?\d{3}[\s\-\.]?\d{4}\b'),
        'fax': re.compile(r'\b(?:fax|FAX)[:\s]*(?:\+?\d{1,3}[\s\-\.]?)?(?:\(?\d{3}\)?[\s\-\.]?)?\d{3}[\s\-\.]?\d{4}\b', re.IGNORECASE),
        'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
        
        # Identification numbers
        'ssn': re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
        'mrn': re.compile(r'\b(?:MRN|Medical\s+Record\s+Number|Patient\s+ID)[:\s#]*\d{5,}\b', re.IGNORECASE),
        'account': re.compile(r'\b(?:Account|Acct)[:\s#]*\d{5,}\b', re.IGNORECASE),
        'license': re.compile(r'\b(?:License|Lic)[:\s#]*[A-Z0-9]{5,}\b', re.IGNORECASE),
        
        # Web and network identifiers
        'url': re.compile(r'\b(?:https?://|www\.)[A-Za-z0-9\-._~:/?#\[\]@!$&\'()*+,;=]+\b', re.IGNORECASE),
        'ipv4': re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b'),
        'ipv6': re.compile(r'\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b'),
        
        # Vehicle and device identifiers
        'vin': re.compile(r'\b[A-HJ-NPR-Z0-9]{17}\b'),
        'device_id': re.compile(r'\b(?:Device|Serial)[:\s#]*[A-Z0-9]{8,}\b', re.IGNORECASE),
    }
    
    # Medical and clinical terms that should NOT be redacted (whitelist)
    # These are common medical terms that might match the name pattern
    CLINICAL_WHITELIST = {
        'Cognitive', 'Memory', 'Score', 'Patient', 'Lifestyle', 'Symptoms',
        'Assessment', 'Diagnosis', 'Treatment', 'Therapy', 'Medication',
        'Disease', 'Disorder', 'Syndrome', 'Condition', 'Risk', 'Factor',
        'Biomarker', 'Imaging', 'Clinical', 'Medical', 'Health', 'Care',
        'Alzheimer', 'Parkinson', 'Diabetes', 'Cardiovascular', 'Stroke',
        'MMSE', 'CDR', 'APOE', 'MRI', 'CT', 'PET', 'Scan', 'Test', 'Result',
        'Protocol', 'Study', 'Trial', 'Research', 'Data', 'Analysis',
        'January', 'February', 'March', 'April', 'May', 'June', 'July',
        'August', 'September', 'October', 'November', 'December',
        'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
    }
    
    # Common medical multi-word phrases that should not be redacted
    CLINICAL_PHRASES = {
        'Alzheimer Disease', 'Alzheimer\'s Disease', 'Parkinson Disease',
        'Parkinson\'s Disease', 'Cognitive Decline', 'Memory Loss',
        'Mild Cognitive', 'Cognitive Impairment', 'Disease Progression',
        'Clinical Trial', 'Risk Factor', 'Clinical Assessment'
    }
    
    def __init__(self, 
                 aggressive: bool = True,
                 hash_identifiers: bool = True,
                 preserve_years: bool = True):
        """
        Initialize PHI scrubber.
        
        Args:
            aggressive: If True, uses stricter detection (may over-redact)
            hash_identifiers: If True, replace with consistent hashes instead of generic markers
            preserve_years: If True, preserves years in dates (HIPAA allows this)
        """
        self.aggressive = aggressive
        self.hash_identifiers = hash_identifiers
        self.preserve_years = preserve_years
        self.detection_count = 0
        
    def detect_phi(self, text: str) -> PHIDetectionResult:
        """
        Detect PHI in text and return detailed results.
        
        Args:
            text: Input text to scan for PHI
            
        Returns:
            PHIDetectionResult with detection details
        """
        phi_found = set()
        locations = []
        confidence_scores = []
        
        for phi_type, pattern in self.PHI_PATTERNS.items():
            matches = pattern.finditer(text)
            for match in matches:
                matched_text = match.group(0)
                
                # Apply whitelist for names
                if phi_type == 'name':
                    # Check if entire match is a known clinical phrase
                    if matched_text in self.CLINICAL_PHRASES:
                        continue
                    # Check if entire match or any word in the match is in whitelist
                    if matched_text in self.CLINICAL_WHITELIST:
                        continue
                    # Check if ALL words in a multi-word phrase are clinical terms
                    # This handles cases like "Alzheimer Disease" where both words are clinical
                    if ' ' in matched_text:
                        words = matched_text.split()
                        if all(word in self.CLINICAL_WHITELIST for word in words):
                            continue
                    # Skip short single words that are likely not names
                    if ' ' not in matched_text and not self.aggressive:
                        continue
                
                phi_found.add(phi_type)
                locations.append((phi_type, match.start(), match.end()))
                confidence_scores.append(self._calculate_confidence(phi_type, matched_text))
        
        sanitized = self.sanitize(text)
        
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 1.0
        
        return PHIDetectionResult(
            has_phi=bool(phi_found),
            phi_types_found=phi_found,
            phi_locations=locations,
            original_text=text,
            sanitized_text=sanitized,
            confidence_score=avg_confidence
        )
    
    def sanitize(self, text: str) -> str:
        """
        Remove/mask all PHI from text.
        
        Args:
            text: Input text containing potential PHI
            
        Returns:
            Sanitized text with PHI removed/masked
        """
        result = text
        
        # Apply patterns in order of specificity (most specific first)
        pattern_order = [
            'ssn', 'mrn', 'account', 'license', 'vin', 'device_id',
            'email', 'url', 'ipv4', 'ipv6',
            'fax', 'phone',
            'zip_code', 'address',
            'date_iso', 'date_us', 'date_text',
            'name'
        ]
        
        for phi_type in pattern_order:
            pattern = self.PHI_PATTERNS[phi_type]
            result = self._replace_pattern(result, phi_type, pattern)
        
        self.detection_count += 1
        return result
    
    def _replace_pattern(self, text: str, phi_type: str, pattern: re.Pattern) -> str:
        """Replace PHI matches with appropriate replacement."""
        def replacer(match):
            matched = match.group(0)
            
            # Special handling for names - check whitelist
            if phi_type == 'name':
                # Check if entire match is a known clinical phrase
                if matched in self.CLINICAL_PHRASES:
                    return matched
                # Check if entire match or any word in the match is in whitelist
                if matched in self.CLINICAL_WHITELIST:
                    return matched
                # Check if ALL words in a multi-word phrase are clinical terms
                # This handles cases like "Alzheimer Disease" where both words are clinical
                if ' ' in matched:
                    words = matched.split()
                    if all(word in self.CLINICAL_WHITELIST for word in words):
                        return matched
                if ' ' not in matched and not self.aggressive:
                    return matched
            
            # Special handling for dates - preserve year if configured
            if phi_type.startswith('date') and self.preserve_years:
                year_match = re.search(r'\b(19|20)\d{2}\b', matched)
                if year_match:
                    year = year_match.group(0)
                    return f'[DATE-{year}]'
            
            # Hash replacement for consistent de-identification
            if self.hash_identifiers:
                hash_val = hashlib.sha256(matched.encode()).hexdigest()[:8]
                return f'[{phi_type.upper()}-{hash_val}]'
            
            # Generic replacement
            return f'[{phi_type.upper()}]'
        
        return pattern.sub(replacer, text)
    
    def _calculate_confidence(self, phi_type: str, matched_text: str) -> float:
        """
        Calculate confidence score for PHI detection.
        
        Returns value between 0 and 1, where 1 is highest confidence.
        """
        # Base confidence by type
        base_confidence = {
            'ssn': 0.99,
            'email': 0.98,
            'phone': 0.95,
            'mrn': 0.95,
            'url': 0.97,
            'ipv4': 0.90,
            'ipv6': 0.95,
            'date_iso': 0.90,
            'date_us': 0.85,
            'zip_code': 0.85,
            'name': 0.70,  # Lower confidence due to false positives
            'address': 0.85,
        }
        
        return base_confidence.get(phi_type, 0.80)
    
    def validate_dataset(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate an entire dataset for PHI presence.
        
        Args:
            data: List of dictionaries representing dataset
            
        Returns:
            Validation report with PHI detection statistics
        """
        total_records = len(data)
        records_with_phi = 0
        phi_by_field = {}
        phi_by_type = {}
        
        for record in data:
            record_has_phi = False
            for field, value in record.items():
                if isinstance(value, str):
                    result = self.detect_phi(value)
                    if result.has_phi:
                        record_has_phi = True
                        
                        # Track PHI by field
                        if field not in phi_by_field:
                            phi_by_field[field] = 0
                        phi_by_field[field] += 1
                        
                        # Track PHI by type
                        for phi_type in result.phi_types_found:
                            if phi_type not in phi_by_type:
                                phi_by_type[phi_type] = 0
                            phi_by_type[phi_type] += 1
            
            if record_has_phi:
                records_with_phi += 1
        
        return {
            'total_records': total_records,
            'records_with_phi': records_with_phi,
            'phi_percentage': (records_with_phi / total_records * 100) if total_records > 0 else 0,
            'phi_by_field': phi_by_field,
            'phi_by_type': phi_by_type,
            'is_clean': records_with_phi == 0,
            'timestamp': datetime.now().isoformat()
        }
    
    def sanitize_dataset(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Sanitize an entire dataset, removing all PHI.
        
        Args:
            data: List of dictionaries representing dataset
            
        Returns:
            Sanitized dataset with PHI removed
        """
        sanitized_data = []
        
        for record in data:
            sanitized_record = {}
            for field, value in record.items():
                if isinstance(value, str):
                    sanitized_record[field] = self.sanitize(value)
                else:
                    sanitized_record[field] = value
            sanitized_data.append(sanitized_record)
        
        logger.info(f"Sanitized {len(sanitized_data)} records")
        return sanitized_data


def enforce_phi_free_ingestion(data: Any, field_name: str = "data") -> bool:
    """
    Enforce that ingested data is PHI-free.
    
    This function should be called at ingestion points to block PHI data.
    
    Args:
        data: Data to validate (string or list of dicts)
        field_name: Name of field for logging
        
    Returns:
        True if data is PHI-free, False if PHI detected
        
    Raises:
        ValueError: If PHI is detected and strict mode is enabled
    """
    scrubber = PHIScrubber(aggressive=True)
    
    if isinstance(data, str):
        result = scrubber.detect_phi(data)
        if result.has_phi:
            logger.error(f"PHI detected in {field_name}: {result.phi_types_found}")
            raise ValueError(
                f"PHI detected in {field_name}. Types found: {result.phi_types_found}. "
                "Please de-identify data before ingestion."
            )
    elif isinstance(data, list):
        report = scrubber.validate_dataset(data)
        if not report['is_clean']:
            logger.error(f"PHI detected in dataset {field_name}: {report}")
            raise ValueError(
                f"PHI detected in {report['records_with_phi']} of {report['total_records']} records. "
                f"Types found: {report['phi_by_type']}. Please de-identify data before ingestion."
            )
    
    return True


# Example usage and testing
if __name__ == "__main__":
    # Test cases
    test_texts = [
        "Patient John Smith, DOB 03/15/1965, MRN: 123456",
        "Contact: john.smith@example.com or call 555-123-4567",
        "Address: 123 Main Street, Springfield, IL 62701",
        "SSN: 123-45-6789, Account #987654",
        "Visit on January 15, 2024 for cognitive assessment",
        "Clean medical text with no PHI - MMSE score 28/30, CDR 0.5"
    ]
    
    scrubber = PHIScrubber(aggressive=True, hash_identifiers=True)
    
    print("PHI Scrubber Test Results:")
    print("=" * 80)
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n[Test {i}] Original: {text}")
        result = scrubber.detect_phi(text)
        print(f"PHI Found: {result.has_phi}")
        if result.has_phi:
            print(f"Types: {result.phi_types_found}")
            print(f"Sanitized: {result.sanitized_text}")
            print(f"Confidence: {result.confidence_score:.2f}")
        else:
            print("✓ No PHI detected")
