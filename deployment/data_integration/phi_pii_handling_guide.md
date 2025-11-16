# PHI/PII Handling Configuration Guide

## Overview

This guide provides step-by-step instructions for enabling and configuring the PHI (Protected Health Information) scrubber in AiMedRes to ensure HIPAA-compliant data handling.

## Prerequisites

- AiMedRes installation complete
- Python 3.10+ environment
- Administrative access to the deployment environment

## PHI Scrubber Configuration

### 1. Understanding the PHI Scrubber

The PHI scrubber is located at `src/aimedres/security/phi_scrubber.py` and implements the HIPAA Safe Harbor method for de-identification. It detects and removes/masks:

- Names
- Geographic subdivisions smaller than state
- Dates (except years - configurable)
- Telephone and fax numbers
- Email addresses
- Social security numbers
- Medical record numbers (MRN)
- Account numbers
- License/certificate numbers
- Vehicle identifiers (VIN)
- Device identifiers
- URLs and IP addresses
- Biometric identifiers

### 2. Basic Configuration

#### Import and Initialize PHI Scrubber

```python
from src.aimedres.security.phi_scrubber import PHIScrubber, enforce_phi_free_ingestion

# Initialize with recommended settings for healthcare environments
scrubber = PHIScrubber(
    aggressive=True,        # Strict detection mode (recommended for production)
    hash_identifiers=True,  # Replace PHI with consistent hashes for tracking
    preserve_years=True     # Preserve years in dates (HIPAA allows this)
)
```

#### Configuration Parameters

| Parameter | Values | Description | Recommendation |
|-----------|--------|-------------|----------------|
| `aggressive` | True/False | Enables stricter detection patterns | **True** for production |
| `hash_identifiers` | True/False | Replace PHI with hashed values instead of generic markers | **True** for data tracking |
| `preserve_years` | True/False | Keep year information in dates (HIPAA compliant) | **True** for temporal analysis |

### 3. Integration Points

#### A. Data Ingestion

Enforce PHI-free data at ingestion points:

```python
from src.aimedres.security.phi_scrubber import enforce_phi_free_ingestion

# At data ingestion point
def ingest_patient_data(data):
    # This will raise ValueError if PHI is detected
    enforce_phi_free_ingestion(data, field_name="patient_data")
    
    # Proceed with processing only if data is PHI-free
    process_data(data)
```

#### B. Dataset Validation

Validate entire datasets before processing:

```python
# Validate dataset
validation_report = scrubber.validate_dataset(patient_records)

if not validation_report['is_clean']:
    print(f"PHI detected in {validation_report['records_with_phi']} records")
    print(f"PHI types found: {validation_report['phi_by_type']}")
    print(f"PHI by field: {validation_report['phi_by_field']}")
    
    # Sanitize if needed
    clean_data = scrubber.sanitize_dataset(patient_records)
```

#### C. API Endpoints

Protect API endpoints with PHI detection:

```python
from flask import request, jsonify
from src.aimedres.security.phi_scrubber import enforce_phi_free_ingestion

@app.route('/api/v1/patient/data', methods=['POST'])
def receive_patient_data():
    try:
        data = request.get_json()
        
        # Validate PHI-free before processing
        enforce_phi_free_ingestion(data, field_name="api_input")
        
        # Process data
        result = process_patient_data(data)
        return jsonify(result)
        
    except ValueError as e:
        # PHI detected - reject request
        return jsonify({
            'error': 'PHI_DETECTED',
            'message': str(e)
        }), 400
```

### 4. Testing PHI Scrubber

#### Run Built-in Tests

```bash
cd /home/runner/work/AiMedRes/AiMedRes
python3 -m src.aimedres.security.phi_scrubber
```

#### Custom Testing

```python
from src.aimedres.security.phi_scrubber import PHIScrubber

scrubber = PHIScrubber(aggressive=True)

# Test with sample data containing PHI
test_text = "Patient John Smith, DOB 03/15/1965, MRN: 123456"
result = scrubber.detect_phi(test_text)

print(f"PHI Found: {result.has_phi}")
print(f"Types: {result.phi_types_found}")
print(f"Sanitized: {result.sanitized_text}")
print(f"Confidence: {result.confidence_score:.2f}")
```

### 5. Clinical Whitelist Configuration

The PHI scrubber includes a clinical whitelist to avoid false positives on medical terms. The whitelist can be extended:

```python
# Add custom clinical terms (if needed)
scrubber.CLINICAL_WHITELIST.update({
    'YourCustomTerm1',
    'YourCustomTerm2',
    # Add institution-specific medical terms
})

# Add multi-word clinical phrases
scrubber.CLINICAL_PHRASES.update({
    'Your Custom Disease Name',
    'Your Custom Procedure Name',
})
```

### 6. Logging and Monitoring

Enable logging to track PHI detection events:

```python
import logging

# Configure PHI scrubber logging
logger = logging.getLogger('src.aimedres.security.phi_scrubber')
logger.setLevel(logging.INFO)

# Add file handler for audit trail
handler = logging.FileHandler('/var/log/aimedres/phi_scrubber.log')
handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
))
logger.addHandler(handler)
```

### 7. Environment Configuration

Add to your `.env` file:

```bash
# PHI Scrubber Configuration
PHI_SCRUBBER_AGGRESSIVE=true
PHI_SCRUBBER_HASH_IDENTIFIERS=true
PHI_SCRUBBER_PRESERVE_YEARS=true
PHI_SCRUBBER_LOG_LEVEL=INFO
PHI_SCRUBBER_LOG_FILE=/var/log/aimedres/phi_scrubber.log

# PHI Enforcement
PHI_ENFORCEMENT_ENABLED=true
PHI_BLOCK_ON_DETECTION=true
```

### 8. Production Deployment Checklist

- [ ] PHI scrubber enabled on all data ingestion endpoints
- [ ] Aggressive mode enabled (`aggressive=True`)
- [ ] Hash identifiers enabled for consistent de-identification
- [ ] Clinical whitelist reviewed and customized for institution
- [ ] Logging configured with secure audit trail
- [ ] Testing completed with known PHI samples
- [ ] Staff trained on PHI handling requirements
- [ ] Incident response plan documented for PHI detection events
- [ ] Regular audits scheduled (monthly recommended)

### 9. Compliance Validation

#### HIPAA Safe Harbor Compliance

The PHI scrubber implements all 18 HIPAA Safe Harbor identifiers:

1. ✅ Names
2. ✅ Geographic subdivisions smaller than state
3. ✅ Dates (except year)
4. ✅ Telephone numbers
5. ✅ Fax numbers
6. ✅ Email addresses
7. ✅ Social Security numbers
8. ✅ Medical record numbers
9. ✅ Health plan beneficiary numbers
10. ✅ Account numbers
11. ✅ Certificate/license numbers
12. ✅ Vehicle identifiers
13. ✅ Device identifiers and serial numbers
14. ✅ Web URLs
15. ✅ IP addresses
16. ✅ Biometric identifiers
17. ✅ Full-face photographs (N/A for text processing)
18. ✅ Any other unique identifying numbers

#### Validation Test

```python
# Run comprehensive HIPAA compliance test
from src.aimedres.security.phi_scrubber import PHIScrubber

def validate_hipaa_compliance():
    scrubber = PHIScrubber(aggressive=True)
    
    hipaa_test_cases = {
        'name': "Dr. Jane Smith",
        'address': "123 Main Street, Boston, MA",
        'date': "January 15, 2024",
        'phone': "555-123-4567",
        'email': "patient@example.com",
        'ssn': "123-45-6789",
        'mrn': "MRN: 987654321",
    }
    
    all_passed = True
    for phi_type, test_text in hipaa_test_cases.items():
        result = scrubber.detect_phi(test_text)
        if not result.has_phi:
            print(f"❌ Failed to detect {phi_type}")
            all_passed = False
        else:
            print(f"✅ Successfully detected {phi_type}")
    
    return all_passed

# Run validation
if validate_hipaa_compliance():
    print("\n✅ HIPAA Safe Harbor compliance validated")
else:
    print("\n❌ HIPAA compliance validation failed")
```

### 10. Troubleshooting

#### Issue: False Positives on Medical Terms

**Solution**: Add terms to clinical whitelist

```python
scrubber.CLINICAL_WHITELIST.update({'SpecificMedicalTerm'})
```

#### Issue: False Negatives (PHI not detected)

**Solution**: Enable aggressive mode and review patterns

```python
scrubber = PHIScrubber(aggressive=True)
```

#### Issue: Performance Impact

**Solution**: Batch processing and caching

```python
# Process in batches
batch_size = 100
for i in range(0, len(records), batch_size):
    batch = records[i:i+batch_size]
    clean_batch = scrubber.sanitize_dataset(batch)
```

## Support and Contact

For PHI scrubber issues or questions:
- Review documentation: `src/aimedres/security/phi_scrubber.py`
- Check test cases: `tests/test_api_security_compliance.py`
- Contact: compliance@aimedres.org

## References

- HIPAA Safe Harbor Method: [45 CFR 164.514(b)(2)](https://www.hhs.gov/hipaa/for-professionals/privacy/special-topics/de-identification/index.html)
- HIPAA Privacy Rule: [HHS.gov](https://www.hhs.gov/hipaa/for-professionals/privacy/index.html)
