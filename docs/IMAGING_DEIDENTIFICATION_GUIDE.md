# Medical Imaging De-identification Guide

**Version**: 1.0.0 | **Last Updated**: November 2025

## Overview

This guide provides comprehensive procedures for de-identifying medical imaging data in compliance with HIPAA Safe Harbor guidelines and medical imaging best practices. The AiMedRes imaging pipeline includes automated de-identification tools to protect patient privacy while preserving data utility for research and analysis.

## HIPAA Safe Harbor Compliance

Our de-identification process addresses all 18 categories of Protected Health Information (PHI) as defined in 45 CFR 164.514(b)(2):

### 1. Names
- **PatientName**: Replaced with "Anonymous"
- **ReferringPhysicianName**: Replaced with "Anonymous MD"
- **OperatorsName**: Replaced with "Anonymous"

### 2. Geographic Subdivisions Smaller Than State
- **InstitutionName**: Replaced with "Anonymous Institution"
- **InstitutionAddress**: Removed
- **StationName**: Replaced with "Anonymous"

### 3. Dates (except year)
- **StudyDate**: Shifted by consistent patient-specific offset
- **SeriesDate**: Shifted by consistent patient-specific offset
- **AcquisitionDate**: Shifted by consistent patient-specific offset
- **PatientBirthDate**: Removed or age-binned for patients >89 years

### 4. Telephone Numbers
- **PatientTelephoneNumbers**: Removed
- Pattern detection and removal from text fields

### 5. Fax Numbers
- Removed from DICOM headers and text fields

### 6. Electronic Mail Addresses
- Pattern detection and removal from text fields

### 7. Social Security Numbers
- Pattern detection and removal from text fields

### 8. Medical Record Numbers
- **PatientID**: Replaced with consistent anonymous ID
- **AccessionNumber**: Replaced with anonymous equivalent

### 9. Health Plan Beneficiary Numbers
- **OtherPatientIDs**: Removed

### 10. Account Numbers
- Removed from all fields

### 11. Certificate/License Numbers
- Removed from all fields

### 12. Vehicle Identifiers and Serial Numbers
- **DeviceSerialNumber**: Removed

### 13. Device Identifiers and Serial Numbers
- **DeviceSerialNumber**: Removed
- Equipment-specific identifiers anonymized

### 14. Web Universal Resource Locators (URLs)
- Pattern detection and removal from text fields

### 15. Internet Protocol (IP) Addresses
- Removed from network-related fields

### 16. Biometric Identifiers
- No direct removal (not typically in imaging metadata)

### 17. Full-Face Photographic Images
- N/A for medical imaging (structural scans only)

### 18. Other Unique Identifying Numbers
- All unique identifiers replaced with anonymous equivalents

## De-identification Process

### 1. DICOM Header Anonymization

The automated process removes or replaces PHI in DICOM headers:

```python
from mlops.imaging.utils.deidentify import MedicalImageDeidentifier

# Initialize de-identifier with encryption
deidentifier = MedicalImageDeidentifier(
    encryption_key="your_secure_key",
    mapping_file="secure_medical_workspace/id_mappings.encrypted"
)

# De-identify a DICOM file
result = deidentifier.deidentify_dicom_file(
    input_path="raw_dicom.dcm",
    output_path="anonymous_dicom.dcm",
    patient_id_override="PATIENT123"
)
```

### 2. Consistent ID Mapping

Patient identifiers are replaced with consistent anonymous IDs using cryptographic hashing:

- Original: `PATIENT123` → Anonymous: `ANON-A1B2C3D4`
- Consistency maintained across all files for the same patient
- Encrypted mapping storage for security

### 3. Date Shifting

Dates are shifted by a consistent, patient-specific offset:

- Maintains temporal relationships within patient data
- Prevents cross-linking between studies
- Shift range: ±365 days

### 4. Time Shifting

Times are shifted consistently with dates:

- Preserves acquisition timing relationships
- Prevents time-based patient identification

### 5. Text Field Anonymization

Automated pattern detection removes PHI from text fields:

- Phone numbers: `\b\d{3}-\d{3}-\d{4}\b`
- Social Security Numbers: `\b\d{3}-\d{2}-\d{4}\b`
- Email addresses: `\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b`
- Street addresses: Pattern-based detection

## Implementation Guidelines

### 1. Environment Setup

```bash
# Set encryption key environment variable
export DUETMIND_DEIDENTIFY_KEY="your_secure_encryption_key"

# Create secure workspace
mkdir -p secure_medical_workspace/{deidentified,audit}
chmod 700 secure_medical_workspace/
```

### 2. De-identification Workflow

```bash
# Setup imaging pipeline
make imaging-setup

# De-identify raw imaging data
make imaging-deidentify

# Validate de-identification
python -c "
from mlops.imaging.utils.deidentify import MedicalImageDeidentifier
import json

# Load and verify de-identification report
with open('secure_medical_workspace/audit/deidentification_report.txt', 'r') as f:
    report = f.read()
    print(report)
"
```

### 3. Quality Assurance

After de-identification, verify:

1. **PHI Removal**: No identifiable information remains
2. **Data Integrity**: Medical information preserved
3. **Consistency**: Same patient has same anonymous ID
4. **Temporal Relationships**: Date/time relationships maintained

### 4. Audit Trail

All de-identification activities are logged:

```
secure_medical_workspace/audit/
├── deidentification.log          # Process audit log
├── deidentification_report.txt   # Summary report
└── access.log                    # Data access log
```

## Technical Implementation

### Encryption

- **Algorithm**: AES-256-GCM for ID mappings
- **Key Derivation**: PBKDF2 with 100,000 iterations
- **Salt**: Installation-specific salt for key generation

### ID Generation

```python
def generate_anonymous_id(original_id: str, id_type: str = "patient") -> str:
    """Generate consistent anonymous ID using cryptographic hash."""
    hash_input = f"{original_id}:{id_type}:duetmind_salt".encode()
    hash_digest = hashlib.sha256(hash_input).hexdigest()[:8]
    return f"ANON-{hash_digest.upper()}"
```

### Date Shifting

```python
def calculate_date_shift(patient_id: str) -> int:
    """Calculate consistent date shift for patient."""
    hash_input = f"{patient_id}:date_shift:duetmind".encode()
    hash_value = int(hashlib.md5(hash_input).hexdigest()[:8], 16)
    return hash_value % 730 - 365  # ±365 days
```

## Compliance Validation

### Internal Validation

1. **Automated Checks**: Verify no PHI patterns remain
2. **Cross-Reference**: Ensure ID consistency
3. **Technical Validation**: Confirm data integrity

### External Validation

1. **Expert Review**: Manual verification by privacy experts
2. **Sample Audits**: Random sampling for quality control
3. **Penetration Testing**: Attempt re-identification

## Data Retention and Disposal

### Retention Policy

- **De-identified Data**: Retained per research requirements
- **ID Mappings**: Encrypted storage with controlled access
- **Audit Logs**: 7-year retention minimum

### Secure Disposal

- **Magnetic Media**: Multi-pass overwriting
- **Solid State**: Cryptographic erasure
- **Documentation**: Disposal certificates required

## Risk Assessment

### Residual Risk Factors

1. **Indirect Identification**: Rare disease patterns
2. **Technical Artifacts**: Equipment-specific signatures
3. **Temporal Patterns**: Unusual acquisition sequences

### Mitigation Strategies

1. **Additional Anonymization**: For rare conditions
2. **Metadata Scrubbing**: Remove technical details
3. **Temporal Jittering**: Add random noise to timestamps

## Integration with Research Workflows

### BIDS Compliance

De-identified data maintains BIDS structure:

```
bids_dataset/
├── dataset_description.json
├── participants.tsv              # Anonymous participant IDs
├── sub-ANON-A1B2C3D4/
│   ├── anat/
│   │   ├── sub-ANON-A1B2C3D4_T1w.nii.gz
│   │   └── sub-ANON-A1B2C3D4_T1w.json
│   └── func/
│       ├── sub-ANON-A1B2C3D4_task-rest_bold.nii.gz
│       └── sub-ANON-A1B2C3D4_task-rest_bold.json
```

### MLOps Integration

```python
# Track de-identification in MLflow
import mlflow

with mlflow.start_run():
    mlflow.log_param("deidentification_method", "HIPAA_Safe_Harbor")
    mlflow.log_param("phi_categories_addressed", 18)
    mlflow.log_artifact("secure_medical_workspace/audit/deidentification_report.txt")
```

## Regulatory Considerations

### FDA Requirements

- **Device Software**: De-identification as software feature
- **Clinical Trials**: Subject privacy protection
- **Post-Market Surveillance**: Anonymous adverse event reporting

### International Compliance

- **GDPR**: Right to erasure considerations
- **Local Regulations**: Country-specific requirements
- **IRB/Ethics**: Institutional review board approval

## Troubleshooting

### Common Issues

1. **Incomplete PHI Removal**: Check custom text fields
2. **ID Inconsistency**: Verify mapping file integrity
3. **Date/Time Errors**: Validate shift calculations

### Support Resources

- **Documentation**: Comprehensive guides and API docs
- **Validation Tools**: Automated PHI detection scripts
- **Expert Consultation**: Privacy and compliance expertise

## Conclusion

The AiMedRes imaging de-identification system provides comprehensive, automated protection of patient privacy while preserving data utility for medical research. By following HIPAA Safe Harbor guidelines and implementing technical safeguards, researchers can confidently use medical imaging data for advancing healthcare outcomes.

For additional support or questions about de-identification procedures, consult the technical documentation or contact the DuetMind support team.