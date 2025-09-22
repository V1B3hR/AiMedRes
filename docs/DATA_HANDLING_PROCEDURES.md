# Data Handling Procedures

## Overview

This document outlines the comprehensive data handling procedures for DuetMind Adaptive, ensuring compliance with healthcare data protection regulations and maintaining the highest standards of data security and privacy.

## Data Classification

### Classification Levels

#### 1. Public Data
- **Description**: Non-sensitive information that can be freely shared
- **Examples**: System documentation, public API specifications
- **Handling**: No special restrictions
- **Retention**: Indefinite

#### 2. Internal Data  
- **Description**: Business information for internal use
- **Examples**: Aggregated statistics, anonymized metrics
- **Handling**: Access controls applied
- **Retention**: 7 years default

#### 3. Confidential Data
- **Description**: Sensitive business or anonymized medical data
- **Examples**: De-identified medical datasets, ML model parameters
- **Handling**: Encryption required, access logging
- **Retention**: 3-7 years based on type

#### 4. Restricted Data
- **Description**: Highly sensitive data requiring maximum protection
- **Examples**: PHI, PII, raw medical data
- **Handling**: Strong encryption, anonymization, strict access controls
- **Retention**: Per healthcare regulations (typically 7 years)

## Medical Data Lifecycle

### 1. Data Ingestion

#### Secure Loading Process
```python
# Load data through secure processor
processor = SecureMedicalDataProcessor(config)
dataset_id = processor.load_and_secure_dataset(
    dataset_source='kaggle',
    dataset_params={
        'dataset_name': 'medical-dataset',
        'file_path': 'data.csv'
    },
    user_id=user_id,
    purpose='training'
)
```

#### Data Validation
- **Format Validation**: Check data types and structure
- **Medical Validation**: Verify medical value ranges
- **Security Validation**: Scan for potential security issues
- **Privacy Validation**: Identify and flag PII

### 2. Data Processing

#### Anonymization Pipeline
1. **Direct Identifier Removal**
   - Remove names, SSNs, medical record numbers
   - Remove contact information (phone, email, address)
   - Remove provider identifiers

2. **Quasi-Identifier Processing**
   - Hash postal codes to geographic regions
   - Generalize dates to age ranges
   - Aggregate rare values

3. **Data Utility Preservation**
   - Maintain statistical properties
   - Preserve correlations for ML training
   - Keep medical validity intact

#### Example Anonymization
```python
def anonymize_patient_data(data):
    """Anonymize patient data while preserving ML utility."""
    
    # Remove direct identifiers
    anonymized = data.drop(columns=[
        'patient_id', 'name', 'ssn', 'phone', 'email'
    ], errors='ignore')
    
    # Hash quasi-identifiers
    if 'zip_code' in anonymized:
        anonymized['region_hash'] = anonymized['zip_code'].apply(
            lambda x: hashlib.sha256(str(x)[:3].encode()).hexdigest()[:8]
        )
        anonymized = anonymized.drop(columns=['zip_code'])
    
    # Generalize sensitive attributes
    if 'birth_date' in anonymized:
        anonymized['age_group'] = pd.cut(
            anonymized['age'], 
            bins=[0, 30, 50, 70, 100], 
            labels=['young', 'middle', 'senior', 'elderly']
        )
        anonymized = anonymized.drop(columns=['birth_date'])
    
    return anonymized
```

### 3. Data Storage

#### Storage Architecture
```
secure_medical_workspace/
├── training/           # Training data isolation
│   ├── dataset_001.encrypted
│   └── dataset_002.encrypted
├── inference/          # Inference data isolation  
│   ├── input_001.encrypted
│   └── results_001.encrypted
├── models/            # Encrypted model storage
│   ├── model_001.encrypted
│   └── model_002.encrypted
└── audit/             # Audit logs
    ├── access.log
    └── retention.log
```

#### Encryption Standards
- **Algorithm**: AES-256-GCM
- **Key Management**: PBKDF2 with 100,000 iterations
- **Salt**: Unique per installation
- **Key Rotation**: Annual or upon compromise

### 4. Data Access

#### Access Control Matrix
| Role | Public | Internal | Confidential | Restricted |
|------|--------|----------|--------------|------------|
| Admin | Read/Write | Read/Write | Read/Write | Read/Write |
| Researcher | Read | Read/Write | Read | None |
| Analyst | Read | Read | Read | None |
| Viewer | Read | Read | None | None |

#### Access Logging
Every data access is logged with:
- User ID and role
- Timestamp and duration
- Data type and identifier
- Purpose and legal basis
- IP address and location
- Success/failure status

```python
# Example access logging
privacy_manager.log_data_access(
    user_id='researcher_001',
    data_type='medical_dataset',
    action='read',
    data_id='dataset_12345',
    purpose='alzheimer_research',
    legal_basis='healthcare_research',
    ip_address='192.168.1.100'
)
```

## Data Retention Policies

### Retention Schedules

#### Medical Data
- **Raw Medical Data**: 7 years (HIPAA requirement)
- **Anonymized Research Data**: 10 years
- **Training Data**: 3 years post-model deployment
- **Inference Results**: 1 year

#### System Data  
- **API Logs**: 90 days
- **Security Logs**: 7 years
- **Audit Trails**: 7 years
- **Error Logs**: 1 year

#### Model Data
- **Production Models**: 3 years post-retirement
- **Experimental Models**: 1 year
- **Model Metrics**: 5 years
- **Training Metadata**: 3 years

### Automated Cleanup
```python
# Configure retention policies
retention_policy = DataRetentionPolicy(
    medical_data_retention_days=2555,      # 7 years
    training_data_retention_days=1095,     # 3 years
    api_logs_retention_days=90,
    audit_logs_retention_days=2555,
    anonymize_after_days=30,
    enable_automatic_deletion=True
)

# Start background cleanup
privacy_manager = PrivacyManager(config)
privacy_manager.start_background_cleanup()
```

## Privacy Protection Measures

### De-identification Techniques

#### Safe Harbor Method (HIPAA)
Remove/generalize these 18 identifiers:
1. Names
2. Geographic subdivisions smaller than state
3. Dates (except year) related to individual
4. Telephone numbers
5. Fax numbers
6. Email addresses
7. Social security numbers
8. Medical record numbers
9. Health plan beneficiary numbers
10. Account numbers
11. Certificate/license numbers
12. Vehicle identifiers and serial numbers
13. Device identifiers and serial numbers
14. Web URLs
15. IP addresses
16. Biometric identifiers
17. Full-face photos
18. Other unique identifying numbers/codes

#### Expert Determination
- Statistical disclosure control
- K-anonymity (k≥5)
- L-diversity for sensitive attributes
- T-closeness for distribution preservation

### Privacy-Preserving ML

#### Differential Privacy
```python
def add_differential_privacy(data, epsilon=1.0):
    """Add calibrated noise for differential privacy."""
    noise_scale = 1.0 / epsilon
    noise = np.random.laplace(0, noise_scale, data.shape)
    return data + noise
```

#### Federated Learning
- Train models without centralizing data
- Aggregate model updates, not raw data
- Secure multi-party computation
- Homomorphic encryption for computations

## Compliance Monitoring

### HIPAA Compliance

#### Administrative Safeguards ✓
- [x] Security Officer designation
- [x] Workforce training programs
- [x] Access management procedures
- [x] Incident response procedures

#### Physical Safeguards ✓
- [x] Facility access controls
- [x] Workstation use restrictions
- [x] Device and media controls
- [x] Secure data center operations

#### Technical Safeguards ✓
- [x] Access control systems
- [x] Audit controls and logging
- [x] Data integrity protections
- [x] Transmission security (TLS)

### GDPR Compliance

#### Data Subject Rights ✓
- [x] Right to access (Article 15)
- [x] Right to rectification (Article 16)
- [x] Right to erasure (Article 17)
- [x] Right to restrict processing (Article 18)
- [x] Right to data portability (Article 20)
- [x] Right to object (Article 21)

#### Data Protection Principles ✓
- [x] Lawfulness, fairness, transparency
- [x] Purpose limitation
- [x] Data minimization
- [x] Accuracy
- [x] Storage limitation
- [x] Integrity and confidentiality

## Data Incident Response

### Incident Classification

#### Level 1: Low Risk
- Single record exposure
- Internal access only
- No sensitive PII involved
- Response: Document and monitor

#### Level 2: Medium Risk
- Multiple records exposed
- Limited external access
- Some PII potentially involved
- Response: Investigate and contain

#### Level 3: High Risk
- Large-scale data exposure
- External unauthorized access
- PHI/sensitive PII involved
- Response: Immediate containment, notification

#### Level 4: Critical Risk
- Massive data breach
- Public exposure
- Sensitive medical data
- Response: Emergency response, regulatory notification

### Response Procedures

#### Immediate (0-1 hours)
1. Identify and contain the incident
2. Assess scope and severity
3. Notify security team
4. Begin evidence preservation

#### Short-term (1-24 hours)
1. Complete incident assessment
2. Notify affected stakeholders
3. Implement additional controls
4. Begin forensic investigation

#### Medium-term (1-7 days)
1. Complete investigation
2. Regulatory notifications (if required)
3. Affected individual notifications
4. Implement corrective measures

#### Long-term (1+ weeks)
1. Post-incident review
2. Policy and procedure updates
3. Additional training if needed
4. Monitoring and verification

## Quality Assurance

### Data Quality Metrics
- **Completeness**: Percentage of non-null values
- **Consistency**: Data format and value consistency
- **Accuracy**: Correctness of data values
- **Validity**: Adherence to defined constraints
- **Uniqueness**: Absence of duplicate records

### Privacy Quality Assessment
```python
def assess_privacy_quality(data):
    """Assess privacy protection quality."""
    
    quality_score = 0.0
    
    # Check for direct identifiers
    direct_identifiers = ['name', 'ssn', 'phone', 'email']
    identifier_found = any(col in data.columns for col in direct_identifiers)
    if not identifier_found:
        quality_score += 0.3
    
    # Check k-anonymity
    k_value = calculate_k_anonymity(data)
    if k_value >= 5:
        quality_score += 0.3
    
    # Check for quasi-identifier generalization
    quasi_id_generalized = check_quasi_identifier_treatment(data)
    if quasi_id_generalized:
        quality_score += 0.2
    
    # Check data utility preservation
    utility_score = assess_data_utility(data)
    quality_score += 0.2 * utility_score
    
    return quality_score
```

## Training and Awareness

### Required Training
- **All Staff**: Basic data protection principles
- **Developers**: Secure coding practices
- **Researchers**: Research data handling
- **Administrators**: System security management

### Training Schedule
- **Initial**: Within 30 days of role assignment
- **Annual**: Comprehensive refresher training
- **Ad-hoc**: When policies change or incidents occur
- **Specialized**: Role-specific advanced training

### Competency Assessment
- Pre and post-training assessments
- Practical exercises and simulations
- Ongoing monitoring and feedback
- Certification requirements for key roles

---

**Document Control**: This document is reviewed quarterly and updated as needed to reflect current best practices and regulatory requirements.