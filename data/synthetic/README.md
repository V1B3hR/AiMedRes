# Synthetic De-identified Medical Research Data

This directory contains **synthetic, de-identified data** for testing and demonstration purposes.

## ⚠️ Important Notice

**ALL DATA IN THIS DIRECTORY IS SYNTHETIC AND CONTAINS NO REAL PHI**

- Data is computer-generated for research and testing
- No real patient information is included
- Safe for public distribution and version control
- Designed to test PHI detection and ML algorithms

## Data Files

### synthetic_alzheimer_patients.csv
Synthetic patient data for Alzheimer's research testing.

**Fields:**
- `patient_id`: Synthetic hash-based identifier
- `age`: Age in years (generalized to 5-year ranges)
- `gender`: 0=Female, 1=Male, 2=Other
- `mmse_score`: Mini-Mental State Examination score (0-30)
- `cdr_score`: Clinical Dementia Rating (0-3)
- `education_years`: Years of education (generalized)
- `diagnosis`: Cognitive status classification
- `assessment_year`: Year only (no specific dates)

**Example:**
```csv
patient_id,age,gender,mmse_score,cdr_score,education_years,diagnosis,assessment_year
SYNTH-A1B2C3,75,0,24,0.5,16,MCI,2023
SYNTH-D4E5F6,82,1,18,1.0,12,Mild_AD,2023
```

### synthetic_fhir_observations.json
Synthetic FHIR-format observations for integration testing.

**Structure:**
- Follows FHIR Observation resource format
- Uses synthetic identifiers
- No real PHI included
- Suitable for FHIR pipeline testing

## Data Generation

Data was generated using:
- Python's Faker library for demographic data
- Statistical distributions based on published research
- Random number generators with fixed seeds for reproducibility
- PHI scrubbing validation to ensure no real data

## Validation

All synthetic data has been validated to be PHI-free:
- ✅ No real names or identifiers
- ✅ No contact information
- ✅ No specific dates (years only)
- ✅ No addresses or locations
- ✅ Passed automated PHI detection scans

## Usage

This data can be used for:
- Unit and integration testing
- ML model development and validation
- Documentation and examples
- CI/CD pipeline testing
- Educational demonstrations

## Generating More Data

To generate additional synthetic data:

```python
from aimedres.utils.synthetic_data import SyntheticDataGenerator

generator = SyntheticDataGenerator(seed=42)
data = generator.generate_alzheimer_dataset(n_patients=1000)
```

## License

This synthetic data is released under the same license as AiMedRes (GPL-3.0) and can be freely used for research and development.

---

**Remember**: Always use synthetic, de-identified, or properly consented data for research. Never commit real PHI to version control systems.
