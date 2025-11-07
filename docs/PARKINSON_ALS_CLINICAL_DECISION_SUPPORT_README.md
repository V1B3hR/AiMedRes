# Clinical Decision Support System - Parkinson's & ALS

**Version**: 1.0.0 | **Last Updated**: November 2025

The `clinical_decision_support.py` module is the core of the AI-powered clinical decision support system. It provides a robust framework for assessing patient risk across multiple medical conditions and recommending appropriate interventions.

## Key Features

- **Multi-Condition Risk Assessment**: Extensible engine for evaluating patient risk for various diseases.
- **Personalized Risk Scoring**: Algorithms that calculate a risk score from 0.0 to 1.0 based on patient data.
- **Risk Level Classification**: Categorization of risk into MINIMAL, LOW, MEDIUM, or HIGH.
- **Intervention Recommendations**: A database of interventions tied to specific conditions and risk levels.
- **Explainable AI (XAI)**: Generation of human-readable explanations for risk assessments.
- **Comprehensive Clinical Summaries**: Aggregated reports for a holistic patient view.

## Supported Conditions

The system currently supports risk assessment for the following conditions:

-   **Alzheimer's Disease**
-   **Cardiovascular Disease**
-   **Diabetes**
-   **Stroke**
-   **Parkinson's Disease**
-   **Amyotrophic Lateral Sclerosis (ALS)**

## Technical Architecture

### `RiskStratificationEngine`

This class is responsible for the core logic of risk assessment.

-   `assess_risk()`: The main entry point for assessing risk for a single condition.
-   `_calculate_*_risk()`: A series of private methods, one for each supported condition, that implement the risk calculation logic based on clinical risk factors.
-   `_determine_risk_level()`: Maps a numerical risk score to a categorical risk level.
-   `_recommend_interventions()`: Suggests clinical interventions based on the risk assessment.
-   `_calculate_confidence()`: Calculates a confidence score based on the completeness of the provided patient data.
-   `_generate_explanation()`: Provides a textual explanation of the factors contributing to the risk score.

### `ClinicalDecisionSupportSystem`

This class orchestrates the overall process, managing assessments for multiple conditions.

-   `comprehensive_assessment()`: Performs risk assessments for a list of conditions.
-   `generate_clinical_summary()`: Creates a high-level summary of the patient's health status, including an overall risk score and a prioritized list of interventions.

## Usage Example

```python
from clinical_decision_support import ClinicalDecisionSupportSystem

# Configuration for the CDSS
config = { ... }

# Initialize the system
cdss = ClinicalDecisionSupportSystem(config)

# Example patient data
patient_data = {
    'patient_id': 'PATIENT_001',
    'Age': 68,
    'M/F': 1, # Male
    'UPDRS_motor_score': 25,
    'family_history_parkinsons': True,
    'ALSFRS_R_score': 42,
    'fvc_percent': 78,
    'family_history_als': False
    # ... other clinical data
}

# Perform a comprehensive assessment
assessments = cdss.comprehensive_assessment(patient_data)

# Generate a clinical summary
summary = cdss.generate_clinical_summary(assessments)

print(summary)
