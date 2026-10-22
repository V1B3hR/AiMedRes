# AiMedRes: Intended Use, Indications for Use & Regulatory Classification

> **Document ID**: AIMEDRES-CLIN-IFU-001  
> **Version**: 1.0.0  
> **Effective Date**: October 2026  
> **Status**: Approved Institutional Research & Regulatory Specification  
> **Regulatory Basis**: FDA 21st Century Cures Act § 3060 / FD&C Act § 520(o)(1)(E) & EU MDR 2017/745 Annex VIII Rule 11

---

## 1. Executive Summary & Regulatory Purpose

The purpose of this document is to establish the formal **Intended Use (IU)**, **Indications for Use (IFU)**, and **Regulatory Classification** for the AiMedRes Clinical Decision Support (CDS) platform. 

In healthcare software development, premature marketing claims or ambiguous clinical scope can introduce patient safety hazards, regulatory enforcement actions (e.g., FDA Warning Letters, MDR non-compliance), and institutional liability. This specification establishes an uncompromised, legally and clinically defensible perimeter for AiMedRes as a **Non-Device Clinical Decision Support (CDS)** research software platform and prepares the foundation for prospective Software as a Medical Device (SaMD) clearance.

---

## 2. Regulatory Classification & Statutory Analysis

### 2.1 United States: FDA 21st Century Cures Act § 3060 Analysis

Under Section 3060 of the 21st Century Cures Act (which amended Section 520(o) of the Federal Food, Drug, and Cosmetic Act [FD&C Act], 21 U.S.C. § 360j(o)(1)(E)), software functions are **exempt from the definition of a medical device** if they meet all four of the following statutory criteria:

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│               FDA 21ST CENTURY CURES ACT § 3060 NON-DEVICE CDS CRITERIA                │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ Criterion 1: Not intended to acquire, process, or analyze medical images or signals    │
│              from in vitro diagnostic devices or signal acquisition systems.           │
│ STATUS: COMPLIANT (Tabular biomarker & derived metric analysis only; image stripping)   │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ Criterion 2: Intended for displaying, analyzing, or printing medical information       │
│              about a patient or other medical information.                             │
│ STATUS: COMPLIANT (Synthesizes tabular clinical scores, demographics, and lab assays)  │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ Criterion 3: Intended to support or provide recommendations to a health care           │
│              professional (HCP) about prevention, diagnosis, or treatment.             │
│ STATUS: COMPLIANT (Provides progression risk stratification and literature guidelines) │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ Criterion 4: Intended to enable the HCP to independently review the basis for the      │
│              recommendations so that the HCP does not rely primarily on any advice.    │
│ STATUS: COMPLIANT (Outputs feature attributions, SHAP scores, and 95% bootstrap CIs)   │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

#### Detailed Statutory Assessment:

1. **Criterion 1 (No Primary Acquisition/Processing)**:
   - AiMedRes does not interface with raw sensor transducers, RF coils, or scanner pulse sequences. 
   - All neuroimaging features utilized by the platform (e.g., in OASIS-2 or PPMI DaTscan analyses) consist of post-acquisition, quality-controlled volumetric or regional summary statistics (e.g., normalized whole brain volume `nWBV`, estimated total intracranial volume `eTIV`, striatal binding ratio `SBR`) extracted by certified external neuroimaging pipelines (e.g., FreeSurfer, Statistical Parametric Mapping).
   - In production service mode (`src/aimedres/service/`), the API consumes standardized structured tabular payloads (FHIR R4 `Observation` and `RiskAssessment` resources) rather than raw DICOM pixel buffers.

2. **Criterion 2 (Medical Information Display & Analysis)**:
   - The platform aggregates and harmonizes structured patient clinical assessments, such as Movement Disorder Society Unified Parkinson's Disease Rating Scale (MDS-UPDRS), Mini-Mental State Examination (MMSE), acoustic voice telemetry metrics, and demographic covariates.

3. **Criterion 3 (Support and Recommendations to HCPs)**:
   - The software outputs multi-variable risk scores, estimated disease progression trajectories, and differential considerations to assist certified clinical specialists. It does not issue definitive diagnoses or autonomous treatment orders.

4. **Criterion 4 (Independent Clinical Reviewability / Explainability)**:
   - AiMedRes explicitly satisfies Criterion 4 by surfacing:
     - The exact mathematical input features driving the model's estimate.
     - SHAP (Shapley Additive exPlanations) local feature contribution rankings.
     - Calibrated probabilities accompanied by 95% bootstrap confidence intervals.
     - Brier score reliability indicators.
     - Direct bibliographic citations to the validating scientific study (e.g., PPMI consortium benchmarks).
   - The clinician is provided with complete visibility into the rationale, enabling independent clinical assessment without primary reliance on software automation.

### 2.2 European Union: EU MDR (2017/745) Classification

Under the European Union Medical Device Regulation (EU MDR 2017/745), Annex VIII Rule 11 specifies:
- *"Software intended to provide information which is used to take decisions with diagnosis or therapeutic purposes is classified as class IIa..."*

**Current Regulatory Status**:
- **Research Use Only (RUO)**: For current investigational deployments, the platform is distributed exclusively for institutional research under Section 520(g) investigational device exemptions and EU MDR Article 65 clinical investigation guidelines.
- **SaMD Transition Roadmap**: For prospective commercial CE-mark clearance, AiMedRes is architected under **IEC 62304 Software Safety Class B** and **ISO 14971** risk management controls to fulfill MDR Class IIa conformity assessment requirements.

---

## 3. Formal Intended Use Statement

```text
"AiMedRes is a modular clinical decision-support (CDS) software platform intended 
to assist licensed healthcare professionals—specifically board-certified neurologists, 
movement disorder specialists, and clinical researchers—by aggregating, analyzing, 
and visualizing structured multivariable clinical data (including longitudinal motor 
symptom ratings, validated acoustic telemetry biomarkers, and post-processed 
structural neuroimaging volumetric indices).

The software provides risk stratification estimates, longitudinal progression 
trajectories, and evidence-based reference guidelines for adult patients undergoing 
clinical evaluation for neurodegenerative movement disorders and cognitive impairment.

AiMedRes is designed strictly as an adjunctive decision-support aid. It does not perform 
autonomous diagnostic determination, does not acquire or reconstruct raw medical 
images, and does not directly prescribe medication or therapeutic interventions. 
The software presents the full statistical rationale, input feature attributions, and 
confidence bounds for each output, enabling the supervising clinician to independently 
evaluate the basis of all analytical findings."
```

---

## 4. Formal Indications for Use (IFU)

### 4.1 Target Patient Population
AiMedRes is indicated for evaluation in:
- Adult human patients aged **30 years and older**.
- Patients presenting with signs, symptoms, or genetic risk factors associated with Parkinsonian syndromes (e.g., tremor, bradykinesia, rigidity, postural instability) or neurodegenerative cognitive decline.
- Patients enrolled in retrospective or prospective observational clinical research studies monitoring longitudinal motor and cognitive decline.

### 4.2 Target User Profile
The software is indicated for operation exclusively by:
- Board-certified Neurologists and Movement Disorder Specialists.
- Geriatricians and Behavioral Neurologists.
- Clinical Neuropsychologists and Clinical Research Fellows under specialist supervision.
- Clinical Trial Investigators conducting protocol-compliant biomedical studies.

*The software is NOT intended for direct-to-consumer, patient self-triage, or unmonitored home use.*

### 4.3 Target Clinical Environment
- Outpatient hospital neurology clinics and tertiary movement-disorder centers.
- Academic medical centers and clinical research institutes.
- Secure, access-controlled data analysis environments conforming to institutional cybersecurity baselines.

---

## 5. Absolute Contraindications & Prohibited Uses

The following applications are strictly contraindicated and prohibited:

```text
╔════════════════════════════════════════════════════════════════════════════════════════╗
║                              PROHIBITED CLINICAL USES                                  ║
╠════════════════════════════════════════════════════════════════════════════════════════╣
║ 1. EMERGENCY & CRITICAL CARE TRIAGE: Prohibited for acute stroke triage, acute         ║
║    head trauma (TBI emergency department disposition), or intracranial hemorrhage.     ║
║                                                                                        ║
║ 2. SOLE DIAGNOSTIC DETERMINATION: Prohibited as an autonomous diagnostic authority     ║
║    without direct clinical examination and independent physician verification.         ║
║                                                                                        ║
║ 3. AUTONOMOUS THERAPY SELECTION: Prohibited for automated closed-loop medication        ║
║    dispensing, infusion pump control, or automated DBS parameter programming.          ║
║                                                                                        ║
║ 4. PEDIATRIC POPULATIONS: Prohibited for pediatric patients under 18 years of age.     ║
║                                                                                        ║
║ 5. RAW IMAGE PRIMARY DIAGNOSTICS: Prohibited for interpreting raw MRI/CT slices         ║
║    without prior radiologist review and post-processing quality control.               ║
╚════════════════════════════════════════════════════════════════════════════════════════╝
```

---

## 6. Clinical Decision Support Transparency & Explainability Controls

To enforce FDA § 3060 Criterion 4 and ensure clinicians can independently review all recommendations, AiMedRes implements the following four architectural controls:

### Control 1: Transparent Input Provenance
Every risk assessment response displays the exact clinical covariates, timestamps, and measurement units ingested by the pipeline. Missing covariates are explicitly identified with the imputation strategy utilized (e.g., fold-internal median imputation).

### Control 2: Local Feature Attribution (SHAP)
For every classification or trajectory output, the model computes and presents local Shapley values:
- Positive contributors (features pushing the prediction toward accelerated progression).
- Negative contributors (protective features or baseline stabilizations).
- Visual waterfall summary plots detailing the top 5 influential biomarkers.

### Control 3: Calibration & Uncertainty Bounds
Point estimates are strictly prohibited from standing alone:
- All probability outputs are calibrated and accompanied by **95% bootstrap confidence intervals** (e.g., $P(\text{Progression}) = 0.76 \text{ [95\% CI: 0.74 – 0.78]}$).
- Brier reliability metrics and historical cohort sample sizes are disclosed to the clinician.

### Control 4: Clinical Practice Guideline Linkage
Every risk alert references verified medical literature and consensus clinical criteria (e.g., Movement Disorder Society Clinical Diagnostic Criteria for Parkinson's Disease, Postuma et al., 2015; National Institute on Aging-Alzheimer's Association diagnostic guidelines).

---

## 7. Document Revision & Signoff History

| Version | Date | Author / Role | Summary of Changes |
| :--- | :--- | :--- | :--- |
| `1.0.0` | October 2026 | AiMedRes Clinical & Regulatory Working Group | Initial formal specification of Intended Use, IFU, and FDA 21st Century Cures Act § 3060 CDS compliance. |
