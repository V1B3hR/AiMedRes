# Institutional Review Board (IRB) / Ethics Committee Study Protocol

> **Protocol ID**: AIMEDRES-IRB-2026-001  
> **Protocol Title**: Retrospective and Prospective Observational Validation of a Machine Learning Clinical Decision-Support System for Multi-Cohort Neurodegenerative Disease Progression (The AiMedRes Study)  
> **Study Type**: Observational, Non-Interventional, Software Diagnostic Validation  
> **Regulatory Status**: Investigational / Non-Device Clinical Decision Support (FDA 21 CFR 812 Exempt / 45 CFR 46)  
> **Version**: 1.0.0 | **Date**: October 2026  

---

## 1. Protocol Summary & Study Synopsis

| Parameter | Specification |
| :--- | :--- |
| **Study Title** | Multi-Cohort Retrospective and Prospective Observational Validation of the AiMedRes Machine Learning Clinical Decision-Support System |
| **Short Title** | The AiMedRes Clinical Validation Study |
| **Protocol ID** | `AIMEDRES-IRB-2026-001` |
| **Study Phase** | Non-interventional Software Diagnostic Accuracy & Decision Support Study |
| **Primary Objective** | To prospectively and retrospectively evaluate the discriminative accuracy (ROC-AUC), precision-recall (PR-AUC), and calibration (Brier score) of participant-isolated machine learning models predicting motor and cognitive progression in neurodegenerative disorders. |
| **Study Population** | Adults $\ge 30$ years evaluated for Parkinsonian movement disorders or mild-to-moderate cognitive impairment across multi-center cohorts. |
| **Target Sample Size** | **Retrospective Phase**: $N \ge 6,000$ patient visits across PPMI ($N \approx 500$ patients), OASIS-2 ($N = 150$ patients, 373 visits), and UCI Parkinson's Telemonitoring ($N = 42$ patients, 5,875 recordings).<br>**Prospective Observational Phase**: $N = 100$ consecutive outpatient clinic evaluations. |
| **Intervention** | None. Purely observational. The software generates adjunctive risk assessments; all treatment decisions remain at the sole discretion of the treating physician. |
| **Primary Endpoints** | 1. 5-fold cross-validated ROC-AUC with 95% bootstrap confidence intervals.<br>2. Brier calibration loss on holdout participant clusters. |
| **Secondary Endpoints** | 1. Clinician agreement rate with SHAP feature contribution rankings.<br>2. Clinical decision turnaround time (minutes saved per complex case review).<br>3. Zero-leakage compliance (100% participant fold isolation). |
| **Study Duration** | 18 Months (6 months retrospective analysis + 12 months prospective observational validation). |

---

## 2. Background & Scientific Rationale

### 2.1 The Clinical Challenge
Neurodegenerative disorders such as Parkinson's disease (PD) and Alzheimer's disease (AD) exhibit marked inter-individual heterogeneity in progression velocity. Current clinical rating instruments—such as the Movement Disorder Society Unified Parkinson's Disease Rating Scale (MDS-UPDRS) and Mini-Mental State Examination (MMSE)—rely heavily on episodic clinic visits, creating significant diagnostic latency and high variance in symptom tracking.

### 2.2 Shortcomings of Published AI/ML Models
A systematic review of published medical machine learning reveals extensive methodological vulnerabilities:
1. **Participant Leakage**: Models evaluate longitudinal repeated visits without grouping on patient ID, artificially inflating test metrics by memorizing patient phenotypes.
2. **Circular Target Definitions**: Models include clinical rating scores (e.g., Clinical Dementia Rating `CDR` or baseline UPDRS) in the feature matrix when predicting the very condition those scores define.
3. **Preprocessing Contamination**: Scaling, imputation, and feature selection are routinely fitted across entire datasets prior to cross-validation.

### 2.3 The AiMedRes Solution
AiMedRes addresses these methodological flaws by enforcing:
- Participant-level isolation (`StratifiedGroupKFold`) with mathematically guaranteed zero patient overlap between train and validation splits.
- Automated dropping of circular target derivatives.
- Pipeline encapsulation (`sklearn.pipeline.Pipeline`) ensuring all scaling and median imputation occur strictly inside cross-validation folds.
- Calibrated risk scoring with 95% bootstrap confidence intervals.

---

## 3. Study Objectives & Hypotheses

### 3.1 Primary Objective
To demonstrate that participant-isolated machine learning pipelines achieve non-inferior and calibrated discrimination predicting meaningful disease progression milestones:
- **Parkinson's Motor Progression**: $\text{ROC-AUC} \ge 0.75$ and $\text{Brier Score} \le 0.20$ on holdout patients.
- **Dementia Diagnostic Stratification**: $\text{ROC-AUC} \ge 0.80$ and $\text{Brier Score} \le 0.18$ on holdout subjects.

### 3.2 Secondary Objectives
1. **Explainability Concordance**: Demonstrate that top-ranked SHAP features align with established Movement Disorder Society (MDS) diagnostic criteria in $\ge 85\%$ of reviewed cases.
2. **Clinical Utility**: Measure clinician confidence and case preparation efficiency when presented with calibrated risk profiles and feature waterfall plots versus standard EHR summary sheets alone.

---

## 4. Patient Cohorts & Eligibility Criteria

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                       STUDY COHORT ELIGIBILITY OVERVIEW                     │
├─────────────────────────────────────────────────────────────────────────────┤
│ RETROSPECTIVE COHORTS:                                                      │
│ 1. PPMI (Parkinson's Progression Markers Initiative)                        │
│    - Striatal Binding Ratios (DaTscan SBR) + MDS-UPDRS Part III motor       │
│ 2. OASIS-2 Longitudinal Dementia MRI Cohort                                │
│    - 150 subjects, 373 scan sessions, normalized whole brain volume (nWBV)  │
│ 3. UCI / Oxford Parkinson's Telemonitoring Study                            │
│    - 42 patients, 5,875 vocal telemetry sessions, 16 acoustic dysphonia     │
├─────────────────────────────────────────────────────────────────────────────┤
│ PROSPECTIVE OBSERVATIONAL COHORT:                                           │
│ - 100 consecutive adult outpatients presenting to the participating         │
│   Neurology / Movement Disorders Clinic.                                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.1 Inclusion Criteria
1. Patient age $\ge 30$ years at the time of evaluation.
2. Documented clinical diagnosis or active clinical evaluation for:
   - Idiopathic Parkinson's Disease (conforming to UK Brain Bank or MDS criteria).
   - Mild Cognitive Impairment (MCI) or Early Dementia.
3. Presence of baseline clinical evaluation plus at least one documented longitudinal follow-up assessment ($\ge 6$ months follow-up).
4. Availability of standardized clinical features (MDS-UPDRS, cognitive battery, or quantitative neuroimaging metrics).

### 4.2 Exclusion Criteria
1. Age $< 30$ years.
2. Atypical or secondary Parkinsonian syndromes (e.g., Multiple System Atrophy [MSA], Progressive Supranuclear Palsy [PSP], drug-induced parkinsonism, normal pressure hydrocephalus).
3. Concurrent severe acute neurological conditions (e.g., acute ischemic stroke, intracranial hemorrhage, acute central nervous system infection).
4. Incomplete baseline demographic records where age or sex cannot be determined.

---

## 5. Statistical Analysis & Leakage Safeguards

### 5.1 Cross-Validation Strategy
All validation will be executed using `StratifiedGroupKFold` ($k = 5$). 
- **Grouping Variable**: `Subject ID` / `PATNO` / `patient_id`.
- **Mathematical Guarantee**:
  $$\text{Subjects}(\text{Train}_k) \cap \text{Subjects}(\text{Val}_k) = \emptyset \quad \forall k \in \{1, \dots, 5\}$$
- All repeat visits for an individual patient remain strictly confined to either the training split or the evaluation split.

### 5.2 Preprocessing Isolation
All data preprocessing transformations—including missing value imputation (`SimpleImputer(strategy="median")`) and variance scaling (`StandardScaler()`)—must be fitted exclusively on $\text{Train}_k$ and subsequently applied to $\text{Val}_k$. Fitting any preprocessor on the unpartitioned dataset is strictly prohibited.

### 5.3 Bootstrap Confidence Interval Computation
All performance metrics (ROC-AUC, PR-AUC, Accuracy, Sensitivity, Specificity) will be reported with **95% two-sided confidence intervals** derived via $B = 1,000$ stratified bootstrap iterations on the out-of-fold predictions:
$$\text{CI}_{95\%} = \left[ \text{Percentile}_{2.5\%}(\hat{\theta}^*), \, \text{Percentile}_{97.5\%}(\hat{\theta}^*) \right]$$

---

## 6. Ethical Conduct, Informed Consent & Privacy

### 6.1 Waiver of Informed Consent for Retrospective Data
For the retrospective analyses involving PPMI, OASIS-2, and UCI open datasets, this protocol requests a **Waiver of Informed Consent** under 45 CFR 46.116(f) based on the following criteria:
1. The research involves no more than minimal risk to participants.
2. The research utilizes secondary, de-identified datasets where all direct identifiers have been stripped prior to investigator access.
3. The research cannot practicably be carried out without the waiver, as participants are drawn from completed historical multi-site consortia.
4. The waiver will not adversely affect the rights and welfare of the subjects.

### 6.2 Prospective Observational Cohort Consent
For the prospective cohort ($N = 100$), written informed consent will be obtained from all participants or their legally authorized representatives (LAR) prior to study enrollment. Participants will be explicitly informed that:
- AiMedRes is an investigational decision-support tool.
- All clinical care, diagnostic determinations, and medication management remain under the direct control of their attending physician.

### 6.3 Data De-Identification & Cybersecurity
- **HIPAA Compliance**: All patient records will be de-identified in accordance with the HIPAA Safe Harbor standard (45 CFR § 164.514(b)(2)). Direct identifiers (names, dates of birth, social security numbers, medical record numbers) are replaced with synthetic cryptographic tokens.
- **Relational Credential Storage**: All researcher access to the study platform requires cryptographic authentication backed by `AuthDatabaseStore` with PBKDF2 hashing, salted passwords, and instant API token revocation capabilities.
- **Quarantine Controls**: Experimental cognitive agent components are quarantined in `src/aimedres/simulation/` and disabled during clinical study execution via fail-closed runtime guards (`assert_simulation_allowed()`).

---

## 7. Data Safety Monitoring & Adverse Event Reporting

Because this study is observational and non-interventional, software outputs are presented to the clinician as adjunctive reference information only. 

### 7.1 Software Discrepancy Reporting
A formal Software Discrepancy Log will be maintained by the Study Coordinator. Any software failure, runtime crash, latency breach (> 2.0 seconds), or blatant contradiction with clinical consensus will be logged within 24 hours and investigated by the software engineering team.

### 7.2 Safety Signal Definition
If an attending clinician observes a discrepancy between model risk predictions and verified clinical trajectory, the clinician's primary diagnostic assessment takes immediate precedence, and the incident will be reviewed during quarterly Data Safety Monitoring Committee meetings.

---

## 8. Protocol Approval & Investigator Signatures

By signing below, the Principal Investigator and Study Biostatistician agree to conduct this study in accordance with the International Council for Harmonisation (ICH) Good Clinical Practice (GCP) guidelines and applicable institutional and regulatory requirements.

```text
Principal Investigator (MD, Neurology):  ___________________________  Date: __________

Study Biostatistician (PhD):             ___________________________  Date: __________

Clinical AI Systems Lead:               ___________________________  Date: __________
```
