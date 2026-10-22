# Clinician-in-the-Loop Usability & Human Factors Engineering Protocol

> **Document ID**: AIMEDRES-CLIN-HFE-001  
> **Version**: 1.0.0  
> **Effective Date**: October 2026  
> **Standards Alignment**: FDA Guidance: *Applying Human Factors and Usability Engineering to Medical Devices* (2016) | ANSI/AAMI HE75:2009 | IEC 62366-1:2015  
> **Target Audience**: Clinical Usability Engineers, Human Factors Specialists, Neurologists, Institutional Review Boards  

---

## 1. Executive Summary & Purpose

The purpose of this document is to establish the formal **Human Factors Engineering (HFE)** and **Clinician-in-the-Loop Usability Protocol** for the AiMedRes Clinical Decision Support (CDS) platform. 

Under **IEC 62366-1:2015** (Application of usability engineering to medical devices) and FDA 2016 Guidance, medical software must undergo rigorous formative and summative usability testing to ensure that clinical user interfaces do not induce diagnostic errors, cognitive fatigue, or automation bias (over-reliance on machine learning recommendations).

This protocol defines:
1. Target clinical user profiles and operational contexts.
2. Formative and summative usability evaluation methodologies.
3. Explainability and feature attribution verification protocols.
4. Quantitative acceptance criteria (System Usability Scale [SUS], NASA-TLX workload, decision turnaround times).
5. Use-related hazard mitigation strategies.

---

## 2. Clinical User Profiles & Environmental Context

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                       CLINICAL USER PERSONA SUMMARY                         │
├─────────────────────────────────────────────────────────────────────────────┤
│ PRIMARY USER: Board-Certified Neurologist / Movement Disorder Specialist     │
│ - Experience: >= 5 years clinical practice post-residency                   │
│ - Patient Volume: 15 - 25 outpatients per clinic session                    │
│ - Critical Needs: Rapid longitudinal progression trajectory summary,        │
│   biomarker change verification (DaTscan / voice / MRI), explainable rationale│
├─────────────────────────────────────────────────────────────────────────────┤
│ SECONDARY USER: Neurology Clinical Research Fellow / Resident               │
│ - Experience: 1 - 3 years post-graduate clinical training                   │
│ - Critical Needs: Clear linkage to Movement Disorder Society guidelines,    │
│   transparent confidence intervals, structured CDS audit trails              │
├─────────────────────────────────────────────────────────────────────────────┤
│ CLINICAL CONTEXT: Outpatient Neurology Specialty Clinic                     │
│ - Time Constraints: 15 to 30 minutes allocated per patient consultation     │
│ - Cognitive Environment: High multitasking, frequent electronic chart checks │
│ - Software Latency SLO: Sub-second response (< 200 ms) for all inferences   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Explainability & Feature Attribution Design Standards

To prevent the model from operating as an opaque "black box", AiMedRes enforces the following explainability standards for all clinical outputs:

### 3.1 Directional Feature Attribution (Waterfall Rationale)
For every predicted disease progression milestone, the system presents an ordered waterfall summary showing:
- **Baseline Risk**: The historical population pre-test probability ($\approx 50\%$).
- **Signed Individual Contributions**:
  - **Risk-Increasing Factors (Red/Amber)**: E.g., Reduced striatal binding ratio in left putamen ($c = +0.28$), advanced age ($c = +0.12$), elevated vocal tremor ($c = +0.08$).
  - **Protective/Stabilizing Factors (Green/Blue)**: E.g., Preserved caudate uptake ($c = -0.15$), intact cognitive MMSE score ($c = -0.10$).
- **Net Probability Output**: The calibrated posterior probability with uncertainty bounds.

### 3.2 Clinical Domain Concordance Verification
The platform includes an automated concordance filter ([`FeatureAttributionEvaluator`](../../src/aimedres/clinical/usability.py)) asserting that model feature contributions conform to established medical reality:
- **Concordance Criterion**: In $\ge 85\%$ of evaluations, top contributing risk factors must match clinical consensus (e.g., lower dopaminergic transporter density in the striatum must correlate with motor progression risk).

---

## 4. Summative Usability Protocol & Study Design

### 4.1 Study Overview & Participant Cohort
- **Participants**: $N = 15$ licensed clinicians (10 board-certified neurologists and 5 senior neurology fellows).
- **Format**: Randomized cross-over design.
- **Evaluation Set**: 20 standardized, de-identified patient vignettes drawn from the PPMI, OASIS-2, and UCI Telemonitoring cohorts:
  - 10 cases reviewed using **Standard EHR Summary Sheets Alone** (Control).
  - 10 cases reviewed using **AiMedRes Decision-Support Cards + Waterfall Explainability** (Intervention).

### 4.2 Study Tasks
For each patient case, the clinician executes four standard workflow tasks:
1. **Task 1 (Review & Stratification)**: Assess patient baseline and multi-modal markers, estimating the 12-month motor progression risk.
2. **Task 2 (Explainability Verification)**: Inspect the top 3 contributing biomarkers and confirm or dispute the statistical rationale.
3. **Task 3 (Clinical Decision)**: Formulate the recommended follow-up interval (e.g., 3 months vs. 6 months vs. 12 months) and staging recommendation.
4. **Task 4 (Electronic Sign-off)**: Record concordance (`concordant`, `modified`, or `overridden`) and sign off.

---

## 5. Quantitative Acceptance Criteria

In accordance with FDA human factors review thresholds and ANSI/AAMI HE75 benchmarks, AiMedRes must achieve the following quantitative criteria:

| Usability Metric | Standard Instrument / Formula | Benchmark Target | Regulatory Acceptance Threshold |
| :--- | :--- | :---: | :---: |
| **System Usability Scale (SUS)** | 10-item Likert Scale (Brooke, 1996) | Score $\ge 85.0$ | **Score $\ge 80.0 / 100$** (Grade A, High Usability) |
| **Cognitive Workload** | NASA-TLX (Hart & Staveland, 1988) | Workload $\le 25.0$ | **Workload $\le 35.0 / 100$** (Low Mental Demand) |
| **Task Completion Rate** | $\frac{\text{Completed Tasks}}{\text{Total Tasks}} \times 100$ | $100\%$ | **$\ge 95\%$** without facilitator prompting |
| **Decision Turnaround Time** | Total review time per complex case | $\le 1.5\text{ min}$ | **$\le 3.0\text{ minutes}$** per complex case |
| **Software Response Latency** | End-to-end inference + explainability | $\le 50\text{ ms}$ | **$\le 200\text{ ms}$** (Sub-second interactive SLO) |
| **Concordance Agreement** | Clinician agreement with model attributions | $\ge 90\%$ | **$\ge 85\%$** concordance across vignettes |
| **Safety-Critical Errors** | Misinterpretations leading to delayed care | $0$ | **$0$ unmitigated safety-critical errors** |

---

## 6. Use-Related Hazard Analysis & Risk Mitigations

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                      USE-RELATED HAZARDS & MITIGATIONS                      │
├─────────────────────────────────────────────────────────────────────────────┤
│ HAZARD 1: AUTOMATION BIAS (OVER-RELIANCE)                                   │
│ Risk: Clinician accepts model output without independent examination.       │
│ Mitigation:                                                                 │
│ - Probability is never presented as a single number; always shown with      │
│   95% bootstrap confidence intervals (e.g., 76% [CI: 74% - 78%]).           │
│ - Mandatory UI banner: "Investigational CDS aid. Clinical determination     │
│   requires independent physical neurological examination."                  │
├─────────────────────────────────────────────────────────────────────────────┤
│ HAZARD 2: DISMISSAL BIAS (UNDER-RELIANCE)                                   │
│ Risk: Clinician ignores valid high-risk early warning due to lack of trust. │
│ Mitigation:                                                                 │
│ - Provide direct bibliographic link to primary research consortium          │
│   (PPMI / Movement Disorder Society criteria) backing the model's weights.  │
│ - Display local waterfall feature contributions explaining the 'why'.       │
├─────────────────────────────────────────────────────────────────────────────┤
│ HAZARD 3: COGNITIVE INFORMATION OVERLOAD                                    │
│ Risk: Displaying 50+ raw telemetry coefficients confuses clinician.        │
│ Mitigation:                                                                 │
│ - Top-3 summary view with progressive disclosure ("View All Features").    │
│ - Clean typography, high-contrast visual indicators, sub-second latency.    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Automated Human Factors Test Harness

AiMedRes implements continuous automated verification of human factors criteria in [`tests/unit/test_usability_and_human_factors.py`](../../tests/unit/test_usability_and_human_factors.py):
1. **Decision Latency Verification**: Benchmarks 50 synthetic patient evaluations, asserting mean compute latency $< 200\text{ ms}$.
2. **Waterfall Consistency**: Asserts that feature attributions correctly sum to the model's logit delta.
3. **SUS & NASA-TLX Scoring Arithmetic**: Asserts zero computational error in Likert score calculations.
4. **Clinical Concordance Engine**: Asserts $\ge 85\%$ directional concordance on known neurodegenerative biomarkers.
