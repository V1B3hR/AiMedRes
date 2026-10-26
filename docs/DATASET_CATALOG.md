# AiMedRes Dataset Catalog & Acquisition Guide

> **Important Notice on Data Governance & Compliance**  
> This repository's data-governance rules are designed to support DUA, privacy, and institutional requirements; they do **not** establish HIPAA or GDPR compliance. **Real patient data must never be committed to this repository.** CI/CD, automated testing, and development use synthetic fixtures located in `ParkinsonDatasets/` or generated on the fly. Real clinical datasets must remain in an untracked, authorised local environment and follow their DUA and institutional controls.

---

## 1. Primary Hub: USC LONI Image & Data Archive (IDA)

- **Portal URL**: [https://ida.loni.usc.edu/login.jsp](https://ida.loni.usc.edu/login.jsp)
- **Institution**: Laboratory of Neuro Imaging (LONI), Keck School of Medicine of USC
- **Access Type**: **Free of charge** for qualified researchers, educators, and students.
- **Access Model**: Controlled scientific access via online Data Use Agreement (DUA) and identity verification.

LONI IDA is the world's premier repository for neurodegenerative, neurovascular, and neurotrauma imaging and clinical data. It directly matches AiMedRes's core disease models.

### How to Register and Access LONI IDA Datasets

1. **Create an Account**: Navigate to [https://ida.loni.usc.edu/login.jsp](https://ida.loni.usc.edu/login.jsp) and click **"Create an Account"** or select **"Access Data"**.
2. **Affiliation & Verification**: Provide your institutional or university email address and institutional affiliation.
3. **Study-Specific DUA Application**: In the IDA portal, browse to the target project (e.g., PPMI, ADNI, MarkVCID) and submit an online application:
   - Provide a brief (1–2 paragraph) description of your proposed research (e.g., "Evaluating machine learning models for early disease progression detection and cross-cohort validation").
   - Review and digitally accept the specific Data Use Agreement (prohibiting patient re-identification and third-party redistribution).
4. **Approval Turnaround**:
   - **PPMI**: Automated or 24–48 hour administrative turnaround.
   - **ADNI**: Standard committee review (typically 1–4 business days).
5. **Data Download**: Once approved, access the Advanced Search/Download interface to retrieve tabular clinical files (CSV), biospecimen assays, and imaging collections (DICOM / NIfTI).

---

## 2. LONI IDA Datasets Matching AiMedRes Modules

| Study Name | Disease / Domain | Modalities | Key Target Variables | AiMedRes Target Module |
| :--- | :--- | :--- | :--- | :--- |
| **PPMI** (Parkinson's Progression Markers Initiative) | Parkinson's Disease | [123I]FP-CIT DaTscan SPECT, 3T MRI (T1/T2/DTI), Biospecimens | Striatal Binding Ratios (SBR), MDS-UPDRS (I–IV), MoCA, CSF $\alpha$-synuclein SAA, NfL | `src/aimedres/training/train_parkinsons.py`<br>`ParkinsonDatasets/` |
| **ADNI** (1, GO, 2, 3, 4) (Alzheimer's Disease Neuroimaging Initiative) | Alzheimer's Disease & MCI | Volumetric MRI (1.5T/3T), FDG-PET, Amyloid-PET, Tau-PET | CDR-SB, MMSE, ADAS-Cog, CSF A$\beta_{42}$ / p-tau181, APOE $\varepsilon4$ | `src/aimedres/training/train_alzheimers.py`<br>`docs/STRUCTURED_ALZ_TRAINING.md` |
| **MarkVCID** (Vascular Contributions to Cognitive Impairment & Dementia) | Vascular Dementia & Stroke Risk | FLAIR MRI, DTI, Perfusion MRI, Endothelial Biomarkers | White Matter Hyperintensity (WMH) volume, Free Water, Microbleeds, Executive function | `src/aimedres/training/train_cardiovascular.py`<br>Clinical Decision Support (CDS) |
| **AIBL** (Australian Imaging, Biomarkers & Lifestyle Study) | Alzheimer's Progression & Ageing | Longitudinal Amyloid-PET, Structural MRI, Cognitive batteries | Amyloid SUVR, CDR, MMSE, Clinical progression | Cross-cohort validation for Alzheimer's models |
| **A4 Study** (Anti-Amyloid in Asymptomatic AD) | Preclinical Alzheimer's | Florbetapir Amyloid-PET, Volumetric MRI, Preclinical Cognitive Composite (PACC96) | Preclinical amyloid accumulation, PACC96 score trajectory | Preclinical prevention and early risk stratification |
| **FTLDNI** (Frontotemporal Lobar Degeneration Neuroimaging Initiative) | Frontotemporal Dementia (bvFTD, PPA) | Volumetric MRI, FDG-PET, Clinical Dementia Rating (CDR plus NACC FTLD) | Frontal/temporal lobar atrophy, FTLD-CDR sum of boxes | Differential diagnosis against Alzheimer's / Parkinson's |
| **TRACK-TBI** (Transforming Research and Clinical Knowledge in TBI) | Traumatic Brain Injury & Neurotrauma | Acute/Subacute CT & MRI, Serum biomarkers (GFAP, UCH-L1) | Extended Glasgow Outcome Scale (GOS-E), ICU duration | Neurotrauma severity and recovery prediction |

---

## 3. Immediate "Zero-Wait" Open-Access Alternatives

If you are developing locally and cannot immediately wait for institutional DUA approval on LONI IDA, the following peer-reviewed, open-access public repositories provide immediate, direct-download data matching repository components. AiMedRes provides native, leakage-safe data connectors in `aimedres.data` for each:

### 3.1 OASIS-2 Longitudinal Neuroimaging (Alzheimer's & Brain Ageing)

* **Portal**: [https://www.oasis-brains.org/](https://www.oasis-brains.org/) / NITRC Open Mirror
* **Reference**: Marcus et al., *Journal of Cognitive Neuroscience*, 2010.
* **Access Type**: Fully open access, no institutional approval required.
* **Target Task**: Early dementia classification (`Nondemented` vs. `Demented` / `Converted`) and cognitive decline prediction.

#### Schema & Variables
| Feature | Type | Range / Values | Description |
| :--- | :--- | :--- | :--- |
| `Subject ID` | String | `OAS2_0001` - `OAS2_0186` | **Participant Group Key** (must group CV folds) |
| `MRI ID` | String | `OAS2_0001_MR1` ... | Longitudinal scan session ID |
| `Visit` | Integer | 1, 2, 3, 4, 5 | Scan visit sequence number |
| `MR Delay` | Integer | 0 - 2639 days | Elapsed days since baseline visit |
| `M/F` | Categorical | `M`, `F` | Biological sex |
| `Age` | Float | 60.0 - 98.0 years | Patient age at scan date |
| `EDUC` | Integer | 6 - 23 years | Years of formal education |
| `SES` | Integer | 1 - 5 | Socioeconomic status (Hollingshead Index) |
| `MMSE` | Integer | 15 - 30 | Mini-Mental State Examination score |
| `eTIV` | Float | 1100 - 2000 $mm^3$ | Estimated Total Intracranial Volume |
| `nWBV` | Float | 0.64 - 0.84 | Normalized Whole Brain Volume (% of eTIV) |
| `ASF` | Float | 0.87 - 1.58 | Atlas Scaling Factor |
| `Group` | Target | `Nondemented`, `Demented`, `Converted` | Clinical diagnostic category |

> [!WARNING]
> **Scientific Integrity Warning on `CDR`**: The Clinical Dementia Rating (`CDR`) is derived by clinicians directly from dementia symptoms. **Never include `CDR` in the input feature matrix when predicting dementia status (`Group`)**, as doing so creates circular target leakage. `OASISDataset` automatically drops `CDR` from features by default.

#### Programmatic Loading & Training
```python
from aimedres.data import OASISDataset, get_leakage_safe_cv

# Automatically loads local raw data if present, or calibrated synthetic fixture:
dataset = OASISDataset.load(binary_dementia_only=True)
print(f"Loaded {len(dataset.df)} records across {len(set(dataset.groups))} subjects")

# Generate participant-isolated cross-validation splits:
cv = get_leakage_safe_cv(dataset.groups, n_splits=5, y=dataset.target.values)
```

---

### 3.2 PRO-ACT ALS Clinical Database (Amyotrophic Lateral Sclerosis)

* **Portal**: [https://nctu.partners.org/ProACT/](https://nctu.partners.org/ProACT/)
* **Scale**: Over 10,700 patient records pooled from 23 Phase II/III clinical trials.
* **Access Type**: Free online registration with instant portal access.
* **Target Task**: Predicting rapid disease progression (`Fast_Progression`) or longitudinal decline slope.

#### Schema & Clinical Features
| Feature | Type | Units / Range | Description |
| :--- | :--- | :--- | :--- |
| `Subject_ID` | String | `ALS_0001` ... | **Participant Group Key** (enforces zero longitudinal fold leakage) |
| `Visit_Month` | Integer | 0, 3, 6, 9, 12 | Month of follow-up evaluation |
| `Age` | Float | 28 - 82 years | Age at diagnosis |
| `Gender` | Categorical | `Male`, `Female` | Biological sex |
| `Site_of_Onset` | Categorical | `Bulbar`, `Limb` | Initial clinical presentation site |
| `Disease_Duration_Months` | Integer | 3 - 60 months | Duration of symptoms prior to assessment |
| `ALSFRS_R_Total` | Integer | 0 - 48 | Revised ALS Functional Rating Scale |
| `FVC_percent` | Float | 20 - 130% | Forced Vital Capacity (% predicted) |
| `Creatinine` | Float | 0.4 - 2.2 mg/dL | Serum creatinine (muscle mass proxy) |
| `Albumin` | Float | 3.1 - 5.2 g/dL | Serum albumin (nutritional status) |
| `ALT` | Float | 8 - 65 U/L | Liver enzyme safety marker |
| `Riluzole_Usage` | Binary | 0, 1 | Standard of care concomitant drug |
| `Fast_Progression` | Target | 0, 1 | Label: 1 if progression slope $\ge 1.0$ pt/mo |

#### Programmatic Loading & Training
```python
from aimedres.data import PROACTALSDataset, get_leakage_safe_cv

# Load PRO-ACT dataset with automatic leakage safeguards:
dataset = PROACTALSDataset.load(target_col="Fast_Progression")

# Run cross-validation grouped strictly by patient ID:
cv = get_leakage_safe_cv(dataset.groups, n_splits=5, y=dataset.target.values)
```

---

### 3.3 PhysioNet PTB-XL ECG Database (Cardiovascular Risk)

* **Portal**: [https://physionet.org/content/ptb-xl/1.0.3/](https://physionet.org/content/ptb-xl/1.0.3/)
* **Reference**: Wagner et al., *Scientific Data*, 2020.
* **Access Type**: Creative Commons Attribution 4.0 International (CC-BY 4.0) — direct open download.
* **Target Task**: Multi-class diagnostic superclass prediction (`NORM`, `MI`, `STTC`, `CD`, `HYP`) or abnormal rhythm screening.

#### Schema & Electrophysiological Metrics
| Feature | Type | Units | Description |
| :--- | :--- | :--- | :--- |
| `patient_id` | String | `PTB_0001` ... | **Patient Identifier** (isolates repeated recordings) |
| `ecg_id` | Integer | 1 - 21837 | Unique record identifier |
| `heart_rate` | Float | 40 - 140 bpm | Resting heart rate |
| `pr_interval` | Float | 100 - 280 ms | PR interval duration |
| `qrs_duration` | Float | 60 - 180 ms | QRS complex duration (ventricular depolarization) |
| `qt_interval` | Float | 300 - 520 ms | Uncorrected QT interval |
| `qtc_interval` | Float | 340 - 500 ms | Bazett-corrected QT interval |
| `diagnostic_class` | Target | NORM / MI / STTC / CD / HYP | Primary ECG diagnostic superclass |
| `is_abnormal` | Target | 0 (Normal) vs 1 (Abnormal) | Binary screening endpoint |

#### Programmatic Loading
```python
from aimedres.data import PTBXLECGDataset

dataset = PTBXLECGDataset.load(target_col="is_abnormal")
print(f"Loaded {len(dataset.df)} ECGs across {len(set(dataset.groups))} patients")
```

---

### 3.4 UCI / Oxford Parkinson's Vocal Telemonitoring Study

* **Portal**: [https://archive.ics.uci.edu/dataset/189/parkinsons+telemonitoring](https://archive.ics.uci.edu/dataset/189/parkinsons+telemonitoring)
* **Reference**: Tsanas et al., *IEEE Transactions on Biomedical Engineering*, 2010.
* **Scale**: 5,875 vocal recordings across 42 patients monitored longitudinally for 6 months.
* **Access Type**: Fully open access, no institutional DUA needed.
* **Target Task**: Motor UPDRS progression prediction ($\Delta\text{motor\_UPDRS} \ge 3.0$ points above patient baseline) using 16 noninvasive acoustic vocal biomarkers.

#### Schema & Clinical Features
| Feature | Type | Units / Range | Description |
| :--- | :--- | :--- | :--- |
| `subject#` | Integer | 1 - 42 | **Participant Group Key** (enforces zero-leakage cross-validation) |
| `age` | Integer | 36 - 85 years | Patient age |
| `sex` | Binary | 0 (Male), 1 (Female) | Biological sex |
| `test_time` | Float | 0 - 180 days | Elapsed study follow-up time |
| `motor_UPDRS` | Float | 5.0 - 45.0 | Clinician-scored motor symptom severity (dropped from features) |
| `Jitter(%)`, `Jitter(Abs)`, etc. | Float | Multiple | Acoustic frequency variation perturbations (vocal tremor proxy) |
| `Shimmer`, `Shimmer(dB)`, etc. | Float | Multiple | Acoustic amplitude variation perturbations (phonation instability) |
| `NHR`, `HNR` | Float | Ratio / dB | Noise-to-Harmonics & Harmonics-to-Noise ratio (dysphonia marker) |
| `RPDE` | Float | 0.2 - 0.9 | Recurrence Period Density Entropy (vocal signal complexity) |
| `DFA` | Float | 0.5 - 0.9 | Detrended Fluctuation Analysis (fractal scaling exponent) |
| `PPE` | Float | 0.05 - 0.6 | Pitch Period Entropy (impaired pitch control) |

#### Programmatic Loading & Training
```python
from aimedres.data import ParkinsonsTelemonitoringDataset, get_leakage_safe_cv

# Load telemonitoring data with automatic circular target exclusion:
dataset = ParkinsonsTelemonitoringDataset.load(progression_threshold=3.0)

# 5-fold CV strictly grouped by patient ID (zero subject overlap):
cv = get_leakage_safe_cv(dataset.groups, n_splits=5, y=dataset.target.values)
```

---

### 3.5 Historical real-world benchmark claims — not active evidence

The repository contains scripts and historical notes for open-access cohort work. They are **not** the maintained PPMI evidence path, and the historical run records do not provide the complete data hashes, environment, protocol, and independent rerun required for an external claim. Do not cite the values formerly listed here as verified clinical-cohort performance.

#### One-Command Acquisition & Training
```bash
# 1. Download genuine patient cohorts to data/real_world/ (git-ignored)
python scripts/download_real_world_datasets.py

# 2. Run participant-isolated cross-validation with 95% bootstrap CIs
python scripts/train_real_world_datasets.py
```

No empirical metric table is maintained in this catalog until a rerun has a frozen protocol, source/version/DUA record, dataset hash, environment, code commit, complete artefacts, and independent review.

---

### 3.6 Direct Download Commands (Zero-Wait Terminal Setup)

To pull real open-access clinical tables directly to your local machine into `data/real_world/`:

```bash
# Execute automated downloader for all zero-wait open-access cohorts:
python scripts/download_real_world_datasets.py

# Verify files with unit test suite
pytest tests/unit/test_open_access_datasets.py
```

---

## 4. Setting Up Local Data for AiMedRes Training

When working with downloaded datasets, adhere strictly to the following directory convention:

```
AiMedRes/
├── ParkinsonDatasets/
│   ├── DATA_GOVERNANCE.md            # Compliance rules (committed)
│   ├── synthetic_pet_sbr_fixture.csv # Synthetic CI fixture (committed)
│   └── README.md
├── data/                             # LOCAL ONLY (ignored by .gitignore)
│   ├── ppmi/
│   │   ├── ppmi_sbr_longitudinal.csv # Real PPMI Curated DaTscan export
│   │   └── mds_updrs_scores.csv      # Real PPMI Clinical MDS-UPDRS
│   ├── adni/
│   │   ├── ADNIMERGE.csv             # Standard ADNI summary table
│   │   └── pet_suvr_summary.csv
│   └── oasis/
│       └── oasis_longitudinal.csv
```

### Running the Leakage-Safe Parkinson's Pipeline with Real PPMI Data

Once your PPMI CSV files are downloaded to `data/ppmi/ppmi_sbr_longitudinal.csv`:

```bash
# Execute the maintained research runner against an authorised local export:
python scripts/reproduce_ppmi_benchmark.py \
  --config configs/parkinsons_ppmi_study.yml \
  --data-path data/ppmi/ppmi_sbr_longitudinal.csv \
  --output-dir .local-results/ppmi_research_run
```

The researcher must verify that the controlled export contains `PATNO`, a preregistered independent endpoint, and sufficient group/class support before interpreting an output. When those conditions hold, the maintained training path uses participant grouping and fold-contained preprocessing; the run still requires independent methodological review before any external claim.
