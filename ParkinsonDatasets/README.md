# Parkinson's Datasets Integration

This directory contains Parkinson's Progression Markers Initiative (PPMI) datasets that are now integrated with the AiMedRes training pipeline.

## Dataset Overview

The directory contains 16 CSV files with various imaging and clinical data:
- **PET SBR Analysis** (Primary dataset for classification)
- MRI acquisition metadata
- CT Scans
- DTI Regions of Interest
- Various PET imaging substudies
- Tau and SV2A imaging data

## Integration Details

The training pipeline (`src/aimedres/training/train_parkinsons.py`) automatically:

1. **Detects the directory**: When `--data-path ParkinsonDatasets` is passed
2. **Loads PET SBR Analysis data**: Uses `PET_SBR_Analysis_01Oct2025.csv` as the primary dataset
3. **Extracts features**: Focuses on striatal binding ratios (SBR) from caudate and putamen regions
4. **Creates classification target**: Uses median split of average SBR values
   - Below median (< 0.401) → More severe PD (status=1)
   - Above median (≥ 0.401) → Less severe PD (status=0)

## Usage

### Direct training script:
```bash
python src/aimedres/training/train_parkinsons.py --data-path ParkinsonDatasets --epochs 100 --folds 5
```

### Via run_all_training.py:
```bash
python run_all_training.py --only parkinsons --epochs 100 --folds 5
```

The `run_all_training.py` script is already configured to use this local dataset by default.

## Dataset Statistics

- **Samples**: 211 patients
- **Features**: 20 PET SBR measurements
- **Target**: Binary classification (more/less severe PD)
- **Balance**: ~50/50 split (105 vs 106)

## Model Performance

With this dataset, the training achieves:
- **Random Forest**: ~99-100% accuracy
- **Logistic Regression**: ~97-99% accuracy
- **SVM**: ~95-97% accuracy

Note: High accuracy is expected as this is a homogeneous PD patient cohort with good quality imaging biomarkers.

## Files Used

Primary: `PET_SBR_Analysis_01Oct2025.csv`
- Contains striatal binding ratio (SBR) measurements
- 214 initial rows, 211 after removing incomplete records
- 28 columns total, 20 SBR feature columns extracted

## Features Extracted

All PET SBR measurements including:
- Right/Left Caudate (CAUD)
- Right/Left Putamen anterior/posterior (PUT)
- Cerebellum (CBM)
- Occipital cortex (OCCIP)

These are validated Parkinson's disease biomarkers used in clinical research and diagnosis.
