"""
Unit tests for Zero-Wait Open-Access Dataset Connectors & Leakage Safety.

Verifies:
1. OASIS-2 Longitudinal dataset loading, schema, and non-circular feature extraction.
2. Zero subject leakage across cross-validation folds for OASIS repeat visits.
3. PRO-ACT ALS clinical trials dataset loading and target independence.
4. Zero subject leakage across cross-validation folds for PRO-ACT longitudinal visits.
5. PTB-XL ECG dataset loading and patient-level isolation.
6. End-to-end leakage-safe classification with fold-isolated preprocessing.
"""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score

from aimedres.data.open_datasets import (
    OASISDataset,
    ParkinsonsTelemonitoringDataset,
    PROACTALSDataset,
    PTBXLECGDataset,
    get_open_dataset,
    get_leakage_safe_cv,
)


def test_oasis_loader_and_schema():
    """Verify OASIS-2 longitudinal loader, schema integrity, and anti-leakage safeguards."""
    dataset = OASISDataset.load()
    
    assert dataset.name == "OASIS-2 Longitudinal Neuroimaging"
    assert len(dataset.df) > 50
    assert dataset.groups is not None
    assert len(dataset.groups) == len(dataset.df)
    
    # Check that CDR was dropped from features to prevent target circularity
    assert "CDR" not in dataset.feature_names
    assert "CDR" not in dataset.features.columns
    
    # Check that Subject ID and MRI ID were excluded from features
    assert "Subject ID" not in dataset.features.columns
    assert "MRI ID" not in dataset.features.columns
    
    # Check key neuroimaging features are retained
    for expected_feat in ["Age", "EDUC", "MMSE", "eTIV", "nWBV", "ASF"]:
        assert expected_feat in dataset.feature_names
        
    # Check binary target distribution
    unique_targets = set(np.unique(dataset.target))
    assert unique_targets.issubset({0, 1})


def test_oasis_zero_subject_leakage():
    """Verify that multiple longitudinal visits for any subject never span both train and val folds."""
    dataset = OASISDataset.load()
    cv = get_leakage_safe_cv(dataset.groups, n_splits=5, y=dataset.target.values)
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(dataset.features, dataset.target, dataset.groups)):
        train_subjects = set(dataset.groups[train_idx])
        val_subjects = set(dataset.groups[val_idx])
        
        overlap = train_subjects.intersection(val_subjects)
        assert len(overlap) == 0, f"Subject leakage detected in OASIS fold {fold}: {overlap}"


def test_proact_als_loader_and_schema():
    """Verify PRO-ACT ALS clinical trials loader and target independence."""
    dataset = PROACTALSDataset.load()
    
    assert "PRO-ACT" in dataset.name
    assert len(dataset.df) > 50
    assert dataset.groups is not None
    
    # Check that progression derivatives are dropped from feature matrix
    assert "Progression_Slope" not in dataset.feature_names
    assert "Progression_Rate" not in dataset.feature_names
    assert "Fast_Progression" not in dataset.feature_names
    
    # Check clinical biomarkers are retained
    for expected_feat in ["Age", "ALSFRS_R_Total", "FVC_percent", "Creatinine", "Albumin"]:
        assert expected_feat in dataset.feature_names
        
    # Check target
    assert set(np.unique(dataset.target)).issubset({0, 1})


def test_proact_zero_subject_leakage():
    """Verify zero subject leakage across PRO-ACT cross-validation folds."""
    dataset = PROACTALSDataset.load()
    cv = get_leakage_safe_cv(dataset.groups, n_splits=5, y=dataset.target.values)
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(dataset.features, dataset.target, dataset.groups)):
        train_subjects = set(dataset.groups[train_idx])
        val_subjects = set(dataset.groups[val_idx])
        
        overlap = train_subjects.intersection(val_subjects)
        assert len(overlap) == 0, f"Subject leakage detected in PRO-ACT fold {fold}: {overlap}"


def test_ptbxl_ecg_loader_and_schema():
    """Verify PTB-XL ECG loader and patient-level group isolation."""
    dataset = PTBXLECGDataset.load()
    
    assert "PTB-XL" in dataset.name
    assert len(dataset.df) > 50
    assert "patient_id" == dataset.group_name
    
    # Check intervals present
    for interval in ["heart_rate", "pr_interval", "qrs_duration", "qt_interval", "qtc_interval"]:
        assert interval in dataset.feature_names
        
    # Check zero patient leakage
    cv = get_leakage_safe_cv(dataset.groups, n_splits=5, y=dataset.target.values)
    for fold, (train_idx, val_idx) in enumerate(cv.split(dataset.features, dataset.target, dataset.groups)):
        train_patients = set(dataset.groups[train_idx])
        val_patients = set(dataset.groups[val_idx])
        overlap = train_patients.intersection(val_patients)
        assert len(overlap) == 0, f"Patient leakage detected in PTB-XL fold {fold}: {overlap}"


def test_end_to_end_oasis_leakage_safe_training():
    """Verify that a pipeline with preprocessing and classifier executes without contamination."""
    dataset = OASISDataset.load()
    X = dataset.features.values
    y = dataset.target.values
    groups = dataset.groups
    
    cv = get_leakage_safe_cv(groups, n_splits=5, y=y)
    fold_aucs = []
    
    for train_idx, val_idx in cv.split(X, y, groups):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        
        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(random_state=42, max_iter=500)),
        ])
        
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict_proba(X_val)[:, 1]
        
        auc = roc_auc_score(y_val, preds)
        fold_aucs.append(auc)
        
    mean_auc = np.mean(fold_aucs)
    assert mean_auc > 0.50, f"Pipeline failed to learn meaningful signal on OASIS: AUC={mean_auc}"
    assert len(fold_aucs) == 5


def test_parkinsons_telemonitoring_loader_and_schema():
    """Verify Parkinson's Telemonitoring loader, anti-leakage drops, and acoustic biomarkers."""
    dataset = ParkinsonsTelemonitoringDataset.load()

    assert "Parkinson" in dataset.name
    assert len(dataset.df) > 300
    assert dataset.groups is not None
    assert len(dataset.groups) == len(dataset.df)

    # Verify circular leakage protection: raw UPDRS scores dropped from input features
    assert "motor_UPDRS" not in dataset.feature_names
    assert "total_UPDRS" not in dataset.feature_names
    assert "baseline_motor" not in dataset.feature_names
    assert "delta_motor" not in dataset.feature_names
    assert "subject#" not in dataset.feature_names

    # Verify acoustic vocal biomarkers are retained
    for biomarker in ["Jitter(%)", "Shimmer", "NHR", "HNR", "RPDE", "DFA", "PPE"]:
        assert biomarker in dataset.feature_names

    # Check binary target
    assert set(np.unique(dataset.target)).issubset({0, 1})


def test_parkinsons_telemonitoring_zero_subject_leakage():
    """Verify that multiple telemetry sessions for any patient never span both train and test folds."""
    dataset = ParkinsonsTelemonitoringDataset.load()
    cv = get_leakage_safe_cv(dataset.groups, n_splits=5, y=dataset.target.values)

    for fold, (train_idx, val_idx) in enumerate(cv.split(dataset.features, dataset.target, dataset.groups)):
        train_patients = set(dataset.groups[train_idx])
        val_patients = set(dataset.groups[val_idx])

        overlap = train_patients.intersection(val_patients)
        assert len(overlap) == 0, f"Patient leakage detected in Parkinson's fold {fold}: {overlap}"


def test_get_open_dataset_factory():
    """Verify factory routes all four clinical cohorts correctly."""
    d1 = get_open_dataset("oasis")
    assert "OASIS" in d1.name

    d2 = get_open_dataset("proact")
    assert "PRO-ACT" in d2.name

    d3 = get_open_dataset("ptbxl")
    assert "PTB-XL" in d3.name

    d4 = get_open_dataset("parkinsons_telemonitoring")
    assert "Parkinson" in d4.name
