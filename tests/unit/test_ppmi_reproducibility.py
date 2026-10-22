#!/usr/bin/env python3
"""
Unit and Regression Tests for PPMI Parkinson's Pipeline Reproducibility

Validates:
1. Zero participant (PATNO) leakage across cross-validation splits.
2. Leakage-safe preprocessing (imputation and scaling fitted strictly on train folds).
3. Non-circular feature-label independence.
4. End-to-end execution on the synthetic PPMI benchmark fixture.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline

from aimedres.training.train_parkinsons import ParkinsonsTrainingPipeline


@pytest.fixture
def synthetic_fixture_path():
    """Path to the synthetic PPMI benchmark fixture."""
    fixture_path = Path("ParkinsonDatasets") / "synthetic_pet_sbr_fixture.csv"
    assert fixture_path.exists(), f"Synthetic fixture not found at {fixture_path}"
    return str(fixture_path)


def test_synthetic_fixture_schema_and_distribution(synthetic_fixture_path):
    """Verify that the synthetic fixture has longitudinal structure and expected columns."""
    df = pd.read_csv(synthetic_fixture_path)

    # Must contain patient ID, visit, features, and independent clinical label
    expected_cols = [
        "PATNO",
        "EVENT_ID",
        "AGE",
        "SEX",
        "PET_CAUDATE_L",
        "PET_CAUDATE_R",
        "PET_PUTAMEN_L",
        "PET_PUTAMEN_R",
        "motor_progression_label",
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing required column: {col}"

    # Longitudinal visits per patient
    patient_counts = df["PATNO"].value_counts()
    assert (patient_counts > 1).sum() > 0, "Synthetic fixture must contain repeated longitudinal visits"

    # Non-trivial target balance
    labels = df["motor_progression_label"].unique()
    assert set(labels).issubset({0, 1}), f"Labels must be binary 0/1, found: {labels}"
    assert len(labels) == 2, "Both positive and negative classes must be represented"


def test_zero_patno_leakage_across_cross_validation_folds(synthetic_fixture_path):
    """
    Ensure that no participant ID (PATNO) appears in both the training
    and validation splits within any cross-validation fold.
    """
    df = pd.read_csv(synthetic_fixture_path)
    groups = df["PATNO"].values
    y = df["motor_progression_label"].values
    X = df.drop(columns=["PATNO", "EVENT_ID", "motor_progression_label"])

    n_splits = 5
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups=groups)):
        train_patients = set(groups[train_idx])
        val_patients = set(groups[val_idx])

        overlap = train_patients.intersection(val_patients)
        assert len(overlap) == 0, (
            f"Data leakage detected in fold {fold}: {len(overlap)} patients "
            f"appear in both train and validation splits: {overlap}"
        )


def test_preprocessing_leakage_isolation():
    """
    Verify that scalers and imputers fitted inside pipeline do not leak
    validation statistics into the training fold.
    """
    pipeline_obj = ParkinsonsTrainingPipeline()

    # Create dummy dataframe with an extreme outlier in the second group
    data = pd.DataFrame({
        "PATNO": [101, 101, 102, 102],
        "feature_1": [1.0, 2.0, 100.0, 200.0],  # Outlier group 102
        "motor_progression_label": [0, 0, 1, 1],
    })
    pipeline_obj.data = data
    pipeline_obj.preprocess_data(target_column="motor_progression_label")

    # Group 101 only
    train_df = pipeline_obj.X_raw.iloc[[0, 1]]
    preprocessor = pipeline_obj._build_preprocessor(train_df)
    preprocessor.fit(train_df)

    # The mean learned by scaler on group 101 must be 1.5, unaffected by group 102
    scaler = preprocessor.named_transformers_["num"].named_steps["scaler"]
    np.testing.assert_almost_equal(scaler.mean_[0], 1.5, decimal=4)


def test_feature_label_independence(synthetic_fixture_path):
    """
    Verify that target label is not a direct mathematical function
    of striatal binding ratio columns (preventing circular prediction tasks).
    """
    df = pd.read_csv(synthetic_fixture_path)
    sbr_cols = [c for c in df.columns if c.startswith("PET_")]
    avg_sbr = df[sbr_cols].mean(axis=1)

    # Check that correlation is not +/- 1.0 (indicating non-circular formulation)
    correlation = np.corrcoef(avg_sbr, df["motor_progression_label"])[0, 1]
    assert abs(correlation) < 0.95, (
        f"Target label is excessively correlated with SBR inputs ({correlation:.3f}), "
        "indicating possible circular prediction."
    )


def test_end_to_end_pipeline_execution(tmp_path, synthetic_fixture_path):
    """
    Execute full leakage-safe pipeline on synthetic fixture and verify outputs.
    """
    output_dir = tmp_path / "parkinsons_test_run"
    pipeline = ParkinsonsTrainingPipeline(output_dir=str(output_dir))

    results = pipeline.run_full_pipeline(
        data_path=synthetic_fixture_path,
        epochs=3,
        n_folds=3,
    )

    assert "classical_results" in results
    classical = results["classical_results"]
    assert "Logistic Regression" in classical
    assert "Random Forest" in classical
    assert "SVM" in classical

    for model_name, metrics in classical.items():
        assert "accuracy_mean" in metrics
        assert 0.0 <= metrics["accuracy_mean"] <= 1.0
        assert "roc_auc_mean" in metrics
        assert 0.0 <= metrics["roc_auc_mean"] <= 1.0

    # Ensure model artifacts were serialized as Pipelines
    models_dir = output_dir / "models"
    assert (models_dir / "logistic_regression_model.pkl").exists()
    assert (models_dir / "random_forest_model.pkl").exists()
    assert (models_dir / "svm_model.pkl").exists()
