"""
Real-World Clinical Dataset Training Runner for AiMedRes.
Trains leakage-safe, participant-isolated models on genuine human clinical cohorts:
1. OASIS-2 Longitudinal Dementia MRI Cohort (373 visits, 150 subjects)
2. UCI Parkinson's Telemonitoring Study (5,875 records, 42 patients, motor UPDRS progression)

Computes out-of-fold predictions, 95% bootstrap confidence intervals, and Brier score calibration.
Logs all runs to LocalExperimentTracker and saves verified model artifacts.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import pickle
import sys
import time
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, precision_recall_curve, roc_auc_score, auc
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from aimedres.research.tracking import LocalExperimentTracker

logger = logging.getLogger("aimedres.real_world_training")


def compute_bootstrap_ci(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    metric_func: Any,
    n_bootstraps: int = 1000,
    alpha: float = 0.05,
    random_seed: int = 42,
) -> Tuple[float, float, float]:
    rng = np.random.RandomState(random_seed)
    point_est = float(metric_func(y_true, y_pred_proba))
    boot_scores: List[float] = []

    for _ in range(n_bootstraps):
        idx = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            boot_scores.append(float(metric_func(y_true[idx], y_pred_proba[idx])))
        except Exception:
            continue

    if not boot_scores:
        return point_est, point_est, point_est

    low = float(np.percentile(boot_scores, 100 * (alpha / 2.0)))
    high = float(np.percentile(boot_scores, 100 * (1.0 - alpha / 2.0)))
    return point_est, low, high


def pr_auc_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    p, r, _ = precision_recall_curve(y_true, y_pred)
    return float(auc(r, p))


# ==============================================================================
# 1. OASIS-2 Longitudinal Training
# ==============================================================================
def train_oasis2_cohort(
    data_path: Path = Path("data/real_world/oasis_longitudinal.csv"),
    output_dir: Path = Path("results/real_world_training/oasis_2"),
    tracker: LocalExperimentTracker | None = None,
) -> Dict[str, Any]:
    logger.info("=== Training Real-World OASIS-2 Longitudinal Dementia Cohort ===")
    df = pd.read_csv(data_path)

    # Participant grouping key
    groups = df["Subject ID"].values

    # Target: Dementia status (excluding CDR to prevent circular leakage)
    y = (df["Group"] != "Nondemented").astype(int).values

    # Features (clinical measurements, MRI volumetrics, demographics)
    feature_cols = ["Age", "EDUC", "SES", "MMSE", "eTIV", "nWBV", "ASF", "M/F"]
    X = df[feature_cols].copy()

    numeric_features = ["Age", "EDUC", "SES", "MMSE", "eTIV", "nWBV", "ASF"]
    categorical_features = ["M/F"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                numeric_features,
            ),
            (
                "cat",
                Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore"))]),
                categorical_features,
            ),
        ]
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
    }

    sgkf = StratifiedGroupKFold(n_splits=5)
    results = {}

    run_id = None
    if tracker:
        run_id = tracker.start_run(
            run_name="oasis2_longitudinal_5fold",
            params={"cohort": "OASIS-2", "n_subjects": len(set(groups)), "n_visits": len(df), "target": "dementia_diagnosis"},
            tags={"domain": "neurology", "modality": "longitudinal_mri"},
            output_dir=output_dir,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "models").mkdir(parents=True, exist_ok=True)

    for name, clf in models.items():
        oof_preds = np.zeros(len(df))
        oof_probs = np.zeros(len(df))

        pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", clf)])

        for train_idx, val_idx in sgkf.split(X, y, groups=groups):
            # Assert zero subject leakage across folds
            assert len(set(groups[train_idx]).intersection(set(groups[val_idx]))) == 0

            X_tr, y_tr = X.iloc[train_idx], y[train_idx]
            X_vl = X.iloc[val_idx]

            pipeline.fit(X_tr, y_tr)
            probs = pipeline.predict_proba(X_vl)[:, 1]
            oof_probs[val_idx] = probs
            oof_preds[val_idx] = (probs >= 0.5).astype(int)

        acc = float(accuracy_score(y, oof_preds))
        roc_auc, roc_low, roc_high = compute_bootstrap_ci(y, oof_probs, roc_auc_score)
        pr_auc, pr_low, pr_high = compute_bootstrap_ci(y, oof_probs, pr_auc_metric)
        brier = float(brier_score_loss(y, oof_probs))

        # Fit final pipeline on all data and save
        pipeline.fit(X, y)
        model_filename = f"{name.lower().replace(' ', '_')}_model.pkl"
        model_path = output_dir / "models" / model_filename
        with open(model_path, "wb") as f:
            pickle.dump(pipeline, f)

        if tracker and run_id:
            tracker.log_metrics(
                {
                    f"{name}_roc_auc": roc_auc,
                    f"{name}_pr_auc": pr_auc,
                    f"{name}_brier": brier,
                    f"{name}_acc": acc,
                },
                run_id=run_id,
            )
            tracker.log_artifact(name, model_path, run_id=run_id)

        results[name] = {
            "accuracy": acc,
            "roc_auc": roc_auc,
            "roc_auc_ci": [roc_low, roc_high],
            "pr_auc": pr_auc,
            "pr_auc_ci": [pr_low, pr_high],
            "brier_score": brier,
        }

    if tracker and run_id:
        tracker.end_run(status="COMPLETED", run_id=run_id)

    return results


# ==============================================================================
# 2. UCI Parkinson's Telemonitoring Training
# ==============================================================================
def train_parkinsons_telemonitoring_cohort(
    data_path: Path = Path("data/real_world/parkinsons_telemonitoring.csv"),
    output_dir: Path = Path("results/real_world_training/parkinsons_telemonitoring"),
    tracker: LocalExperimentTracker | None = None,
) -> Dict[str, Any]:
    logger.info("=== Training Real-World UCI Parkinson's Telemonitoring Cohort ===")
    df = pd.read_csv(data_path)

    # Participant grouping key
    groups = df["subject#"].values

    # Calculate subject-specific baseline motor_UPDRS (at min test_time)
    baseline_motor = df.groupby("subject#").apply(lambda g: g.sort_values("test_time").iloc[0]["motor_UPDRS"]).to_dict()
    df["baseline_motor"] = df["subject#"].map(baseline_motor)
    df["delta_motor"] = df["motor_UPDRS"] - df["baseline_motor"]

    # Target: Meaningful motor progression milestone (delta >= 3.0 points)
    y = (df["delta_motor"] >= 3.0).astype(int).values

    # Features: 16 acoustic vocal biomarkers + age + sex + elapsed time
    acoustic_cols = [
        "Jitter(%)", "Jitter(Abs)", "Jitter:RAP", "Jitter:PPQ5", "Jitter:DDP",
        "Shimmer", "Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5", "Shimmer:APQ11", "Shimmer:DDA",
        "NHR", "HNR", "RPDE", "DFA", "PPE"
    ]
    feature_cols = ["age", "sex", "test_time"] + acoustic_cols
    X = df[feature_cols].copy()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                feature_cols,
            ),
        ]
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42),
    }

    sgkf = StratifiedGroupKFold(n_splits=5)
    results = {}

    run_id = None
    if tracker:
        run_id = tracker.start_run(
            run_name="parkinsons_telemonitoring_5fold",
            params={"cohort": "UCI_Parkinsons_Telemonitoring", "n_patients": len(set(groups)), "n_recordings": len(df), "target": "delta_motor_updrs_ge_3"},
            tags={"domain": "neurology", "modality": "vocal_telemonitoring"},
            output_dir=output_dir,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "models").mkdir(parents=True, exist_ok=True)

    for name, clf in models.items():
        oof_preds = np.zeros(len(df))
        oof_probs = np.zeros(len(df))

        pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", clf)])

        for train_idx, val_idx in sgkf.split(X, y, groups=groups):
            # Assert zero subject leakage across folds
            assert len(set(groups[train_idx]).intersection(set(groups[val_idx]))) == 0

            X_tr, y_tr = X.iloc[train_idx], y[train_idx]
            X_vl = X.iloc[val_idx]

            pipeline.fit(X_tr, y_tr)
            probs = pipeline.predict_proba(X_vl)[:, 1]
            oof_probs[val_idx] = probs
            oof_preds[val_idx] = (probs >= 0.5).astype(int)

        acc = float(accuracy_score(y, oof_preds))
        roc_auc, roc_low, roc_high = compute_bootstrap_ci(y, oof_probs, roc_auc_score)
        pr_auc, pr_low, pr_high = compute_bootstrap_ci(y, oof_probs, pr_auc_metric)
        brier = float(brier_score_loss(y, oof_probs))

        # Fit final pipeline on all data and save
        pipeline.fit(X, y)
        model_filename = f"{name.lower().replace(' ', '_')}_model.pkl"
        model_path = output_dir / "models" / model_filename
        with open(model_path, "wb") as f:
            pickle.dump(pipeline, f)

        if tracker and run_id:
            tracker.log_metrics(
                {
                    f"{name}_roc_auc": roc_auc,
                    f"{name}_pr_auc": pr_auc,
                    f"{name}_brier": brier,
                    f"{name}_acc": acc,
                },
                run_id=run_id,
            )
            tracker.log_artifact(name, model_path, run_id=run_id)

        results[name] = {
            "accuracy": acc,
            "roc_auc": roc_auc,
            "roc_auc_ci": [roc_low, roc_high],
            "pr_auc": pr_auc,
            "pr_auc_ci": [pr_low, pr_high],
            "brier_score": brier,
        }

    if tracker and run_id:
        tracker.end_run(status="COMPLETED", run_id=run_id)

    return results


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    start_time = time.time()

    tracker = LocalExperimentTracker(
        db_path="results/real_world_training/experiments.db",
        experiment_name="real_world_clinical_validation",
    )

    print("================================================================================")
    print("        AiMedRes: Real-World Clinical Dataset Training & Validation             ")
    print("================================================================================")

    # 1. OASIS-2
    oasis_res = train_oasis2_cohort(tracker=tracker)

    # 2. Parkinson's Telemonitoring
    park_res = train_parkinsons_telemonitoring_cohort(tracker=tracker)

    # Save summary json
    summary_path = Path("results/real_world_training/real_world_benchmark_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"oasis_2": oasis_res, "parkinsons_telemonitoring": park_res}, f, indent=2)

    elapsed = time.time() - start_time

    print("\n================================================================================")
    print("                     REAL-WORLD CLINICAL EVALUATION RESULTS                    ")
    print("================================================================================")
    print("\n[COHORT 1] OASIS-2 Longitudinal Dementia MRI (373 visits, 150 human subjects):")
    print(f"  Target: Clinical Dementia Diagnosis (Zero-Leakage GroupKFold on Subject ID)")
    for m, mdict in oasis_res.items():
        ci_str = f"[{mdict['roc_auc_ci'][0]:.3f} - {mdict['roc_auc_ci'][1]:.3f}]"
        print(f"  * {m:20s} | ROC-AUC: {mdict['roc_auc']:.3f} {ci_str} | PR-AUC: {mdict['pr_auc']:.3f} | Brier: {mdict['brier_score']:.3f}")

    print("\n[COHORT 2] UCI Parkinson's Telemonitoring (5,875 recordings, 42 patients):")
    print(f"  Target: Longitudinal Motor Progression (Delta motor_UPDRS >= 3.0 points)")
    for m, mdict in park_res.items():
        ci_str = f"[{mdict['roc_auc_ci'][0]:.3f} - {mdict['roc_auc_ci'][1]:.3f}]"
        print(f"  * {m:20s} | ROC-AUC: {mdict['roc_auc']:.3f} {ci_str} | PR-AUC: {mdict['pr_auc']:.3f} | Brier: {mdict['brier_score']:.3f}")

    print("\n================================================================================")
    print(f"Total Real-World Training Time: {elapsed:.2f} seconds")
    print(f"Trained Models and Audit Records Saved To: {Path('results/real_world_training').resolve()}")
    print("================================================================================")


if __name__ == "__main__":
    main()
