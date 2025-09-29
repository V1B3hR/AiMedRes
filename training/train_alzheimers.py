#!/usr/bin/env python3
"""
Enhanced Alzheimer's Disease Classification Pipeline with:
- Ensemble methods (Random Forest, XGBoost, LightGBM)
- Feature engineering (Polynomial & interaction features)
- Model tuning (GridSearchCV for hyperparameter optimization)
"""

from __future__ import annotations

import os
import sys
import json
import pickle
import random
import logging
import argparse
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

warnings.filterwarnings("ignore")

def _try_import(module_name: str):
    try:
        return __import__(module_name), True
    except ImportError:
        return None, False

xgb, XGBOOST_AVAILABLE = _try_import("xgboost")
lgb, LIGHTGBM_AVAILABLE = _try_import("lightgbm")
mlflow, MLFLOW_AVAILABLE = _try_import("mlflow")
yaml, YAML_AVAILABLE = _try_import("yaml")
kagglehub, KAGGLEHUB_AVAILABLE = _try_import("kagglehub")
plt, MPL_AVAILABLE = _try_import("matplotlib.pyplot")
imblearn, IMBLEARN_AVAILABLE = _try_import("imblearn")

if IMBLEARN_AVAILABLE:
    from imblearn.over_sampling import SMOTE

# -------- Configuration -------- #
DEFAULT_CONFIG = {
    'output_dir': 'alzheimer_outputs_enhanced',
    'dataset_handle': 'rabieelkharoua/alzheimers-disease-dataset',
    'target_column': 'Diagnosis',
    'folds': 5,
    'random_seed': 42,
    'use_smote': True,
    'smote_k_neighbors': 5,
    'run_voting_ensemble': True,
    'calibration_plots': True,
    'mlflow_experiment_name': 'Alzheimer_Classification_Enhanced',
    'model_selection_metric': 'f1_macro',
    'enable_polynomial_features': True,
    'poly_degree': 2,
    'poly_interaction_only': False,
    'model_tuning': True
}

def load_config(path: str) -> Dict[str, Any]:
    if not Path(path).exists():
        if not YAML_AVAILABLE:
            raise ImportError("PyYAML required to create default config.")
        with open(path, 'w') as f:
            yaml.dump(DEFAULT_CONFIG, f)
        return DEFAULT_CONFIG
    if not YAML_AVAILABLE:
        raise ImportError("PyYAML required to parse config.")
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def init_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("alz_pipeline")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s")
    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def download_dataset(handle: str) -> Path:
    if not KAGGLEHUB_AVAILABLE:
        raise ImportError("Install kagglehub to download dataset.")
    path = kagglehub.dataset_download(handle)
    csvs = list(Path(path).glob("*.csv"))
    if not csvs:
        raise FileNotFoundError("No CSV files found in downloaded dataset.")
    return csvs[0]

def load_data(cfg: Dict[str, Any]) -> pd.DataFrame:
    csv_path = download_dataset(cfg['dataset_handle'])
    return pd.read_csv(csv_path)

def preprocess(df: pd.DataFrame, target_col: str, cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.Series, LabelEncoder]:
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")
    id_cols = [c for c in df.columns if 'id' in c.lower()]
    df = df.drop(columns=id_cols).dropna(subset=[target_col])
    X = df.drop(columns=[target_col])
    y_raw = df[target_col]
    le = LabelEncoder()
    y = pd.Series(le.fit_transform(y_raw), name='target')

    # Feature engineering: Polynomial & interaction features
    if cfg.get('enable_polynomial_features', False):
        num_cols = X.select_dtypes(include=np.number).columns.tolist()
        X_num = X[num_cols]
        poly = PolynomialFeatures(
            degree=cfg.get('poly_degree', 2),
            interaction_only=cfg.get('poly_interaction_only', False),
            include_bias=False
        )
        X_poly = pd.DataFrame(
            poly.fit_transform(X_num),
            columns=poly.get_feature_names_out(num_cols)
        )
        # Replace numeric columns with polynomial features
        X = X.drop(columns=num_cols)
        X = pd.concat([X, X_poly], axis=1)

    return X, y, le

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    try:
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)

    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', ohe)
    ])

    ct = ColumnTransformer([
        ('num', num_pipe, num_cols),
        ('cat', cat_pipe, cat_cols)
    ], remainder='drop')
    return ct

def get_models(cfg: Dict[str, Any]) -> Dict[str, Any]:
    models: Dict[str, Any] = {}
    # Random Forest
    models['RandomForest'] = RandomForestClassifier(
        random_state=cfg['random_seed'],
        n_jobs=-1
    )
    # XGBoost
    if XGBOOST_AVAILABLE:
        models['XGBoost'] = xgb.XGBClassifier(
            random_state=cfg['random_seed'],
            eval_metric='logloss',
            n_jobs=-1
        )
    # LightGBM
    if LIGHTGBM_AVAILABLE:
        models['LightGBM'] = lgb.LGBMClassifier(
            random_state=cfg['random_seed'],
            n_jobs=-1
        )
    return models

def get_param_grids():
    grids = {
        'RandomForest': {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [None, 5, 10],
            'classifier__min_samples_split': [2, 5]
        },
        'XGBoost': {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [3, 5, 7],
            'classifier__learning_rate': [0.01, 0.1]
        },
        'LightGBM': {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [-1, 5, 10],
            'classifier__learning_rate': [0.01, 0.1]
        }
    }
    return grids

def compute_metrics(y_true, y_pred, y_proba):
    n_classes = y_proba.shape[1]
    out = {
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'f1_macro': f1_score(y_true, y_pred, average='macro')
    }
    try:
        if n_classes == 2:
            out['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
        else:
            out['roc_auc_ovr'] = roc_auc_score(y_true, y_proba, multi_class='ovr')
    except Exception as e:
        out['roc_auc_error'] = str(e)
    return out

def plot_confusion(model_name: str, y_true, y_pred, out_dir: Path):
    if not MPL_AVAILABLE:
        return
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    path = out_dir / f'cm_{model_name}.png'
    plt.savefig(path)
    plt.close()
    return path

def plot_calibration(model_name: str, y_true, y_proba, out_dir: Path):
    if not MPL_AVAILABLE:
        return
    if y_proba.shape[1] != 2:
        return
    from sklearn.calibration import CalibrationDisplay
    fig, ax = plt.subplots()
    CalibrationDisplay.from_predictions(y_true, y_proba[:, 1], n_bins=10, ax=ax, name=model_name)
    ax.set_title(f'Calibration - {model_name}')
    path = out_dir / f'calibration_{model_name}.png'
    fig.savefig(path)
    plt.close(fig)
    return path

def cross_validate_model(
    name: str,
    model,
    X: pd.DataFrame,
    y: pd.Series,
    preprocessor: ColumnTransformer,
    cfg: Dict[str, Any],
    logger: logging.Logger,
    param_grid=None
) -> Dict[str, Any]:
    folds = cfg['folds']
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=cfg['random_seed'])
    fold_metrics: List[Dict[str, float]] = []
    oof_pred = np.zeros(len(y), dtype=int)
    oof_proba_list: List[np.ndarray] = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        # Fit preprocessor per fold (unbiased)
        preprocessor_fold = pickle.loads(pickle.dumps(preprocessor))
        X_tr_tf = preprocessor_fold.fit_transform(X_tr)
        X_va_tf = preprocessor_fold.transform(X_va)

        # SMOTE (numeric-only matrix already)
        if cfg['use_smote'] and IMBLEARN_AVAILABLE:
            unique, counts = np.unique(y_tr, return_counts=True)
            min_count = counts.min()
            k_neighbors = min(cfg.get('smote_k_neighbors', 5), max(1, min_count - 1))
            sm = SMOTE(random_state=cfg['random_seed'], k_neighbors=k_neighbors)
            X_tr_tf, y_tr = sm.fit_resample(X_tr_tf, y_tr)

        # Model tuning with GridSearchCV
        if param_grid and cfg.get('model_tuning', False):
            pipe = Pipeline([
                ('preprocessor', preprocessor_fold),
                ('classifier', pickle.loads(pickle.dumps(model)))
            ])
            grid = GridSearchCV(pipe, param_grid, cv=3, n_jobs=-1, scoring='f1_macro')
            grid.fit(X_tr, y_tr)
            best_clf = grid.best_estimator_.named_steps['classifier']
            logger.info(f"[{name}] Fold {fold+1} best params: {grid.best_params_}")
        else:
            best_clf = pickle.loads(pickle.dumps(model))
            best_clf.fit(X_tr_tf, y_tr)

        y_va_pred = best_clf.predict(X_va_tf)
        if hasattr(best_clf, "predict_proba"):
            y_va_proba = best_clf.predict_proba(X_va_tf)
        else:
            y_va_proba = np.zeros((len(y_va_pred), len(np.unique(y))))
            for i, p in enumerate(y_va_pred):
                y_va_proba[i, p] = 1.0

        metrics = compute_metrics(y_va, y_va_pred, y_va_proba)
        fold_metrics.append(metrics)
        oof_pred[va_idx] = y_va_pred
        oof_proba_list.append(y_va_proba)

        logger.info(f"[{name}] Fold {fold+1}/{folds} "
                    + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items() if not k.endswith("_error")))

    # Aggregate OOF
    oof_proba = np.zeros((len(y), oof_proba_list[0].shape[1]))
    for (tr_idx, va_idx), proba in zip(skf.split(X, y), oof_proba_list):
        oof_proba[va_idx] = proba

    final_metrics = compute_metrics(y, oof_pred, oof_proba)
    return {
        'fold_metrics': fold_metrics,
        'oof_metrics': final_metrics,
        'oof_pred': oof_pred,
        'oof_proba': oof_proba
    }

def fit_full_model(
    name: str,
    model,
    X: pd.DataFrame,
    y: pd.Series,
    preprocessor: ColumnTransformer,
    cfg: Dict[str, Any],
    param_grid=None
):
    preprocessor_full = pickle.loads(pickle.dumps(preprocessor))
    X_tf = preprocessor_full.fit_transform(X)

    if cfg['use_smote'] and IMBLEARN_AVAILABLE:
        unique, counts = np.unique(y, return_counts=True)
        min_count = counts.min()
        k_neighbors = min(cfg.get('smote_k_neighbors', 5), max(1, min_count - 1))
        sm = SMOTE(random_state=cfg['random_seed'], k_neighbors=k_neighbors)
        X_tf, y = sm.fit_resample(X_tf, y)

    # Model tuning with GridSearchCV
    if param_grid and cfg.get('model_tuning', False):
        pipe = Pipeline([
            ('preprocessor', preprocessor_full),
            ('classifier', pickle.loads(pickle.dumps(model)))
        ])
        grid = GridSearchCV(pipe, param_grid, cv=3, n_jobs=-1, scoring='f1_macro')
        grid.fit(X, y)
        best_clf = grid.best_estimator_.named_steps['classifier']
    else:
        best_clf = pickle.loads(pickle.dumps(model))
        best_clf.fit(X_tf, y)

    artifact = {
        'preprocessor': preprocessor_full,
        'classifier': best_clf
    }
    return artifact

def pipeline_predict(artifact: Dict[str, Any], X: pd.DataFrame):
    pp = artifact['preprocessor']
    clf = artifact['classifier']
    X_tf = pp.transform(X)
    pred = clf.predict(X_tf)
    proba = clf.predict_proba(X_tf) if hasattr(clf, "predict_proba") else None
    return pred, proba

def build_ensemble(artifacts: Dict[str, Dict[str, Any]], cfg: Dict[str, Any]):
    estimators = []
    for name, art in artifacts.items():
        pipe = Pipeline([
            ('preprocessor', art['preprocessor']),
            ('classifier', art['classifier'])
        ])
        estimators.append((name, pipe))
    ensemble = VotingClassifier(estimators=estimators, voting='soft')
    return ensemble

def save_pickle(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def main(config_path: str):
    cfg = load_config(config_path)
    out_dir = Path(cfg['output_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = out_dir / 'logs'
    log_dir.mkdir(exist_ok=True)
    logger = init_logger(log_dir / 'run.log')
    set_seed(cfg['random_seed'])

    if MLFLOW_AVAILABLE:
        mlflow.set_experiment(cfg['mlflow_experiment_name'])
        mlflow.start_run()
        mlflow.log_params({k: v for k, v in cfg.items() if not isinstance(v, dict)})

    logger.info("=== Pipeline Start ===")
    df = load_data(cfg)
    X, y, label_encoder = preprocess(df, cfg['target_column'], cfg)
    n_classes = len(np.unique(y))
    logger.info(f"Data shape: X={X.shape}, y={y.shape}, classes={n_classes}")

    save_pickle(label_encoder, out_dir / 'preprocessors' / 'label_encoder.pkl')

    base_preprocessor = build_preprocessor(X)
    models = get_models(cfg)
    param_grids = get_param_grids()
    cv_results = {}
    model_artifacts = {}

    vis_dir = out_dir / 'visualizations'
    vis_dir.mkdir(exist_ok=True)

    for name, model in models.items():
        logger.info(f"---- CV {name} ----")
        param_grid = param_grids[name] if name in param_grids else None
        res = cross_validate_model(name, model, X, y, base_preprocessor, cfg, logger, param_grid)
        cv_results[name] = {
            'oof_metrics': res['oof_metrics'],
            'fold_metrics': res['fold_metrics']
        }

        paths_to_log = []
        cm_path = plot_confusion(name, y, res['oof_pred'], vis_dir)
        if cm_path:
            paths_to_log.append(cm_path)
        if cfg.get('calibration_plots', True):
            cal_path = plot_calibration(name, y, res['oof_proba'], vis_dir)
            if cal_path:
                paths_to_log.append(cal_path)
        if MLFLOW_AVAILABLE:
            for p in paths_to_log:
                mlflow.log_artifact(str(p))
            for i, foldm in enumerate(res['fold_metrics']):
                mlflow.log_metrics({f"{name}_fold{i+1}_{k}": v for k, v in foldm.items() if not k.endswith("_error")})

        artifact = fit_full_model(name, model, X, y, base_preprocessor, cfg, param_grid)
        model_artifacts[name] = artifact
        save_pickle(artifact, out_dir / 'models' / f'{name}.pkl')

    selection_metric = cfg['model_selection_metric']
    best_name = max(
        cv_results.keys(),
        key=lambda m: cv_results[m]['oof_metrics'].get(selection_metric, -np.inf)
    )
    best_metrics = cv_results[best_name]['oof_metrics']
    logger.info(f"Selected best model: {best_name} ({selection_metric}={best_metrics.get(selection_metric):.4f})")
    with open(out_dir / 'models' / 'best_model.json', 'w') as f:
        json.dump({'best_model': best_name, 'metric': selection_metric, 'metrics': best_metrics}, f, indent=2)

    if cfg.get('run_voting_ensemble', True) and len(model_artifacts) > 1:
        logger.info("---- Building Ensemble ----")
        ensemble = build_ensemble(model_artifacts, cfg)
        ensemble.fit(X, y)
        save_pickle(ensemble, out_dir / 'models' / 'voting_ensemble.pkl')
        skf = StratifiedKFold(n_splits=cfg['folds'], shuffle=True, random_state=cfg['random_seed'])
        ens_pred = np.zeros(len(y), dtype=int)
        ens_proba = []
        for tr_idx, va_idx in skf.split(X, y):
            X_va = X.iloc[va_idx]
            ens_pred[va_idx] = ensemble.predict(X_va)
            ens_proba.append(ensemble.predict_proba(X_va))
        ens_proba_full = np.zeros((len(y), ens_proba[0].shape[1]))
        for (tr_idx, va_idx), proba in zip(skf.split(X, y), ens_proba):
            ens_proba_full[va_idx] = proba
        ens_metrics = compute_metrics(y, ens_pred, ens_proba_full)
        cv_results['Ensemble'] = {'oof_metrics': ens_metrics, 'fold_metrics': None}
        logger.info("Ensemble metrics: " + ", ".join(f"{k}={v:.4f}" for k, v in ens_metrics.items() if not k.endswith('_error')))
        if MLFLOW_AVAILABLE:
            mlflow.log_metrics({f"Ensemble_{k}": v for k, v in ens_metrics.items() if not k.endswith('_error')})
            cm_path = plot_confusion("Ensemble", y, ens_pred, vis_dir)
            if cm_path: mlflow.log_artifact(str(cm_path))
            if cfg.get('calibration_plots', True):
                cal_path = plot_calibration("Ensemble", y, ens_proba_full, vis_dir)
                if cal_path: mlflow.log_artifact(str(cal_path))

    report = {
        'timestamp': datetime.utcnow().isoformat(),
        'classes': label_encoder.classes_.tolist(),
        'cv_results': cv_results,
        'best_model': best_name
    }
    (out_dir / 'metrics').mkdir(exist_ok=True, parents=True)
    with open(out_dir / 'metrics' / 'final_report.json', 'w') as f:
        json.dump(report, f, indent=2)

    if MLFLOW_AVAILABLE:
        mlflow.log_artifact(str(out_dir / 'metrics' / 'final_report.json'))
        mlflow.log_metrics({f"{best_name}_selected_{k}": v for k, v in best_metrics.items() if not k.endswith('_error')})
        mlflow.end_run()

    logger.info("=== Pipeline Complete ===")

def load_artifact(path: Path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def predict(raw_df: pd.DataFrame, artifacts_dir: str):
    base = Path(artifacts_dir)
    best_meta_path = base / 'models' / 'best_model.json'
    if best_meta_path.exists():
        meta = json.loads(best_meta_path.read_text())
        best_name = meta['best_model']
    else:
        best_name = 'Ensemble' if (base / 'models' / 'voting_ensemble.pkl').exists() else None

    if best_name == 'Ensemble':
        model_path = base / 'models' / 'voting_ensemble.pkl'
        ensemble = load_artifact(model_path)
        preds = ensemble.predict(raw_df)
        probas = ensemble.predict_proba(raw_df)
    else:
        model_path = base / 'models' / f'{best_name}.pkl'
        artifact = load_artifact(model_path)
        preds, probas = pipeline_predict(artifact, raw_df)

    le_path = base / 'preprocessors' / 'label_encoder.pkl'
    label_encoder = load_artifact(le_path)
    pred_labels = label_encoder.inverse_transform(preds)
    return pred_labels, probas

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yml')
    args = parser.parse_args()
    main(args.config)
