"""
AiMedRes Zero-Wait Open-Access Dataset Connectors & Leakage-Safe Loaders.

This module provides direct, reproducible access to peer-reviewed, free-to-public
medical research datasets (OASIS-2 for Alzheimer's/Neuroimaging, PRO-ACT for ALS,
PTB-XL for Cardiovascular ECG).

Each loader enforces:
1. Participant-level grouping (zero leakage across longitudinal visits).
2. Elimination of circular target leakage (e.g., dropping CDR when predicting Dementia).
3. Zero preprocessing leakage isolation.
4. Seamless offline fallback to calibrated synthetic benchmark fixtures.
"""

from __future__ import annotations

import logging
import os
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

logger = logging.getLogger(__name__)

# Base directories
REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data"
SYNTHETIC_DIR = DATA_DIR / "synthetic"
RAW_DATA_DIR = DATA_DIR / "raw"
REAL_WORLD_DIR = DATA_DIR / "real_world"

# Public mirror URLs (Zero-wait open access)
OASIS_PUBLIC_MIRROR_URL = (
    "https://raw.githubusercontent.com/V1B3hR/AiMedRes/main/data/synthetic/synthetic_oasis_longitudinal.csv"
)


@dataclass
class DatasetContainer:
    """Standardized container for clinical research datasets."""

    name: str
    df: pd.DataFrame
    features: pd.DataFrame
    target: pd.Series
    groups: np.ndarray
    feature_names: List[str]
    target_name: str
    group_name: str
    is_synthetic: bool
    source_path: str
    metadata: Dict[str, Any] = field(default_factory=dict)


def get_leakage_safe_cv(
    groups: np.ndarray,
    n_splits: int = 5,
    y: Optional[np.ndarray] = None,
    random_state: int = 42,
) -> Union[StratifiedGroupKFold, GroupKFold]:
    """
    Returns a leakage-safe cross-validation splitter that guarantees
    all longitudinal visits for a patient remain within the same fold.
    """
    unique_groups = len(np.unique(groups))
    effective_splits = min(n_splits, unique_groups)
    if effective_splits < n_splits:
        logger.warning(
            f"Requested {n_splits} splits but only {unique_groups} unique subjects found. "
            f"Using {effective_splits} splits."
        )

    if y is not None:
        try:
            return StratifiedGroupKFold(
                n_splits=effective_splits, shuffle=True, random_state=random_state
            )
        except Exception:
            return GroupKFold(n_splits=effective_splits)
    return GroupKFold(n_splits=effective_splits)


class OASISDataset:
    """
    Loader for the Open Access Series of Imaging Studies (OASIS-2) Longitudinal Dataset.
    Marcus et al., 2010. Neuroimaging study of normal aging and Alzheimer's Disease.
    """

    DEFAULT_RAW_FILE = RAW_DATA_DIR / "oasis_longitudinal.csv"
    DEFAULT_SYNTHETIC_FILE = SYNTHETIC_DIR / "synthetic_oasis_longitudinal.csv"

    @classmethod
    def load(
        cls,
        data_path: Optional[Union[str, Path]] = None,
        target_col: str = "Group",
        binary_dementia_only: bool = True,
        drop_circular_cdr: bool = True,
    ) -> DatasetContainer:
        """
        Load and validate the OASIS longitudinal dataset.

        Args:
            data_path: Path to local CSV file. If None, checks data/raw/ then data/synthetic/.
            target_col: Prediction target ('Group' or 'CDR').
            binary_dementia_only: Convert 'Group' to binary (0=Nondemented, 1=Demented/Converted).
            drop_circular_cdr: Drop Clinical Dementia Rating (CDR) from input features to prevent circular leakage.
        """
        resolved_path, is_synthetic = cls._resolve_path(data_path)
        df = pd.read_csv(resolved_path)
        logger.info(
            f"Loaded OASIS dataset from {resolved_path} (synthetic={is_synthetic}, shape={df.shape})"
        )

        group_col = "Subject ID"
        if group_col not in df.columns:
            # Fallback column search
            for candidate in ["subject_id", "Patient_ID", "ID", "SubjectID"]:
                if candidate in df.columns:
                    group_col = candidate
                    break

        if group_col not in df.columns:
            raise ValueError(f"Participant grouping column '{group_col}' missing from OASIS table.")

        # Setup target
        if target_col == "Group":
            if binary_dementia_only:
                # 0 for Nondemented, 1 for Demented or Converted
                y = df["Group"].apply(lambda g: 0 if str(g).strip().lower() == "nondemented" else 1)
            else:
                y = df["Group"].astype("category").cat.codes
        elif target_col in df.columns:
            y = df[target_col]
        else:
            raise KeyError(f"Target column '{target_col}' not found in OASIS dataframe.")

        # Exclude IDs, target, and circular leakage features
        drop_cols = [group_col, "Group", "MRI ID"]
        if drop_circular_cdr and "CDR" in df.columns and target_col != "CDR":
            drop_cols.append("CDR")
            logger.info("Audit safeguard: Dropped 'CDR' from features to prevent target circularity.")

        # Also drop non-numeric / string identifiers
        for col in ["Hand"]:
            if col in df.columns:
                drop_cols.append(col)

        feature_cols = [c for c in df.columns if c not in drop_cols]
        X = df[feature_cols].copy()

        # Clean/encode categorical demographics
        if "M/F" in X.columns:
            X["M/F"] = X["M/F"].apply(lambda s: 1 if str(s).strip().upper() == "M" else 0)

        groups = df[group_col].values

        return DatasetContainer(
            name="OASIS-2 Longitudinal Neuroimaging",
            df=df,
            features=X,
            target=y,
            groups=groups,
            feature_names=list(X.columns),
            target_name=target_col,
            group_name=group_col,
            is_synthetic=is_synthetic,
            source_path=str(resolved_path),
            metadata={
                "n_records": len(df),
                "n_unique_subjects": len(np.unique(groups)),
                "binary_dementia": binary_dementia_only,
                "dropped_circular_features": ["CDR"] if drop_circular_cdr else [],
            },
        )

    @classmethod
    def _resolve_path(cls, user_path: Optional[Union[str, Path]]) -> Tuple[Path, bool]:
        if user_path is not None:
            p = Path(user_path)
            if p.exists():
                return p, False
            raise FileNotFoundError(f"Specified OASIS path does not exist: {p}")

        if cls.DEFAULT_RAW_FILE.exists():
            return cls.DEFAULT_RAW_FILE, False

        real_world_file = REAL_WORLD_DIR / "oasis_longitudinal.csv"
        if real_world_file.exists():
            return real_world_file, False

        if cls.DEFAULT_SYNTHETIC_FILE.exists():
            return cls.DEFAULT_SYNTHETIC_FILE, True

        raise FileNotFoundError(
            "No OASIS dataset found. Place raw 'oasis_longitudinal.csv' in data/raw/ "
            "or generate synthetic fixture via scripts/generate_open_fixtures.py."
        )


class PROACTALSDataset:
    """
    Loader for the Pooled Resource Open-Access ALS Clinical Trials (PRO-ACT) Database.
    Longitudinal motor function (ALSFRS-R), respiratory vitals, and disease progression.
    """

    DEFAULT_RAW_FILE = RAW_DATA_DIR / "proact_als.csv"
    DEFAULT_SYNTHETIC_FILE = SYNTHETIC_DIR / "synthetic_proact_als.csv"

    @classmethod
    def load(
        cls,
        data_path: Optional[Union[str, Path]] = None,
        target_col: str = "Fast_Progression",
        drop_progression_derivatives: bool = True,
    ) -> DatasetContainer:
        """
        Load and validate the PRO-ACT ALS clinical trials dataset.
        """
        resolved_path, is_synthetic = cls._resolve_path(data_path)
        df = pd.read_csv(resolved_path)
        logger.info(
            f"Loaded PRO-ACT ALS dataset from {resolved_path} (synthetic={is_synthetic}, shape={df.shape})"
        )

        group_col = "Subject_ID"
        if group_col not in df.columns:
            for candidate in ["subject_id", "Patient_ID", "ID", "PatientID"]:
                if candidate in df.columns:
                    group_col = candidate
                    break

        if group_col not in df.columns:
            raise ValueError(f"Participant grouping column '{group_col}' missing from PRO-ACT table.")

        if target_col not in df.columns:
            raise KeyError(f"Target column '{target_col}' not found in PRO-ACT dataframe.")

        y = df[target_col]

        # Prevent circular leakage: drop progression rate from features if predicting progression
        drop_cols = [group_col, target_col]
        if drop_progression_derivatives:
            for deriv in ["Progression_Slope", "Progression_Rate", "Fast_Progression"]:
                if deriv in df.columns and deriv not in drop_cols:
                    drop_cols.append(deriv)
                    logger.info(f"Audit safeguard: Dropped '{deriv}' to prevent target leakage.")

        feature_cols = [c for c in df.columns if c not in drop_cols]
        X = df[feature_cols].copy()

        # Encode categorical columns
        if "Gender" in X.columns:
            X["Gender"] = X["Gender"].apply(lambda g: 1 if str(g).strip().lower() == "male" else 0)
        if "Site_of_Onset" in X.columns:
            X["Site_of_Onset"] = X["Site_of_Onset"].apply(
                lambda s: 1 if str(s).strip().lower() == "bulbar" else 0
            )

        groups = df[group_col].values

        return DatasetContainer(
            name="PRO-ACT Open-Access ALS Clinical Trials",
            df=df,
            features=X,
            target=y,
            groups=groups,
            feature_names=list(X.columns),
            target_name=target_col,
            group_name=group_col,
            is_synthetic=is_synthetic,
            source_path=str(resolved_path),
            metadata={
                "n_records": len(df),
                "n_unique_subjects": len(np.unique(groups)),
                "target": target_col,
            },
        )

    @classmethod
    def _resolve_path(cls, user_path: Optional[Union[str, Path]]) -> Tuple[Path, bool]:
        if user_path is not None:
            p = Path(user_path)
            if p.exists():
                return p, False
            raise FileNotFoundError(f"Specified PRO-ACT path does not exist: {p}")

        if cls.DEFAULT_RAW_FILE.exists():
            return cls.DEFAULT_RAW_FILE, False

        if cls.DEFAULT_SYNTHETIC_FILE.exists():
            return cls.DEFAULT_SYNTHETIC_FILE, True

        raise FileNotFoundError(
            "No PRO-ACT ALS dataset found. Place raw 'proact_als.csv' in data/raw/ "
            "or generate synthetic fixture via scripts/generate_open_fixtures.py."
        )


class PTBXLECGDataset:
    """
    Loader for the PhysioNet PTB-XL ECG Diagnostic Database.
    Wagner et al., 2020. CC-BY 4.0 Open Access.
    """

    DEFAULT_RAW_FILE = RAW_DATA_DIR / "ptbxl_ecg.csv"
    DEFAULT_SYNTHETIC_FILE = SYNTHETIC_DIR / "synthetic_ptbxl_ecg.csv"

    @classmethod
    def load(
        cls,
        data_path: Optional[Union[str, Path]] = None,
        target_col: str = "is_abnormal",
    ) -> DatasetContainer:
        """
        Load and validate the PTB-XL ECG dataset.
        """
        resolved_path, is_synthetic = cls._resolve_path(data_path)
        df = pd.read_csv(resolved_path)
        logger.info(
            f"Loaded PTB-XL ECG dataset from {resolved_path} (synthetic={is_synthetic}, shape={df.shape})"
        )

        group_col = "patient_id"
        if group_col not in df.columns:
            group_col = "ecg_id"  # fallback if single scan per patient

        if target_col not in df.columns:
            raise KeyError(f"Target column '{target_col}' not found in PTB-XL dataframe.")

        y = df[target_col]

        drop_cols = ["ecg_id", "patient_id", target_col]
        if "diagnostic_class" in df.columns and target_col != "diagnostic_class":
            drop_cols.append("diagnostic_class")

        feature_cols = [c for c in df.columns if c not in drop_cols]
        X = df[feature_cols].copy()
        groups = df[group_col].values

        return DatasetContainer(
            name="PhysioNet PTB-XL ECG Database",
            df=df,
            features=X,
            target=y,
            groups=groups,
            feature_names=list(X.columns),
            target_name=target_col,
            group_name=group_col,
            is_synthetic=is_synthetic,
            source_path=str(resolved_path),
            metadata={
                "n_records": len(df),
                "n_unique_patients": len(np.unique(groups)),
                "target": target_col,
            },
        )

    @classmethod
    def _resolve_path(cls, user_path: Optional[Union[str, Path]]) -> Tuple[Path, bool]:
        if user_path is not None:
            p = Path(user_path)
            if p.exists():
                return p, False
            raise FileNotFoundError(f"Specified PTB-XL path does not exist: {p}")

        if cls.DEFAULT_RAW_FILE.exists():
            return cls.DEFAULT_RAW_FILE, False

        if cls.DEFAULT_SYNTHETIC_FILE.exists():
            return cls.DEFAULT_SYNTHETIC_FILE, True

class ParkinsonsTelemonitoringDataset:
    """
    Loader for the Oxford / UCI Parkinson's Telemonitoring Dataset.
    Tsanas et al., 2010. Accurate telemonitoring of Parkinson's disease progression
    by noninvasive speech tests. (5,875 recordings across 42 patients).
    """

    DEFAULT_RAW_FILE = RAW_DATA_DIR / "parkinsons_telemonitoring.csv"
    REAL_WORLD_FILE = REAL_WORLD_DIR / "parkinsons_telemonitoring.csv"
    DEFAULT_SYNTHETIC_FILE = SYNTHETIC_DIR / "synthetic_parkinsons_telemonitoring.csv"

    @classmethod
    def load(
        cls,
        data_path: Optional[Union[str, Path]] = None,
        target_col: str = "delta_motor_progression",
        progression_threshold: float = 3.0,
    ) -> DatasetContainer:
        """
        Load and validate the Parkinson's Telemonitoring dataset.

        Args:
            data_path: Path to local CSV file.
            target_col: Target label ('delta_motor_progression' or continuous motor score).
            progression_threshold: Threshold in UPDRS points for meaningful motor progression.
        """
        resolved_path, is_synthetic = cls._resolve_path(data_path)
        df = pd.read_csv(resolved_path)
        logger.info(
            f"Loaded Parkinson's Telemonitoring dataset from {resolved_path} (synthetic={is_synthetic}, shape={df.shape})"
        )

        group_col = "subject#"
        if group_col not in df.columns:
            for candidate in ["subject_id", "Patient_ID", "ID", "subject"]:
                if candidate in df.columns:
                    group_col = candidate
                    break

        if group_col not in df.columns:
            raise ValueError(f"Participant grouping column '{group_col}' missing from Parkinson's table.")

        groups = df[group_col].values

        # Calculate subject-specific baseline motor_UPDRS (at earliest test_time)
        baseline_motor = df.groupby(group_col).apply(
            lambda g: g.sort_values("test_time").iloc[0]["motor_UPDRS"]
        ).to_dict()
        df["baseline_motor"] = df[group_col].map(baseline_motor)
        df["delta_motor"] = df["motor_UPDRS"] - df["baseline_motor"]

        # Target: Meaningful motor progression milestone
        if target_col == "delta_motor_progression":
            y = (df["delta_motor"] >= progression_threshold).astype(int)
        elif target_col in df.columns:
            y = df[target_col]
        else:
            raise KeyError(f"Target column '{target_col}' not found in Parkinson's dataframe.")

        # Exclude IDs, raw UPDRS scores, and baseline to prevent circular target leakage
        drop_cols = [
            group_col,
            "motor_UPDRS",
            "total_UPDRS",
            "baseline_motor",
            "delta_motor",
        ]
        feature_cols = [c for c in df.columns if c not in drop_cols]
        X = df[feature_cols].copy()

        return DatasetContainer(
            name="UCI Parkinson's Vocal Telemonitoring",
            df=df,
            features=X,
            target=y,
            groups=groups,
            feature_names=list(X.columns),
            target_name=target_col,
            group_name=group_col,
            is_synthetic=is_synthetic,
            source_path=str(resolved_path),
            metadata={
                "n_records": len(df),
                "n_unique_patients": len(np.unique(groups)),
                "target": target_col,
                "progression_threshold": progression_threshold,
            },
        )

    @classmethod
    def _resolve_path(cls, user_path: Optional[Union[str, Path]]) -> Tuple[Path, bool]:
        if user_path is not None:
            p = Path(user_path)
            if p.exists():
                return p, False
            raise FileNotFoundError(f"Specified Parkinson's telemonitoring path does not exist: {p}")

        if cls.DEFAULT_RAW_FILE.exists():
            return cls.DEFAULT_RAW_FILE, False

        if cls.REAL_WORLD_FILE.exists():
            return cls.REAL_WORLD_FILE, False

        if cls.DEFAULT_SYNTHETIC_FILE.exists():
            return cls.DEFAULT_SYNTHETIC_FILE, True

        raise FileNotFoundError("No Parkinson's Telemonitoring dataset found.")


def get_open_dataset(name: str, **kwargs) -> DatasetContainer:
    """Convenience factory function for loading open-access datasets."""
    canonical = name.strip().lower()
    if "oasis" in canonical or "alzheimer" in canonical:
        return OASISDataset.load(**kwargs)
    elif "proact" in canonical or "als" in canonical:
        return PROACTALSDataset.load(**kwargs)
    elif "ptb" in canonical or "ecg" in canonical or "cardio" in canonical:
        return PTBXLECGDataset.load(**kwargs)
    elif "parkinson" in canonical or "telemonitoring" in canonical or "updrs" in canonical:
        return ParkinsonsTelemonitoringDataset.load(**kwargs)
    else:
        raise ValueError(
            f"Unknown open dataset '{name}'. Available: 'oasis', 'proact_als', 'ptbxl_ecg', 'parkinsons_telemonitoring'."
        )
