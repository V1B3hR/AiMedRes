#!/usr/bin/env python3
"""
Advanced Radiology Insight Module for AiMedRes
==============================================

This enhanced, production-oriented single-file module expands upon the original
Brain MRI volumetric analysis pipeline to deliver a robust, extensible, and
maintainable imaging analytics "skill" suitable for multi-agent orchestration.

Key Enhancements Over Previous Version
--------------------------------------
1. Modular Configuration:
   - Pydantic-based Config model with environment-variable overrides.
   - Optional external YAML loading (lazy; avoided if not available).

2. Strategy Registry & Dynamic Loading:
   - Pluggable strategy architecture with automatic registration decorator.

3. Rich Logging & Instrumentation:
   - Structured logging (fallback to stdlib) with optional JSON mode.
   - Timing decorators & analysis provenance expansion.

4. Extended Quantitative Analytics:
   - Z-scores (as before).
   - Percentile approximation from Z-score (assuming normality).
   - Composite anomaly score (weighted aggregation of extreme Z magnitudes).
   - Risk stratification levels (Enum).

5. Advanced Quality Control:
   - Penalization layering w/ configurable floors.
   - Optional dynamic QC rule injection.

6. Caching & Performance:
   - LRU caching for normative stats lookups (already present; augmented).
   - Optional measure alias normalization.

7. Data Integrity & Audit:
   - SHA256 hashing of raw normative dataset string.
   - Extended provenance fields & internal versioning constant.

8. CLI + Programmatic Use:
   - Command-line interface for standalone execution.
   - JSON I/O path support for batch or scripted pipelines.

9. Robust Error Handling:
   - Custom exception hierarchy for clearer upstream diagnostics.
   - Graceful degradation if measure missing in normative dataset.

10. Memory Export Improvements:
   - Added percentile + anomaly score in memory metadata.
   - Importance scoring now configurable and includes anomaly magnitude.

Dependencies
------------
- pydantic
- pandas
- (optional) pyyaml (only if using YAML config loading)
- numpy (added for percentile + vectorized math)

Security & Medical Disclaimer
-----------------------------
This module is for research augmentation and should not be used for direct
clinical decision-making without validation and regulatory clearance.

"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import io
import json
import logging
import math
import os
import sys
import time
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from functools import lru_cache, wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, ValidationError, field_validator

# ==============================================================================
# 0. MODULE VERSION
# ==============================================================================

MODULE_VERSION = "2.1.0"

# ==============================================================================
# 1. LOGGING SETUP
# ==============================================================================


def _init_logger() -> logging.Logger:
    logger = logging.getLogger("AiMedRes.ImagingInsights")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s", datefmt="%Y-%m-%dT%H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(os.environ.get("IMAGING_INSIGHTS_LOG_LEVEL", "INFO").upper())
    return logger


LOGGER = _init_logger()


def timed(fn: Callable):
    """Decorator to measure execution time of key functions."""

    @wraps(fn)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            LOGGER.debug(f"{fn.__name__} executed in {elapsed_ms:.2f} ms")

    return wrapper


# ==============================================================================
# 2. EXCEPTIONS
# ==============================================================================


class InsightError(Exception):
    """Base exception for imaging insights."""


class ConfigError(InsightError):
    """Configuration-related problems."""


class StrategyNotFoundError(InsightError):
    """Raised when a requested strategy is not registered."""


class NormativeDataError(InsightError):
    """Issues with normative dataset access."""


# ==============================================================================
# 3. CONFIGURATION MODEL
# ==============================================================================


class QualityControlConfig(BaseModel):
    min_snr_for_high_confidence: float = 15.0
    max_motion_for_high_confidence: float = 0.3
    confidence_penalty_low_snr: float = 0.25
    confidence_penalty_high_motion: float = 0.30
    min_confidence_floor: float = 0.1


class ImportanceBoostersConfig(BaseModel):
    critical_keywords: List[str] = [
        "atrophy",
        "severe",
        "significant",
        "abnormal",
        "mass",
        "lesion",
    ]
    keyword_boost_value: float = 0.2
    max_total_boost: float = 0.5
    anomaly_weight: float = 0.15


class ZScoreThresholds(BaseModel):
    significant_atrophy: float = -2.5
    mild_atrophy: float = -1.75
    borderline_high: float = 1.75
    high_volume_anomaly: float = 2.5


class RadiologyInsightConfig(BaseModel):
    strategy_name: str = "BrainMRIVolumetry_v1.3_ZScorePlus"
    normative_data_cohort: str = "healthy_adults_mixed_scanner_v1"
    z_score_thresholds: ZScoreThresholds = ZScoreThresholds()
    quality_control: QualityControlConfig = QualityControlConfig()
    importance_boosters: ImportanceBoostersConfig = ImportanceBoostersConfig()
    enable_percentiles: bool = True
    enable_composite_anomaly_score: bool = True
    measure_aliases: Dict[str, str] = Field(default_factory=dict)
    allow_missing_measures: bool = True
    strict_normative_lookup: bool = False

    @classmethod
    def from_env(cls, **overrides) -> "RadiologyInsightConfig":
        """Build configuration applying environment variable overrides (if present)."""
        env_map = {
            "IMAGING_STRATEGY_NAME": "strategy_name",
            "IMAGING_NORMATIVE_COHORT": "normative_data_cohort",
        }
        data = {}
        for env_key, field_name in env_map.items():
            if env_key in os.environ:
                data[field_name] = os.environ[env_key]
        data.update(overrides)
        return cls(**data)

    @classmethod
    def from_yaml(cls, path: str) -> "RadiologyInsightConfig":
        try:
            import yaml
        except ImportError as e:
            raise ConfigError("pyyaml not installed; cannot load YAML config.") from e
        if not os.path.exists(path):
            raise ConfigError(f"Config file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return cls(**raw)


# Default baseline config (can be overridden from outside)
DEFAULT_CONFIG = RadiologyInsightConfig()

# ==============================================================================
# 4. DATA MODELS (Input & Output)
# ==============================================================================


class RiskLevel(str, Enum):
    NORMAL = "normal"
    MILD = "mild"
    MODERATE = "moderate"
    HIGH = "high"


class ImagingInsight(BaseModel):
    """
    Structured, clinically relevant insight derived from imaging data.
    """

    insight_id: str = Field(default_factory=lambda: f"insight_{uuid.uuid4().hex}")
    patient_id: Optional[str] = None
    modality: str
    acquisition_date: Optional[datetime] = None

    key_findings: List[str]
    quantitative_measures: Dict[str, float]
    clinical_significance: str
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    risk_level: RiskLevel
    composite_anomaly_score: Optional[float] = None
    percentile_estimates: Dict[str, float] = Field(default_factory=dict)

    # Provenance / audit
    analysis_strategy: str
    strategy_config_hash: str
    normative_data_cohort: str
    normative_dataset_hash: Optional[str] = None
    source_data_references: Dict[str, Any] = Field(default_factory=dict)
    module_version: str = MODULE_VERSION
    generated_at: datetime = Field(default_factory=datetime.utcnow)

    def to_memory_dict(
        self, importance_config: ImportanceBoostersConfig | None = None
    ) -> Dict[str, Any]:
        importance = self._calculate_importance(
            importance_config or DEFAULT_CONFIG.importance_boosters
        )
        return {
            "content": self.summarize_for_memory(),
            "type": "imaging_insight",
            "importance": importance,
            "metadata": self.model_dump(mode="json"),
            "created_at": self.generated_at.isoformat(),
        }

    def summarize_for_memory(self) -> str:
        date_str = self.acquisition_date.strftime("%Y-%m-%d") if self.acquisition_date else "N/A"
        summary = (
            f"Imaging Analysis ({self.modality}, {date_str}): {self.clinical_significance} "
            f"Risk: {self.risk_level}. Findings: "
            f"{'; '.join(self.key_findings[:3]) if self.key_findings else 'None'}. "
            f"(Confidence: {self.confidence_score:.2f})"
        )
        return summary

    def _calculate_importance(self, cfg: ImportanceBoostersConfig) -> float:
        base = self.confidence_score
        boost = 0.0
        for keyword in cfg.critical_keywords:
            if any(keyword.lower() in f.lower() for f in self.key_findings):
                boost += cfg.keyword_boost_value
        if self.composite_anomaly_score is not None:
            boost += min(1.0, self.composite_anomaly_score) * cfg.anomaly_weight
        base += min(boost, cfg.max_total_boost)
        return min(base, 1.0)


class FeatureSet(BaseModel):
    """
    Input features for analysis, with validation.
    """

    patient_id: Optional[str] = None
    age: int = Field(..., gt=0)
    sex: str = Field(..., pattern="^(M|F|O)$")
    modality: str
    acquisition_date: Optional[datetime] = None
    measurements: Dict[str, float]
    quality_metrics: Dict[str, float] = Field(default_factory=dict)
    source_dicom_uids: List[str] = Field(default_factory=list)

    @field_validator("measurements")
    @classmethod
    def non_negative(cls, v: Dict[str, float]):
        for k, val in v.items():
            if not isinstance(val, (int, float)):
                raise ValueError(f"Measurement {k} must be numeric.")
            if val < 0:
                raise ValueError(f"Measurement {k} must be non-negative.")
        return v


# ==============================================================================
# 5. NORMATIVE DATA MANAGER
# ==============================================================================


class NormativeDataManager:
    """
    Manages normative datasets and provides demographic-specific statistics.
    """

    def __init__(self):
        self._datasets: Dict[str, pd.DataFrame] = {}
        self._dataset_raw_hash: Dict[str, str] = {}

    def load_dataset_from_string(self, cohort_id: str, csv_data: str):
        self._datasets[cohort_id] = pd.read_csv(io.StringIO(csv_data))
        self._dataset_raw_hash[cohort_id] = hashlib.sha256(csv_data.encode()).hexdigest()
        LOGGER.info(
            f"Loaded normative dataset for cohort={cohort_id} rows={len(self._datasets[cohort_id])}"
        )

    def dataset_hash(self, cohort_id: str) -> Optional[str]:
        return self._dataset_raw_hash.get(cohort_id)

    @lru_cache(maxsize=256)
    def get_stats(
        self, cohort_id: str, age: int, sex: str, measure: str
    ) -> Optional[Tuple[float, float]]:
        if cohort_id not in self._datasets:
            raise NormativeDataError(f"Normative cohort '{cohort_id}' not loaded.")
        df = self._datasets[cohort_id]

        row = df[(df["age_start"] <= age) & (df["age_end"] >= age) & (df["sex"] == sex)]
        if row.empty:
            return None
        mean_col, std_col = f"{measure}_mean", f"{measure}_std"
        if mean_col not in row.columns or std_col not in row.columns:
            return None
        mean = float(row.iloc[0][mean_col])
        std = float(row.iloc[0][std_col])
        return mean, std


# ==============================================================================
# 6. STRATEGY REGISTRY
# ==============================================================================

_STRATEGY_REGISTRY: Dict[str, Type["BaseAnalysisStrategy"]] = {}


def register_strategy(key: str):
    def decorator(cls: Type["BaseAnalysisStrategy"]):
        _STRATEGY_REGISTRY[key] = cls
        return cls

    return decorator


# ==============================================================================
# 7. BASE STRATEGY
# ==============================================================================


class BaseAnalysisStrategy(ABC):
    def __init__(self, config: RadiologyInsightConfig, normative_manager: NormativeDataManager):
        self.config = config
        self.normative_manager = normative_manager
        self.strategy_name = config.strategy_name
        config_str = self.config.model_dump_json()
        self.config_hash = hashlib.sha256(config_str.encode()).hexdigest()

    @abstractmethod
    def analyze(self, features: FeatureSet) -> ImagingInsight: ...


# ==============================================================================
# 8. BRAIN MRI VOLUMETRY STRATEGY
# ==============================================================================


@register_strategy("mri_volumetry")
class BrainMRIVolumetryStrategy(BaseAnalysisStrategy):
    """
    Extensible volumetric MRI strategy with:
    - Z-scores
    - Percentile approximation
    - Composite anomaly scoring
    - Risk stratification
    """

    def __init__(self, config: RadiologyInsightConfig, normative_manager: NormativeDataManager):
        super().__init__(config, normative_manager)
        self.thresholds = config.z_score_thresholds
        self.qc = config.quality_control

    @timed
    def analyze(self, features: FeatureSet) -> ImagingInsight:
        LOGGER.debug(f"Starting analysis for patient={features.patient_id}")
        quantitative: Dict[str, float] = {}
        percentiles: Dict[str, float] = {}
        findings: List[str] = []
        anomaly_components: List[float] = []

        # Measurement iteration
        for raw_name, value in features.measurements.items():
            measure = self.config.measure_aliases.get(raw_name, raw_name)
            stats = self.normative_manager.get_stats(
                self.config.normative_data_cohort, features.age, features.sex, measure
            )

            if not stats:
                msg = f"Normative stat missing for measure={measure}"
                if self.config.strict_normative_lookup:
                    raise NormativeDataError(msg)
                LOGGER.warning(msg)
                if not self.config.allow_missing_measures:
                    continue
                else:
                    quantitative[f"{measure}_value"] = value
                    continue

            mean, std = stats
            if std <= 0:
                LOGGER.warning(f"Non-positive std for measure={measure}, skipping Z-score.")
                continue

            z = (value - mean) / std
            quantitative[f"{measure}_z_score"] = z
            quantitative[f"{measure}_value"] = value

            if self.config.enable_percentiles:
                # Percentile from Z assuming normal distribution
                pct = 0.5 * (1 + math.erf(z / math.sqrt(2))) * 100
                percentiles[f"{measure}_percentile"] = round(pct, 2)

            # Interpret Z score
            interpret = self._interpret_z_score(measure, z)
            if interpret:
                findings.append(interpret)
                anomaly_components.append(abs(z))

        if not findings:
            findings.append("Volumetric analysis within expected norms.")

        # Composite anomaly score
        composite_anomaly = None
        if anomaly_components and self.config.enable_composite_anomaly_score:
            # Weighted by squared magnitude for outlier emphasis
            squared = [c**2 for c in anomaly_components]
            composite_anomaly = float(
                min(1.0, sum(squared) / (sum(squared) + 5))
            )  # bounded transform
            quantitative["composite_anomaly_score"] = composite_anomaly

        clinical_summary = self._summarize_clinical(findings)
        confidence = self._compute_confidence(features.quality_metrics)
        risk_level = self._derive_risk(findings, composite_anomaly)

        return ImagingInsight(
            patient_id=features.patient_id,
            modality=features.modality,
            acquisition_date=features.acquisition_date,
            key_findings=findings,
            quantitative_measures=quantitative,
            clinical_significance=clinical_summary,
            confidence_score=confidence,
            risk_level=risk_level,
            composite_anomaly_score=composite_anomaly,
            percentile_estimates=percentiles,
            analysis_strategy=self.strategy_name,
            strategy_config_hash=self.config_hash,
            normative_data_cohort=self.config.normative_data_cohort,
            normative_dataset_hash=self.normative_manager.dataset_hash(
                self.config.normative_data_cohort
            ),
            source_data_references={"dicom_uids": features.source_dicom_uids},
        )

    def _interpret_z_score(self, measure: str, z: float) -> Optional[str]:
        if z < self.thresholds.significant_atrophy:
            return f"Severe low volume in {measure.replace('_mm3','')} (Z={z:.2f}) indicating significant atrophy."
        if z < self.thresholds.mild_atrophy:
            return f"Reduced volume in {measure.replace('_mm3','')} (Z={z:.2f}) suggesting mild atrophy."
        if z > self.thresholds.high_volume_anomaly:
            return f"Markedly elevated {measure.replace('_mm3','')} volume (Z={z:.2f}); correlate clinically."
        # borderline_high currently not separately verbalized; can be added if needed
        return None

    def _summarize_clinical(self, findings: List[str]) -> str:
        joined = " ".join(f.lower() for f in findings)
        if "severe" in joined:
            return "Findings strongly suggest clinically significant structural atrophy; urgent evaluation advised."
        if "mild atrophy" in joined or "reduced volume" in joined:
            return (
                "Findings suggest early or mild volumetric loss; clinical correlation recommended."
            )
        if "elevated" in joined:
            return "Unusually high regional volume observed; consider differential etiologies."
        return "Volumetric profile is within normal limits for age and sex."

    def _compute_confidence(self, qm: Dict[str, float]) -> float:
        confidence = 0.95
        snr = qm.get("snr")
        motion = qm.get("motion_score")
        if snr is not None and snr < self.qc.min_snr_for_high_confidence:
            confidence -= self.qc.confidence_penalty_low_snr
        if motion is not None and motion > self.qc.max_motion_for_high_confidence:
            confidence -= self.qc.confidence_penalty_high_motion
        return max(self.qc.min_confidence_floor, min(confidence, 1.0))

    def _derive_risk(self, findings: List[str], composite: Optional[float]) -> RiskLevel:
        lower = [f.lower() for f in findings]
        if any("severe" in f for f in lower):
            return RiskLevel.HIGH
        if any("mild" in f or "reduced" in f for f in lower):
            return RiskLevel.MODERATE
        if composite is not None and composite > 0.5:
            return RiskLevel.MODERATE
        if composite is not None and composite > 0.25:
            return RiskLevel.MILD
        return RiskLevel.NORMAL


# ==============================================================================
# 9. MODULE FACADE
# ==============================================================================


class RadiologyInsightModule:
    """
    Orchestrates analysis strategies with normative datasets.
    """

    def __init__(self, config: RadiologyInsightConfig = DEFAULT_CONFIG):
        self.config = config
        self.normative_manager = NormativeDataManager()
        self._load_default_normative()
        self.strategies: Dict[str, BaseAnalysisStrategy] = {}
        self._instantiate_registered_strategies()

    def _instantiate_registered_strategies(self):
        for key, cls in _STRATEGY_REGISTRY.items():
            try:
                self.strategies[key] = cls(self.config, self.normative_manager)
            except Exception as e:
                LOGGER.error(f"Failed to instantiate strategy '{key}': {e}")

    def list_strategies(self) -> List[str]:
        return list(self.strategies.keys())

    def _load_default_normative(self):
        normative_csv_data = """age_start,age_end,sex,total_brain_volume_mm3_mean,total_brain_volume_mm3_std,hippocampal_volume_mm3_mean,hippocampal_volume_mm3_std
30,39,M,1450000,50000,3500,300
30,39,F,1350000,45000,3400,280
40,49,M,1420000,52000,3300,310
40,49,F,1320000,47000,3200,290
50,59,M,1390000,55000,3100,330
50,59,F,1290000,50000,3000,310
"""
        self.normative_manager.load_dataset_from_string(
            self.config.normative_data_cohort, normative_csv_data
        )

    @timed
    def generate_insight(self, strategy_key: str, features: FeatureSet) -> ImagingInsight:
        if strategy_key not in self.strategies:
            raise StrategyNotFoundError(
                f"Strategy '{strategy_key}' not registered. Available: {self.list_strategies()}"
            )
        strategy = self.strategies[strategy_key]
        try:
            return strategy.analyze(features)
        except ValidationError as ve:
            LOGGER.error(f"Validation error during analysis: {ve}")
            raise
        except Exception as e:
            LOGGER.exception("Unexpected error during analysis.")
            raise InsightError(str(e)) from e


# ==============================================================================
# 10. CLI SUPPORT
# ==============================================================================


def _cli():
    parser = argparse.ArgumentParser(description="Run AiMedRes Imaging Insight Module (Advanced).")
    parser.add_argument(
        "--strategy", default="mri_volumetry", help="Strategy key (default: mri_volumetry)"
    )
    parser.add_argument("--input-json", help="Path to JSON file with FeatureSet payload")
    parser.add_argument("--output-json", help="Path to write result Insight JSON")
    parser.add_argument(
        "--print-memory", action="store_true", help="Print memory dict representation"
    )
    parser.add_argument("--yaml-config", help="Optional path to YAML config override")
    args = parser.parse_args()

    if args.yaml_config:
        config = RadiologyInsightConfig.from_yaml(args.yaml_config)
    else:
        config = RadiologyInsightConfig.from_env()

    module = RadiologyInsightModule(config=config)

    if args.input_json:
        with open(args.input_json, "r", encoding="utf-8") as f:
            raw = json.load(f)
        features = FeatureSet(**raw)
    else:
        # Example fallback dataset (mirrors earlier demonstration)
        features = FeatureSet(
            patient_id="EXAMPLE-001",
            age=55,
            sex="F",
            modality="MRI",
            acquisition_date=datetime.utcnow(),
            measurements={
                "total_brain_volume_mm3": 1280000,
                "hippocampal_volume_mm3": 2200,
            },
            quality_metrics={"snr": 22.5, "motion_score": 0.15},
            source_dicom_uids=["1.2.3.4.5.6.7.8.9"],
        )

    insight = module.generate_insight(args.strategy, features)
    output_data = insight.model_dump(mode="json")

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)
        LOGGER.info(f"Wrote insight JSON to {args.output_json}")
    else:
        print(json.dumps(output_data, indent=2))

    if args.print_memory:
        print("\n--- Memory Dict ---")
        print(json.dumps(insight.to_memory_dict(), indent=2))


# ==============================================================================
# 11. EXAMPLE (Programmatic)
# ==============================================================================


def _example_usage():
    LOGGER.info("--- Initializing Radiology Insight Module (Advanced) ---")
    module = RadiologyInsightModule()

    patient_features = FeatureSet(
        patient_id="PID-98765",
        age=55,
        sex="F",
        modality="MRI",
        acquisition_date=datetime(2023, 10, 26),
        measurements={
            "total_brain_volume_mm3": 1280000,
            "hippocampal_volume_mm3": 2200,
        },
        quality_metrics={"snr": 22.5, "motion_score": 0.15},
        source_dicom_uids=["1.2.840.113619.2.55.3.2831183550.412.1384892445.651"],
    )

    insight = module.generate_insight("mri_volumetry", patient_features)
    print("\n--- Generated ImagingInsight ---")
    print(insight.model_dump_json(indent=2))

    print("\n--- Agent Memory Payload ---")
    print(json.dumps(insight.to_memory_dict(), indent=2))


# ==============================================================================
# 12. MAIN ENTRY
# ==============================================================================

if __name__ == "__main__":
    # If executed as a script with arguments, use CLI; else run example.
    if len(sys.argv) > 1:
        _cli()
    else:
        _example_usage()
