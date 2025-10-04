#!/usr/bin/env python3
"""
Single-File Radiology Insight Module for AiMedRes.

This module provides a self-contained, production-grade component for analyzing
medical imaging features and generating structured, clinically relevant insights.
It is designed to be integrated as a "skill" for an AI agent within the AiMedRes
multi-agent system.

Key Features:
- Pydantic models for robust data validation.
- Configuration-driven logic to separate parameters from code.
- Z-score calculation against normative data for scientific rigor.
- Strategy pattern for extensible analysis logic.
- Detailed provenance and auditability in the output.

Dependencies:
- pydantic
- pandas
"""

import io
import uuid
import hashlib
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod
from functools import lru_cache

import pandas as pd
from pydantic import BaseModel, Field

# ==============================================================================
# 1. CONFIGURATION (Simulating an external configs/radiology_config.yaml)
# ==============================================================================

# In a real AiMedRes integration, this dictionary would be loaded from a YAML file.
RADIOLOGY_INSIGHT_CONFIG = {
    "strategy_name": "BrainMRIVolumetry_v1.2_ZScore",
    "normative_data_cohort": "healthy_adults_mixed_scanner_v1",
    "z_score_thresholds": {
        "significant_atrophy": -2.5,
        "mild_atrophy": -1.75,
        "borderline_high": 1.75,
        "high_volume_anomaly": 2.5,
    },
    "quality_control": {
        "min_snr_for_high_confidence": 15.0,
        "max_motion_for_high_confidence": 0.3,
        "confidence_penalty_low_snr": 0.25,
        "confidence_penalty_high_motion": 0.30,
    },
    "importance_boosters": {
        "critical_keywords": ["atrophy", "severe", "significant", "abnormal", "mass", "lesion"],
        "keyword_boost_value": 0.2,
    },
}

# ==============================================================================
# 2. DATA MODELS (For robust, validated data structures)
# ==============================================================================

class ImagingInsight(BaseModel):
    """
    Structured, clinically relevant insight derived from imaging data.
    This is the primary output of the module.
    """
    insight_id: str = Field(default_factory=lambda: f"insight_{uuid.uuid4().hex}")
    patient_id: Optional[str] = None
    modality: str
    acquisition_date: Optional[datetime] = None
    
    key_findings: List[str] = Field(..., description="Concise, clinically relevant findings.")
    quantitative_measures: Dict[str, float] = Field(..., description="Key numerical results, like Z-scores.")
    clinical_significance: str = Field(..., description="A summary of the overall clinical meaning.")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence in the analysis (0-1).")
    
    # --- Provenance and Auditability Fields ---
    analysis_strategy: str = Field(..., description="Name and version of the analysis strategy used.")
    strategy_config_hash: str = Field(..., description="SHA256 hash of the configuration used for this analysis.")
    normative_data_cohort: str = Field(..., description="Identifier for the normative dataset used for comparison.")
    source_data_references: Dict[str, Any] = Field(default_factory=dict, description="Links to source data (e.g., DICOM UIDs).")
    
    generated_at: datetime = Field(default_factory=datetime.now)

    def to_memory_dict(self) -> Dict[str, Any]:
        """Formats the insight for storage in the AiMedRes agent memory system."""
        return {
            "content": self.summarize_for_memory(),
            "type": "imaging_insight",
            "importance": self._calculate_importance(),
            "metadata": self.model_dump(mode='json'),
            "created_at": self.generated_at.isoformat()
        }

    def summarize_for_memory(self) -> str:
        """Creates a natural language summary for quick recall by an agent."""
        date_str = self.acquisition_date.strftime('%Y-%m-%d') if self.acquisition_date else "N/A"
        summary = (
            f"Imaging Analysis ({self.modality}, {date_str}): "
            f"{self.clinical_significance}. "
            f"Key findings: {'; '.join(self.key_findings[:3])}. "
            f"(Confidence: {self.confidence_score:.2f})"
        )
        return summary

    def _calculate_importance(self) -> float:
        """Calculates a memory importance score based on findings."""
        base_importance = self.confidence_score
        # In a real system, these keywords would come from the config
        critical_keywords = ["atrophy", "lesion", "tumor", "severe", "significant", "abnormal"]
        for keyword in critical_keywords:
            if any(keyword.lower() in finding.lower() for finding in self.key_findings):
                base_importance += 0.2 # Use a configured boost value
        return min(base_importance, 1.0)


class FeatureSet(BaseModel):
    """
    Input features for analysis, with built-in validation.
    This is the primary input to the module.
    """
    patient_id: Optional[str] = None
    age: int = Field(..., gt=0, description="Patient age in years.")
    sex: str = Field(..., pattern="^(M|F|O)$", description="Patient sex (M, F, or O).")
    modality: str
    acquisition_date: Optional[datetime] = None
    
    measurements: Dict[str, float] = Field(..., description="Quantitative measurements from image processing (e.g., volumes).")
    quality_metrics: Dict[str, float] = Field(default_factory=dict, description="Image quality scores (e.g., SNR, motion).")
    source_dicom_uids: List[str] = Field(default_factory=list)

# ==============================================================================
# 3. NORMATIVE DATA MANAGER (For scientifically-grounded comparisons)
# ==============================================================================

class NormativeDataManager:
    """Loads and provides access to normative datasets for Z-score calculation."""
    
    def __init__(self):
        self._datasets = {}

    def load_dataset_from_string(self, cohort_id: str, csv_data: str):
        """Loads a normative dataset from a CSV string into a pandas DataFrame."""
        self._datasets[cohort_id] = pd.read_csv(io.StringIO(csv_data))

    @lru_cache(maxsize=128)
    def get_stats(self, cohort_id: str, age: int, sex: str, measure: str) -> Optional[tuple[float, float]]:
        """
        Retrieves the mean and standard deviation for a given demographic and measurement.
        
        Note: This uses a simplified age-group matching. A real implementation
        might use more sophisticated age-based regression models.
        """
        if cohort_id not in self._datasets:
            raise ValueError(f"Normative data cohort '{cohort_id}' not loaded.")
        
        df = self._datasets[cohort_id]
        
        # Find the matching demographic row
        row = df[
            (df['age_start'] <= age) &
            (df['age_end'] >= age) &
            (df['sex'] == sex)
        ]
        
        if row.empty:
            return None # No matching normative data for this demographic
            
        mean_col, std_col = f"{measure}_mean", f"{measure}_std"
        if mean_col not in row.columns or std_col not in row.columns:
            return None # The measure is not in this normative dataset
            
        mean = row.iloc[0][mean_col]
        std = row.iloc[0][std_col]
        
        return mean, std

# ==============================================================================
# 4. ANALYSIS STRATEGY (The core analysis logic)
# ==============================================================================

class BaseAnalysisStrategy(ABC):
    """Abstract base class for all analysis strategies."""
    def __init__(self, config: dict, normative_manager: NormativeDataManager):
        self.config = config
        self.normative_manager = normative_manager
        self.strategy_name = "base_strategy"
        
        # Pre-calculate a hash of the config for auditability
        config_str = json.dumps(self.config, sort_keys=True)
        self.config_hash = hashlib.sha256(config_str.encode()).hexdigest()

    @abstractmethod
    def analyze(self, features: FeatureSet) -> ImagingInsight:
        """Analyzes the features and returns a structured ImagingInsight."""
        pass

class BrainMRIVolumetryStrategy(BaseAnalysisStrategy):
    """
    Analyzes brain MRI volumetric data against normative data to find anomalies.
    """
    def __init__(self, config: dict, normative_manager: NormativeDataManager):
        super().__init__(config, normative_manager)
        self.strategy_name = self.config.get("strategy_name", "BrainMRIVolumetry_default")
        self.thresholds = self.config.get("z_score_thresholds", {})
        self.qc_config = self.config.get("quality_control", {})
        self.cohort_id = self.config.get("normative_data_cohort")

    def analyze(self, features: FeatureSet) -> ImagingInsight:
        key_findings = []
        quantitative_measures = {}

        for name, value in features.measurements.items():
            stats = self.normative_manager.get_stats(self.cohort_id, features.age, features.sex, name)
            if stats:
                mean, std = stats
                if std > 0:
                    z_score = (value - mean) / std
                    quantitative_measures[f"{name}_z_score"] = z_score
                    finding = self._interpret_z_score(name.replace('_mm3', ''), z_score)
                    if finding:
                        key_findings.append(finding)
        
        clinical_significance = self._assess_clinical_significance(key_findings)
        confidence = self._calculate_confidence(features.quality_metrics)

        return ImagingInsight(
            patient_id=features.patient_id,
            modality="MRI",
            acquisition_date=features.acquisition_date,
            key_findings=key_findings or ["Volumetric analysis within normal limits."],
            quantitative_measures=quantitative_measures,
            clinical_significance=clinical_significance,
            confidence_score=confidence,
            analysis_strategy=self.strategy_name,
            strategy_config_hash=self.config_hash,
            normative_data_cohort=self.cohort_id,
            source_data_references={"dicom_uids": features.source_dicom_uids}
        )

    def _interpret_z_score(self, measure_name: str, z_score: float) -> Optional[str]:
        """Translates a Z-score into a clinical finding based on config thresholds."""
        if z_score < self.thresholds.get("significant_atrophy", -2.5):
            return f"Significant low volume in {measure_name} suggesting severe atrophy."
        elif z_score < self.thresholds.get("mild_atrophy", -1.75):
            return f"Mildly reduced volume in {measure_name} suggesting potential atrophy."
        elif z_score > self.thresholds.get("high_volume_anomaly", 2.5):
            return f"Significantly high volume in {measure_name}, clinical correlation recommended."
        return None

    def _assess_clinical_significance(self, findings: List[str]) -> str:
        """Generates an overall clinical summary."""
        if any("severe atrophy" in f.lower() for f in findings):
            return "Findings are highly suggestive of significant cerebral atrophy; urgent clinical review required."
        if any("atrophy" in f.lower() for f in findings):
            return "Volumetric findings suggest potential cerebral atrophy; clinical correlation is recommended."
        return "Volumetric analysis is within normal limits compared to the age-and-sex-matched cohort."

    def _calculate_confidence(self, quality_metrics: dict) -> float:
        """Calculates analysis confidence based on image quality."""
        confidence = 0.95  # Start with high confidence
        snr = quality_metrics.get('snr')
        motion = quality_metrics.get('motion_score')

        if snr is not None and snr < self.qc_config.get("min_snr_for_high_confidence", 15.0):
            confidence -= self.qc_config.get("confidence_penalty_low_snr", 0.25)
        if motion is not None and motion > self.qc_config.get("max_motion_for_high_confidence", 0.3):
            confidence -= self.qc_config.get("confidence_penalty_high_motion", 0.30)
        
        return max(0.1, min(confidence, 1.0))

# ==============================================================================
# 5. MAIN MODULE FACADE (The primary entry point for AiMedRes)
# ==============================================================================

class RadiologyInsightModule:
    """
    A facade that orchestrates the analysis process.
    This class would be instantiated once in the AiMedRes system.
    """
    def __init__(self, config: dict):
        self.config = config
        self.normative_manager = NormativeDataManager()
        self._load_data()
        
        self.strategies: Dict[str, BaseAnalysisStrategy] = {
            "mri_volumetry": BrainMRIVolumetryStrategy(
                self.config, self.normative_manager
            )
        }

    def _load_data(self):
        """Loads all necessary data assets."""
        # In a real system, this would read from files specified in the config.
        # Here, we embed the data for portability.
        normative_csv_data = """age_start,age_end,sex,total_brain_volume_mm3_mean,total_brain_volume_mm3_std,hippocampal_volume_mm3_mean,hippocampal_volume_mm3_std
30,39,M,1450000,50000,3500,300
30,39,F,1350000,45000,3400,280
40,49,M,1420000,52000,3300,310
40,49,F,1320000,47000,3200,290
50,59,M,1390000,55000,3100,330
50,59,F,1290000,50000,3000,310
"""
        cohort_id = self.config.get("normative_data_cohort")
        self.normative_manager.load_dataset_from_string(cohort_id, normative_csv_data)

    def generate_insight(self, strategy_name: str, features: FeatureSet) -> ImagingInsight:
        """
        Generates an imaging insight using a specified strategy.

        Args:
            strategy_name: The key for the desired analysis strategy (e.g., "mri_volumetry").
            features: A validated FeatureSet object containing the patient's data.

        Returns:
            A structured ImagingInsight object.
        """
        if strategy_name not in self.strategies:
            raise ValueError(f"Strategy '{strategy_name}' not registered.")
        
        strategy = self.strategies[strategy_name]
        return strategy.analyze(features)

# ==============================================================================
# 6. EXAMPLE USAGE (Demonstrates how to use the module)
# ==============================================================================

if __name__ == "__main__":
    print("--- Initializing Radiology Insight Module ---")
    # 1. Instantiate the main module with the configuration
    #    In AiMedRes, this would be a singleton or a shared component.
    insight_module = RadiologyInsightModule(config=RADIOLOGY_INSIGHT_CONFIG)
    
    print("\n--- Preparing Patient Data for Analysis ---")
    # 2. Create a FeatureSet for a patient.
    #    This data would typically come from a data processing pipeline.
    #    This patient has a low hippocampal volume for their age.
    patient_features = FeatureSet(
        patient_id="PID-98765",
        age=55,
        sex="F",
        modality="MRI",
        acquisition_date=datetime(2023, 10, 26),
        measurements={
            "total_brain_volume_mm3": 1280000, # Within normal range
            "hippocampal_volume_mm3": 2200,    # Very low, should trigger an alert (Z < -2.5)
        },
        quality_metrics={"snr": 22.5, "motion_score": 0.15},
        source_dicom_uids=["1.2.840.113619.2.55.3.2831183550.412.1384892445.651"]
    )
    print(f"Analyzing data for Patient: {patient_features.patient_id}, Age: {patient_features.age}, Sex: {patient_features.sex}")

    print("\n--- Generating Insight ---")
    # 3. Run the analysis to generate the insight
    try:
        imaging_insight = insight_module.generate_insight(
            strategy_name="mri_volumetry",
            features=patient_features
        )
        
        print("\n--- [SUCCESS] Generated ImagingInsight Object ---")
        # Pydantic's model_dump_json provides a clean, readable output
        print(imaging_insight.model_dump_json(indent=2))

        print("\n--- [INTEGRATION] Data for Agent Memory System ---")
        # 4. Get the dictionary formatted for the agent's memory
        memory_data = imaging_insight.to_memory_dict()
        print(json.dumps(memory_data, indent=2))
        
    except Exception as e:
        print(f"\n--- [ERROR] Analysis failed: {e} ---")
