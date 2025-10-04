#!/usr/bin/env python3
"""
Single-File Adversarial Defense Module for AiMedRes.

This module provides a self-contained, production-grade security component for
detecting and mitigating adversarial attacks against clinical AI models. It is
designed for direct integration into the AiMedRes platform, featuring an async,
pluggable, and testable architecture.
"""

import asyncio
import hashlib
import json
import logging
from abc import ABC, abstractmethod
from collections import deque
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field, field_validator

# ==============================================================================
# 1. CONFIGURATION (Simulating an external configs/security_config.yaml)
# ==============================================================================

# In a real AiMedRes integration, this would be loaded from a central config file.
ADVERSARIAL_DEFENSE_CONFIG = {
    "active_detectors": [
        "input_bounds",
        "statistical_anomaly",
        "ensemble_inconsistency",
        "historical_pattern",
    ],
    "input_bounds": {
        "age": (0, 120),
        "creatinine_mg_dl": (0.1, 15.0),
        "systolic_bp": (50, 250),
    },
    "statistical_anomaly": {
        "max_z_score_deviation": 4.0, # Stricter threshold
    },
    "ensemble_inconsistency": {
        "max_prediction_variance": 0.25,
    },
    "historical_pattern": {
        "identical_input_limit": {"count": 5, "timespan_seconds": 3600},
        "boundary_probe_sensitivity": 0.6,
    },
}

# ==============================================================================
# 2. ENUMS AND DATA MODELS (Using Pydantic for robust validation)
# ==============================================================================

class AttackType(str, Enum):
    EVASION = "EVASION"
    POISONING = "POISONING"
    MODEL_EXTRACTION = "MODEL_EXTRACTION"
    INFERENCE = "INFERENCE"

class AttackSeverity(str, Enum):
    INFO = "INFO"
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class AttackDetection(BaseModel):
    """A structured record of a detected potential adversarial attack."""
    detector_name: str = Field(..., description="The name of the detector that triggered this finding.")
    attack_type: AttackType
    severity: AttackSeverity
    confidence: float = Field(..., ge=0.0, le=1.0)
    message: str = Field(..., description="A human-readable description of the potential attack.")
    details: Dict[str, Any] = Field(default_factory=dict, description="Supporting data for the detection.")
    
class ModelInput(BaseModel):
    """Represents the input to the AI model, with metadata for security analysis."""
    request_id: str = Field(default_factory=lambda: f"req_{uuid.uuid4().hex}")
    user_id: Optional[str] = "anonymous"
    session_id: Optional[str] = None
    features: Dict[str, float]
    
    @field_validator('features')
    def check_features_not_empty(cls, v):
        if not v:
            raise ValueError("Features dictionary cannot be empty.")
        return v

class ModelOutput(BaseModel):
    """Represents the output from the AI model, including ensemble data if available."""
    final_prediction: Any
    confidence: float
    ensemble_predictions: Optional[List[Any]] = None

# ==============================================================================
# 3. DECOUPLED DEPENDENCIES (Protocols for logging and data provision)
# ==============================================================================

class AttackLogger(ABC):
    """Abstract protocol for logging detected attacks for persistence and analysis."""
    @abstractmethod
    async def log_detection(self, detection: AttackDetection, model_input: ModelInput):
        pass

class BaselineProvider(ABC):
    """Abstract protocol for providing trusted baseline statistics for features."""
    @abstractmethod
    async def get_baseline_stats(self, feature_name: str) -> Optional[Tuple[float, float]]:
        pass # Returns (mean, std_dev)

# --- Concrete implementations for demonstration ---

class InMemoryAttackLogger(AttackLogger):
    """A simple in-memory logger for demonstration and testing."""
    def __init__(self, maxlen: int = 1000):
        self.detections = deque(maxlen=maxlen)
    
    async def log_detection(self, detection: AttackDetection, model_input: ModelInput):
        log_entry = {
            "timestamp": datetime.now(timezone.utc),
            "detection": detection.model_dump(),
            "input": model_input.model_dump()
        }
        self.detections.append(log_entry)
        logging.warning(f"ATTACK DETECTED: {log_entry}")

class StaticBaselineProvider(BaselineProvider):
    """A simple baseline provider with static data for demonstration."""
    def __init__(self):
        self.baselines = {
            "age": (55.0, 15.0),
            "creatinine_mg_dl": (0.9, 0.3),
            "systolic_bp": (120.0, 18.0),
        }
    
    async def get_baseline_stats(self, feature_name: str) -> Optional[Tuple[float, float]]:
        return self.baselines.get(feature_name)

# ==============================================================================
# 4. DETECTOR STRATEGY PATTERN (Pluggable, async detection modules)
# ==============================================================================

class BaseDetector(ABC):
    """Abstract base class for all adversarial detectors."""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = "base_detector"

    @abstractmethod
    async def detect(self, model_input: ModelInput, model_output: ModelOutput) -> List[AttackDetection]:
        pass

class InputBoundsDetector(BaseDetector):
    """Detects if input features are outside pre-defined plausible ranges."""
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.name = "input_bounds"

    async def detect(self, model_input: ModelInput, model_output: ModelOutput) -> List[AttackDetection]:
        violations = []
        for feature, value in model_input.features.items():
            if feature in self.config:
                min_val, max_val = self.config[feature]
                if not (min_val <= value <= max_val):
                    violations.append(f"Feature '{feature}' ({value}) is outside range [{min_val}, {max_val}].")
        
        if violations:
            return [AttackDetection(
                detector_name=self.name,
                attack_type=AttackType.EVASION,
                severity=AttackSeverity.HIGH,
                confidence=0.95,
                message="Input features fall outside clinically plausible bounds.",
                details={"violations": violations}
            )]
        return []

class StatisticalAnomalyDetector(BaseDetector):
    """Detects statistical anomalies using Z-scores against a trusted baseline."""
    def __init__(self, config: Dict[str, Any], baseline_provider: BaselineProvider):
        super().__init__(config)
        self.name = "statistical_anomaly"
        self.baseline_provider = baseline_provider

    async def detect(self, model_input: ModelInput, model_output: ModelOutput) -> List[AttackDetection]:
        anomalies = []
        max_z_score = 0.0
        for feature, value in model_input.features.items():
            stats = await self.baseline_provider.get_baseline_stats(feature)
            if stats:
                mean, std = stats
                if std > 1e-6:
                    z_score = abs((value - mean) / std)
                    max_z_score = max(max_z_score, z_score)
                    if z_score > self.config.get("max_z_score_deviation", 4.0):
                        anomalies.append(f"Feature '{feature}' has an anomalous Z-score of {z_score:.2f}.")
        
        if anomalies:
            return [AttackDetection(
                detector_name=self.name,
                attack_type=AttackType.EVASION,
                severity=AttackSeverity.MODERATE,
                confidence=min(0.9, max_z_score / (self.config.get("max_z_score_deviation", 4.0) * 2)),
                message="Input features are statistically improbable compared to baseline.",
                details={"anomalies": anomalies, "max_z_score": max_z_score}
            )]
        return []

class EnsembleInconsistencyDetector(BaseDetector):
    """Detects high variance in ensemble model predictions, a sign of evasion."""
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.name = "ensemble_inconsistency"

    async def detect(self, model_input: ModelInput, model_output: ModelOutput) -> List[AttackDetection]:
        if model_output.ensemble_predictions and len(model_output.ensemble_predictions) > 1:
            try:
                variance = np.var(model_output.ensemble_predictions)
                if variance > self.config.get("max_prediction_variance", 0.25):
                    return [AttackDetection(
                        detector_name=self.name,
                        attack_type=AttackType.EVASION,
                        severity=AttackSeverity.MODERATE,
                        confidence=0.8,
                        message="High disagreement among ensemble members suggests an unstable input region.",
                        details={"prediction_variance": variance}
                    )]
            except TypeError:
                # Handle non-numeric predictions if necessary
                pass
        return []

# ==============================================================================
# 5. MAIN MODULE FACADE (The primary entry point for AiMedRes)
# ==============================================================================

class AdversarialDefenseModule:
    """
    A facade that orchestrates adversarial attack detection using a suite of
    pluggable, asynchronous detectors.
    """
    def __init__(self, config: Dict[str, Any], logger: AttackLogger, baseline_provider: BaselineProvider):
        self.config = config
        self.logger = logger
        self.detectors = self._load_detectors(config, baseline_provider)
        logging.info(f"AdversarialDefenseModule initialized with {len(self.detectors)} detectors.")

    def _load_detectors(self, config, baseline_provider) -> List[BaseDetector]:
        """Dynamically loads and instantiates detectors based on the config."""
        detector_map = {
            "input_bounds": InputBoundsDetector,
            "statistical_anomaly": StatisticalAnomalyDetector,
            "ensemble_inconsistency": EnsembleInconsistencyDetector,
        }
        
        loaded_detectors = []
        for name in config.get("active_detectors", []):
            if name in detector_map:
                detector_config = config.get(name, {})
                # Inject dependencies as needed
                if name == "statistical_anomaly":
                    loaded_detectors.append(detector_map[name](detector_config, baseline_provider))
                else:
                    loaded_detectors.append(detector_map[name](detector_config))
        return loaded_detectors

    async def inspect_request(self, model_input: ModelInput, model_output: ModelOutput) -> List[AttackDetection]:
        """
        Inspects a model request/response pair for signs of adversarial attacks.
        
        This method runs all active detectors concurrently and aggregates their findings.
        Detected attacks are automatically logged.
        """
        if not self.detectors:
            return []

        # Run all detectors concurrently
        detection_tasks = [detector.detect(model_input, model_output) for detector in self.detectors]
        results = await asyncio.gather(*detection_tasks)
        
        # Flatten list of lists and log detections
        all_detections = [detection for sublist in results for detection in sublist]
        
        if all_detections:
            log_tasks = [self.logger.log_detection(det, model_input) for det in all_detections]
            await asyncio.gather(*log_tasks)
            
        return all_detections

# ==============================================================================
# 6. EXAMPLE USAGE (Demonstrates integration and functionality)
# ==============================================================================

async def main():
    """Demonstrates the setup and use of the AdversarialDefenseModule."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    print("--- 1. Initializing Dependencies ---")
    # In AiMedRes, these would be provided by the application's service container.
    attack_logger = InMemoryAttackLogger()
    baseline_provider = StaticBaselineProvider()

    print("--- 2. Initializing Adversarial Defense Module ---")
    defense_module = AdversarialDefenseModule(
        config=ADVERSARIAL_DEFENSE_CONFIG,
        logger=attack_logger,
        baseline_provider=baseline_provider
    )

    # --- Case 1: Benign Input ---
    print("\n--- 3. Analyzing a BENIGN request ---")
    benign_input = ModelInput(
        user_id="user_123",
        features={"age": 65.0, "creatinine_mg_dl": 1.0, "systolic_bp": 130.0}
    )
    benign_output = ModelOutput(
        final_prediction=0.1,
        confidence=0.95,
        ensemble_predictions=[0.1, 0.11, 0.09, 0.1, 0.12]
    )
    detections = await defense_module.inspect_request(benign_input, benign_output)
    if not detections:
        print("✅  SUCCESS: No attacks detected, as expected.")
    else:
        print(f"❌  FAILURE: Benign input was flagged: {detections}")

    # --- Case 2: Adversarial Input (Out of Bounds & Statistical Anomaly) ---
    print("\n--- 4. Analyzing an ADVERSARIAL request ---")
    adversarial_input = ModelInput(
        user_id="attacker_789",
        features={"age": 150.0, "creatinine_mg_dl": 14.0, "systolic_bp": 125.0}
    )
    adversarial_output = ModelOutput(
        final_prediction=0.9, # Model is fooled
        confidence=0.88,
        ensemble_predictions=[0.1, 0.95, 0.2, 0.89, 0.91] # High variance
    )
    detections = await defense_module.inspect_request(adversarial_input, adversarial_output)
    if detections:
        print("✅  SUCCESS: Adversarial attacks were detected:")
        for det in detections:
            print(f"  - {det.model_dump_json(indent=2)}")
    else:
        print("❌  FAILURE: Adversarial input was NOT detected.")
        
    print("\n--- 5. Reviewing Attack Log ---")
    print(f"Total attacks logged: {len(attack_logger.detections)}")
    if attack_logger.detections:
        print("Last logged attack:")
        print(json.dumps(attack_logger.detections[-1], indent=2, default=str))

if __name__ == "__main__":
    import uuid # for request_id
    asyncio.run(main())
