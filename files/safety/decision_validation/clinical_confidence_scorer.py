"""
Clinical Confidence Scorer - Decision Certainty Metrics

Provides comprehensive confidence scoring with uncertainty quantification
for AI clinical decisions, supporting human oversight triggers.
(Enhanced version with richer uncertainty semantics & advanced metrics.)
"""

import logging
import json
import hashlib
import uuid
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple, Iterable

import numpy as np

logger = logging.getLogger('duetmind.clinical_confidence_scorer')


class ConfidenceLevel(Enum):
    """Standardized confidence levels for clinical decisions"""
    VERY_LOW = "VERY_LOW"      # 0-40%
    LOW = "LOW"                # 40-60%
    MODERATE = "MODERATE"      # 60-80%
    HIGH = "HIGH"              # 80-95%
    VERY_HIGH = "VERY_HIGH"    # 95-100%


class UncertaintyType(Enum):
    """
    Rich uncertainty taxonomy for clinical AI decisions.

    Each entry payload tuple:
      (code, reducible, description, mitigation_hint, default_weight)

    - reducible: True if can be decreased with more data / modeling.
    - default_weight: Suggested proportional influence in aggregate score.
    """
    # Core Original Dimensions (kept for backward compatibility)
    EPISTEMIC       = ("EPISTEMIC", True,  "Model knowledge / parameter uncertainty",
                       "Collect more diverse training data; model retraining; Bayesian ensembling",
                       0.25)
    ALEATORIC       = ("ALEATORIC", False, "Inherent data noise / measurement randomness",
                       "Improve data quality pipelines; better sensors",
                       0.15)
    DISTRIBUTIONAL  = ("DISTRIBUTIONAL", True,  "Out-of-distribution / population mismatch",
                       "Deploy drift monitors; domain adaptation; expand cohort coverage",
                       0.20)
    TEMPORAL        = ("TEMPORAL", True,  "Time-based drift (e.g., evolving clinical practice)",
                       "Temporal re-calibration; rolling retrain schedule",
                       0.10)

    # New Extended Dimensions
    CONCEPT_DRIFT   = ("CONCEPT_DRIFT", True, "Shift in label semantics or protocol definition",
                       "Continuous labeling audits; maintain concept registries",
                       0.10)
    POPULATION_SHIFT= ("POPULATION_SHIFT", True, "Demographic / epidemiological shift",
                       "Re-weight sampling; fairness monitoring; stratified evaluation",
                       0.08)
    MEASUREMENT     = ("MEASUREMENT", True, "Device / sensor calibration or systematic bias",
                       "Instrument calibration; cross-device validation",
                       0.05)
    HUMAN_INPUT     = ("HUMAN_INPUT", True, "Uncertainty from manual data entry inconsistencies",
                       "UI validation; double data entry for critical fields",
                       0.03)
    AGGREGATION     = ("AGGREGATION", True, "Variance introduced by model fusion / ensemble logic",
                       "Refine ensemble weighting; gating network calibration",
                       0.04)

    @property
    def code(self) -> str:
        return self.value[0]

    @property
    def reducible(self) -> bool:
        return self.value[1]

    @property
    def description(self) -> str:
        return self.value[2]

    @property
    def mitigation_hint(self) -> str:
        return self.value[3]

    @property
    def default_weight(self) -> float:
        return self.value[4]

    @classmethod
    def from_str(cls, name: str) -> "UncertaintyType":
        normalized = name.strip().upper()
        for member in cls:
            if member.code == normalized or member.name == normalized:
                return member
        raise ValueError(f"Unknown UncertaintyType: {name}")

    @classmethod
    def weighted_members(cls, include: Optional[Iterable["UncertaintyType"]] = None) -> Dict["UncertaintyType", float]:
        members = include if include is not None else list(cls)
        # Normalize weights of selected subset
        weights = {m: m.default_weight for m in members}
        total = sum(weights.values()) or 1.0
        return {m: w / total for m, w in weights.items()}

    @classmethod
    def reducible_types(cls) -> List["UncertaintyType"]:
        return [m for m in cls if m.reducible]

    @classmethod
    def irreducible_types(cls) -> List["UncertaintyType"]:
        return [m for m in cls if not m.reducible]


@dataclass
class ConfidenceMetrics:
    """
    Comprehensive confidence and uncertainty metrics (enhanced).

    New fields / behaviors:
    - decision_id: Stable UUID for traceability
    - model_version, model_name: Provenance metadata
    - input_hash: Hash of normalized input context for audit linking
    - calibration_metadata: Raw calibration params or version tag
    - triggers: List of human review / escalation triggers
    - notes: Arbitrary annotation list
    - reliability_index: Derived dynamic property
    - flexible serialization helpers

    Value ranges enforced in __post_init__ (0.0–1.0 normalization).
    """
    raw_confidence: float
    calibrated_confidence: float
    confidence_level: ConfidenceLevel
    uncertainty_score: float
    uncertainty_breakdown: Dict[UncertaintyType, float]
    prediction_entropy: float
    model_agreement: float
    historical_accuracy: float
    risk_adjusted_confidence: float
    timestamp: datetime

    # New / extended metadata
    decision_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    model_version: Optional[str] = None
    model_name: Optional[str] = None
    input_hash: Optional[str] = None
    calibration_metadata: Optional[Dict[str, Any]] = None
    triggers: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)  # reserved for future
    schema_version: str = "2.0"

    def __post_init__(self):
        # Normalize / clamp numeric fields
        def _clamp(x: float) -> float:
            return max(0.0, min(1.0, float(x)))

        numeric_fields = [
            'raw_confidence', 'calibrated_confidence', 'uncertainty_score',
            'prediction_entropy', 'model_agreement', 'historical_accuracy',
            'risk_adjusted_confidence'
        ]
        for f in numeric_fields:
            val = getattr(self, f)
            if not isinstance(val, (int, float)):
                raise TypeError(f"{f} must be numeric, got {type(val)}")
            setattr(self, f, _clamp(val))

        # Convert uncertainty_breakdown keys if provided as strings
        converted = {}
        for k, v in self.uncertainty_breakdown.items():
            if isinstance(k, UncertaintyType):
                key = k
            else:
                key = UncertaintyType.from_str(str(k))
            converted[key] = _clamp(v)
        self.uncertainty_breakdown = converted

        # Recompute and ensure aggregate matches if large drift
        recomputed = self.total_uncertainty
        if abs(recomputed - self.uncertainty_score) > 0.15:
            logger.debug(f"Adjusting uncertainty_score {self.uncertainty_score:.3f} -> {recomputed:.3f}")
            self.uncertainty_score = recomputed

        if self.input_hash is None:
            # A stable hash could combine sorted uncertainty keys + raw_confidence
            base = json.dumps({
                "uc": self.uncertainty_breakdown_as_dict(),
                "rc": self.raw_confidence,
                "ts": self.timestamp.isoformat()
            }, sort_keys=True)
            self.input_hash = hashlib.sha256(base.encode('utf-8')).hexdigest()

    @property
    def total_uncertainty(self) -> float:
        # Weighted recomputation using present subset
        weights = UncertaintyType.weighted_members(self.uncertainty_breakdown.keys())
        return float(sum(self.uncertainty_breakdown[u] * weights.get(u, 0.0) for u in self.uncertainty_breakdown))

    @property
    def reliability_index(self) -> float:
        """
        Composite reliability heuristic:
        Blend of (risk_adjusted_confidence, model_agreement, historical_accuracy, inverse uncertainty).
        """
        inverse_uncertainty = 1.0 - self.uncertainty_score
        components = [
            self.risk_adjusted_confidence,
            self.model_agreement,
            self.historical_accuracy,
            inverse_uncertainty
        ]
        return round(float(sum(components) / len(components)), 4)

    def uncertainty_breakdown_as_dict(self) -> Dict[str, float]:
        return {u.code: v for u, v in self.uncertainty_breakdown.items()}

    def add_trigger(self, trigger: str):
        if trigger not in self.triggers:
            self.triggers.append(trigger)

    def add_note(self, note: str):
        self.notes.append(note)

    def escalate_required(self,
                          min_confidence: float = 0.80,
                          max_uncertainty: float = 0.40,
                          min_reliability: float = 0.75) -> bool:
        """
        Policy check for human escalation.
        Returns True if any risk dimension falls outside acceptable corridor.
        """
        if self.risk_adjusted_confidence < min_confidence:
            return True
        if self.uncertainty_score > max_uncertainty:
            return True
        if self.reliability_index < min_reliability:
            return True
        return False

    def explain(self, top_n: int = 3) -> str:
        """
        Provide a human-friendly explanation summarizing dominant uncertainty drivers.
        """
        sorted_unc = sorted(self.uncertainty_breakdown.items(), key=lambda kv: kv[1], reverse=True)
        drivers = ", ".join(f"{u.code}:{v:.2f}" for u, v in sorted_unc[:top_n])
        return (f"ConfidenceLevel={self.confidence_level.value} "
                f"(RiskAdj={self.risk_adjusted_confidence:.2f}, Reliability={self.reliability_index:.2f}). "
                f"Top Uncertainty Drivers: {drivers}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "decision_id": self.decision_id,
            "model_version": self.model_version,
            "model_name": self.model_name,
            "input_hash": self.input_hash,
            "raw_confidence": self.raw_confidence,
            "calibrated_confidence": self.calibrated_confidence,
            "confidence_level": self.confidence_level.value,
            "uncertainty_score": self.uncertainty_score,
            "uncertainty_breakdown": self.uncertainty_breakdown_as_dict(),
            "prediction_entropy": self.prediction_entropy,
            "model_agreement": self.model_agreement,
            "historical_accuracy": self.historical_accuracy,
            "risk_adjusted_confidence": self.risk_adjusted_confidence,
            "timestamp": self.timestamp.isoformat(),
            "reliability_index": self.reliability_index,
            "triggers": list(self.triggers),
            "notes": list(self.notes),
            "calibration_metadata": self.calibration_metadata,
            "extra": self.extra
        }

    def to_json(self, **kwargs) -> str:
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConfidenceMetrics":
        ts = data.get("timestamp")
        if isinstance(ts, str):
            timestamp = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        else:
            timestamp = datetime.now(timezone.utc)
        breakdown = data.get("uncertainty_breakdown", {})

        # Map potential legacy keys to current enum names:
        norm_breakdown = {}
        for k, v in breakdown.items():
            norm_breakdown[UncertaintyType.from_str(k)] = v

        return cls(
            raw_confidence=data["raw_confidence"],
            calibrated_confidence=data["calibrated_confidence"],
            confidence_level=ConfidenceLevel(data["confidence_level"]),
            uncertainty_score=data["uncertainty_score"],
            uncertainty_breakdown=norm_breakdown,
            prediction_entropy=data["prediction_entropy"],
            model_agreement=data["model_agreement"],
            historical_accuracy=data["historical_accuracy"],
            risk_adjusted_confidence=data["risk_adjusted_confidence"],
            timestamp=timestamp,
            decision_id=data.get("decision_id", str(uuid.uuid4())),
            model_version=data.get("model_version"),
            model_name=data.get("model_name"),
            input_hash=data.get("input_hash"),
            calibration_metadata=data.get("calibration_metadata"),
            triggers=data.get("triggers", []),
            notes=data.get("notes", []),
            extra=data.get("extra", {}),
            schema_version=data.get("schema_version", "2.0")
        )


class ClinicalConfidenceScorer:
    """
    Advanced confidence scoring system for clinical AI decisions (enhanced).
    Improvements:
      - Integrates enriched UncertaintyType weights
      - Optionally exposes reliability index & escalation logic
      - Backward compatible with prior API surface
    """
    def __init__(self,
                 calibration_data: Optional[Dict[str, Any]] = None,
                 model_performance_history: Optional[List[Dict[str, Any]]] = None,
                 custom_uncertainty_weights: Optional[Dict[UncertaintyType, float]] = None,
                 model_name: Optional[str] = None,
                 model_version: Optional[str] = None):
        self.calibration_data = calibration_data or {}
        self.model_performance_history = model_performance_history or []
        self.confidence_history: List[ConfidenceMetrics] = []

        # Default calibration parameters
        self.calibration_params = {
            'temperature': 1.0,
            'platt_scaling_a': 1.0,
            'platt_scaling_b': 0.0
        }
        if self.calibration_data:
            self._load_calibration_parameters()

        # Normalize custom weights if provided
        if custom_uncertainty_weights:
            total = sum(custom_uncertainty_weights.values()) or 1.0
            self.uncertainty_weights = {
                k: v / total for k, v in custom_uncertainty_weights.items()
            }
        else:
            self.uncertainty_weights = UncertaintyType.weighted_members()

        self.model_name = model_name
        self.model_version = model_version

    def score_confidence(self,
                         model_outputs: Dict[str, Any],
                         clinical_context: Dict[str, Any],
                         model_metadata: Optional[Dict[str, Any]] = None) -> ConfidenceMetrics:
        raw_confidence = self._extract_raw_confidence(model_outputs)
        uncertainty_breakdown = self._calculate_uncertainty_breakdown(model_outputs, clinical_context)
        uncertainty_score = self._aggregate_uncertainty_scores(uncertainty_breakdown)
        calibrated_confidence = self._calibrate_confidence(raw_confidence, clinical_context, uncertainty_score)

        prediction_entropy = self._calculate_prediction_entropy(model_outputs)
        model_agreement = self._calculate_model_agreement(model_outputs)
        historical_accuracy = self._get_historical_accuracy(clinical_context)
        risk_adjusted_confidence = self._calculate_risk_adjusted_confidence(calibrated_confidence,
                                                                            clinical_context,
                                                                            uncertainty_score)
        confidence_level = self._determine_confidence_level(risk_adjusted_confidence)

        metrics = ConfidenceMetrics(
            raw_confidence=raw_confidence,
            calibrated_confidence=calibrated_confidence,
            confidence_level=confidence_level,
            uncertainty_score=uncertainty_score,
            uncertainty_breakdown=uncertainty_breakdown,
            prediction_entropy=prediction_entropy,
            model_agreement=model_agreement,
            historical_accuracy=historical_accuracy,
            risk_adjusted_confidence=risk_adjusted_confidence,
            timestamp=datetime.now(timezone.utc),
            model_version=self.model_version,
            model_name=self.model_name,
            calibration_metadata=self.calibration_params
        )

        # Example automatic triggers
        if metrics.escalate_required():
            metrics.add_trigger("HUMAN_REVIEW_REQUIRED")
        if metrics.uncertainty_score > 0.6 and UncertaintyType.DISTRIBUTIONAL in metrics.uncertainty_breakdown:
            metrics.add_trigger("POTENTIAL_OOD")

        self.confidence_history.append(metrics)
        return metrics

    # -------- Existing (mostly unchanged) internal methods --------
    def _extract_raw_confidence(self, model_outputs: Dict[str, Any]) -> float:
        if 'confidence' in model_outputs:
            return float(model_outputs['confidence'])
        elif 'prediction_probabilities' in model_outputs:
            probs = model_outputs['prediction_probabilities']
            if isinstance(probs, (list, np.ndarray)):
                return float(np.max(probs))
            elif isinstance(probs, dict):
                return float(max(probs.values()))
        elif 'class_probabilities' in model_outputs:
            probs = model_outputs['class_probabilities']
            if isinstance(probs, dict):
                return float(max(probs.values()))
        logger.warning("No explicit confidence found in model outputs, using default 0.7")
        return 0.7

    def _calculate_uncertainty_breakdown(self,
                                         model_outputs: Dict[str, Any],
                                         clinical_context: Dict[str, Any]) -> Dict[UncertaintyType, float]:
        breakdown: Dict[UncertaintyType, float] = {}

        # Core uncertainties (existing logic)
        epistemic = float(model_outputs.get('model_variance',
                                            model_outputs.get('ensemble_disagreement', 0.0)))
        breakdown[UncertaintyType.EPISTEMIC] = min(max(epistemic, 0.0), 1.0)

        aleatoric = float(model_outputs.get('data_uncertainty',
                                            model_outputs.get('prediction_variance', 0.0)))
        breakdown[UncertaintyType.ALEATORIC] = min(max(aleatoric, 0.0), 1.0)

        distributional = self._calculate_distributional_uncertainty(model_outputs, clinical_context)
        breakdown[UncertaintyType.DISTRIBUTIONAL] = distributional

        temporal = self._calculate_temporal_uncertainty(clinical_context)
        breakdown[UncertaintyType.TEMPORAL] = temporal

        # New optional sources (heuristic examples)
        if 'concept_drift_score' in model_outputs:
            breakdown[UncertaintyType.CONCEPT_DRIFT] = float(model_outputs['concept_drift_score'])
        if 'population_shift_score' in model_outputs:
            breakdown[UncertaintyType.POPULATION_SHIFT] = float(model_outputs['population_shift_score'])
        if clinical_context.get('manual_entry_ratio') is not None:
            # Higher manual entry ratio -> higher human input uncertainty
            breakdown[UncertaintyType.HUMAN_INPUT] = min(
                1.0, float(clinical_context['manual_entry_ratio']) * 0.5
            )
        if 'sensor_calibration_delta' in model_outputs:
            breakdown[UncertaintyType.MEASUREMENT] = min(
                1.0, abs(float(model_outputs['sensor_calibration_delta']))
            )
        if 'ensemble_predictions' in model_outputs:
            # Use ensemble disagreement again but scaled to AGGREGATION bucket
            preds = model_outputs['ensemble_predictions']
            if isinstance(preds, (list, np.ndarray)) and len(preds) > 1:
                agg_var = float(np.std(preds))
                breakdown[UncertaintyType.AGGREGATION] = min(agg_var, 1.0)

        # Clamp
        for k in list(breakdown.keys()):
            breakdown[k] = max(0.0, min(1.0, float(breakdown[k])))

        return breakdown

    def _calculate_distributional_uncertainty(self,
                                              model_outputs: Dict[str, Any],
                                              clinical_context: Dict[str, Any]) -> float:
        if 'ood_score' in model_outputs:
            return float(model_outputs['ood_score'])

        unusual_factors = 0
        age = clinical_context.get('patient_age', 0)
        if age > 90 or age < 1:
            unusual_factors += 1
        if clinical_context.get('condition_severity') == 'CRITICAL':
            unusual_factors += 1
        if clinical_context.get('comorbidity_count', 0) > 5:
            unusual_factors += 1
        return min(unusual_factors * 0.2, 1.0)

    def _calculate_temporal_uncertainty(self, clinical_context: Dict[str, Any]) -> float:
        temporal_uncertainty = 0.0
        if clinical_context.get('symptom_onset_hours', 0) < 6:
            temporal_uncertainty += 0.1
        if clinical_context.get('condition_progression') == 'RAPID':
            temporal_uncertainty += 0.2
        return min(temporal_uncertainty, 1.0)

    def _aggregate_uncertainty_scores(self,
                                      uncertainty_breakdown: Dict[UncertaintyType, float]) -> float:
        if not uncertainty_breakdown:
            return 0.0
        # Use dynamic weights (intersection of configured weights & present keys)
        active_weights = {
            k: self.uncertainty_weights.get(k, k.default_weight)
            for k in uncertainty_breakdown.keys()
        }
        total = sum(active_weights.values()) or 1.0
        active_weights = {k: v / total for k, v in active_weights.items()}
        weighted_sum = sum(uncertainty_breakdown[u] * active_weights[u] for u in uncertainty_breakdown)
        return min(weighted_sum, 1.0)

    def _calibrate_confidence(self,
                              raw_confidence: float,
                              clinical_context: Dict[str, Any],
                              uncertainty_score: float) -> float:
        calibrated = raw_confidence / self.calibration_params['temperature']
        a = self.calibration_params['platt_scaling_a']
        b = self.calibration_params['platt_scaling_b']
        calibrated = 1.0 / (1.0 + np.exp(a * calibrated + b))
        calibrated = calibrated * (1.0 - uncertainty_score * 0.5)
        return max(0.01, min(0.99, calibrated))

    def _calculate_prediction_entropy(self, model_outputs: Dict[str, Any]) -> float:
        if 'class_probabilities' in model_outputs:
            probs = model_outputs['class_probabilities']
            if isinstance(probs, dict):
                probs = list(probs.values())
            probs = np.array(probs, dtype=float)
            probs = probs[probs > 0]
            if probs.size == 0:
                return 0.0
            entropy = -np.sum(probs * np.log2(probs))
            # Normalize by log2(num_classes) if desired; leaving raw for now.
            return float(entropy)
        return 0.0

    def _calculate_model_agreement(self, model_outputs: Dict[str, Any]) -> float:
        if 'ensemble_predictions' in model_outputs:
            predictions = model_outputs['ensemble_predictions']
            if isinstance(predictions, (list, np.ndarray)) and len(predictions) > 1:
                disagreement = np.std(predictions)
                agreement = 1.0 - min(disagreement, 1.0)
                return agreement
        return 1.0

    def _get_historical_accuracy(self, clinical_context: Dict[str, Any]) -> float:
        severity = clinical_context.get('condition_severity', 'MODERATE')
        complexity_scores = {
            'MINIMAL': 0.95,
            'LOW': 0.90,
            'MODERATE': 0.85,
            'HIGH': 0.75,
            'CRITICAL': 0.70
        }
        return complexity_scores.get(severity, 0.80)

    def _calculate_risk_adjusted_confidence(self,
                                            calibrated_confidence: float,
                                            clinical_context: Dict[str, Any],
                                            uncertainty_score: float) -> float:
        risk_adjustment = 1.0
        severity = clinical_context.get('condition_severity')
        if severity == 'CRITICAL':
            risk_adjustment *= 0.8
        elif severity == 'HIGH':
            risk_adjustment *= 0.9
        age = clinical_context.get('patient_age', 50)
        if age > 80 or age < 18:
            risk_adjustment *= 0.95
        comorbidities = clinical_context.get('comorbidity_count', 0)
        if comorbidities > 3:
            risk_adjustment *= 0.9
        adjusted_confidence = calibrated_confidence * risk_adjustment
        return max(0.01, min(0.99, adjusted_confidence))

    def _determine_confidence_level(self, confidence_score: float) -> ConfidenceLevel:
        if confidence_score >= 0.95:
            return ConfidenceLevel.VERY_HIGH
        elif confidence_score >= 0.80:
            return ConfidenceLevel.HIGH
        elif confidence_score >= 0.60:
            return ConfidenceLevel.MODERATE
        elif confidence_score >= 0.40:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW

    def _load_calibration_parameters(self):
        if 'temperature' in self.calibration_data:
            self.calibration_params['temperature'] = self.calibration_data['temperature']
        if 'platt_scaling' in self.calibration_data:
            platt = self.calibration_data['platt_scaling']
            self.calibration_params['platt_scaling_a'] = platt.get('a', 1.0)
            self.calibration_params['platt_scaling_b'] = platt.get('b', 0.0)

    def get_confidence_summary(self, hours_back: int = 24) -> Dict[str, Any]:
        cutoff = datetime.now(timezone.utc).timestamp() - (hours_back * 3600)
        recent_metrics = [m for m in self.confidence_history if m.timestamp.timestamp() > cutoff]
        if not recent_metrics:
            return {'error': 'No recent confidence data available'}

        return {
            'total_assessments': len(recent_metrics),
            'average_confidence': statistics.mean(m.risk_adjusted_confidence for m in recent_metrics),
            'confidence_distribution': {
                level.value: sum(1 for m in recent_metrics if m.confidence_level == level)
                for level in ConfidenceLevel
            },
            'average_uncertainty': statistics.mean(m.uncertainty_score for m in recent_metrics),
            'high_uncertainty_cases': sum(1 for m in recent_metrics if m.uncertainty_score > 0.7),
            'average_reliability_index': statistics.mean(m.reliability_index for m in recent_metrics),
            'escalations': sum(1 for m in recent_metrics if "HUMAN_REVIEW_REQUIRED" in m.triggers)
        }
