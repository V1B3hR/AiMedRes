"""
Clinical Confidence Scorer - Decision Certainty Metrics

Provides comprehensive confidence scoring with uncertainty quantification
for AI clinical decisions, supporting human oversight triggers.
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import statistics
from datetime import datetime, timezone


logger = logging.getLogger('duetmind.clinical_confidence_scorer')


class ConfidenceLevel(Enum):
    """Standardized confidence levels for clinical decisions"""
    VERY_LOW = "VERY_LOW"      # 0-40%
    LOW = "LOW"                # 40-60%
    MODERATE = "MODERATE"      # 60-80%
    HIGH = "HIGH"              # 80-95%
    VERY_HIGH = "VERY_HIGH"    # 95-100%


class UncertaintyType(Enum):
    """Types of uncertainty in clinical AI decisions"""
    EPISTEMIC = "EPISTEMIC"        # Model uncertainty (lack of knowledge)
    ALEATORIC = "ALEATORIC"        # Data uncertainty (inherent noise)
    DISTRIBUTIONAL = "DISTRIBUTIONAL"  # Out-of-distribution uncertainty
    TEMPORAL = "TEMPORAL"          # Time-related uncertainty


@dataclass
class ConfidenceMetrics:
    """Comprehensive confidence and uncertainty metrics"""
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


class ClinicalConfidenceScorer:
    """
    Advanced confidence scoring system for clinical AI decisions.
    
    Provides:
    - Uncertainty quantification across multiple dimensions
    - Confidence calibration based on historical performance
    - Risk-adjusted confidence scoring
    - Support for ensemble model confidence aggregation
    """
    
    def __init__(self, 
                 calibration_data: Optional[Dict[str, Any]] = None,
                 model_performance_history: Optional[List[Dict[str, Any]]] = None):
        """
        Initialize the confidence scorer.
        
        Args:
            calibration_data: Historical calibration data for confidence adjustment
            model_performance_history: Historical model performance data
        """
        self.calibration_data = calibration_data or {}
        self.model_performance_history = model_performance_history or []
        self.confidence_history = []
        
        # Default calibration parameters
        self.calibration_params = {
            'temperature': 1.0,
            'platt_scaling_a': 1.0,
            'platt_scaling_b': 0.0
        }
        
        # Load calibration parameters if available
        if self.calibration_data:
            self._load_calibration_parameters()
    
    def score_confidence(self,
                        model_outputs: Dict[str, Any],
                        clinical_context: Dict[str, Any],
                        model_metadata: Optional[Dict[str, Any]] = None) -> ConfidenceMetrics:
        """
        Calculate comprehensive confidence metrics for a clinical AI decision.
        
        Args:
            model_outputs: Raw model outputs including predictions and uncertainties
            clinical_context: Clinical context that may affect confidence
            model_metadata: Additional model information
            
        Returns:
            ConfidenceMetrics object with detailed confidence assessment
        """
        # Extract raw confidence
        raw_confidence = self._extract_raw_confidence(model_outputs)
        
        # Calculate uncertainty components
        uncertainty_breakdown = self._calculate_uncertainty_breakdown(
            model_outputs, clinical_context
        )
        
        # Calculate overall uncertainty score
        uncertainty_score = self._aggregate_uncertainty_scores(uncertainty_breakdown)
        
        # Apply confidence calibration
        calibrated_confidence = self._calibrate_confidence(
            raw_confidence, clinical_context, uncertainty_score
        )
        
        # Calculate additional metrics
        prediction_entropy = self._calculate_prediction_entropy(model_outputs)
        model_agreement = self._calculate_model_agreement(model_outputs)
        historical_accuracy = self._get_historical_accuracy(clinical_context)
        
        # Risk-adjusted confidence
        risk_adjusted_confidence = self._calculate_risk_adjusted_confidence(
            calibrated_confidence, clinical_context, uncertainty_score
        )
        
        # Determine confidence level
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
            timestamp=datetime.now(timezone.utc)
        )
        
        # Store for historical tracking
        self.confidence_history.append(metrics)
        
        return metrics
    
    def _extract_raw_confidence(self, model_outputs: Dict[str, Any]) -> float:
        """Extract raw confidence from model outputs"""
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
        
        # Fallback: assume moderate confidence if no explicit confidence provided
        logger.warning("No explicit confidence found in model outputs, using default 0.7")
        return 0.7
    
    def _calculate_uncertainty_breakdown(self, 
                                       model_outputs: Dict[str, Any],
                                       clinical_context: Dict[str, Any]) -> Dict[UncertaintyType, float]:
        """Calculate uncertainty across different dimensions"""
        uncertainty_breakdown = {}
        
        # Epistemic uncertainty (model knowledge uncertainty)
        epistemic = 0.0
        if 'model_variance' in model_outputs:
            epistemic = float(model_outputs['model_variance'])
        elif 'ensemble_disagreement' in model_outputs:
            epistemic = float(model_outputs['ensemble_disagreement'])
        uncertainty_breakdown[UncertaintyType.EPISTEMIC] = epistemic
        
        # Aleatoric uncertainty (data noise uncertainty)
        aleatoric = 0.0
        if 'data_uncertainty' in model_outputs:
            aleatoric = float(model_outputs['data_uncertainty'])
        elif 'prediction_variance' in model_outputs:
            aleatoric = float(model_outputs['prediction_variance'])
        uncertainty_breakdown[UncertaintyType.ALEATORIC] = aleatoric
        
        # Distributional uncertainty (out-of-distribution detection)
        distributional = self._calculate_distributional_uncertainty(
            model_outputs, clinical_context
        )
        uncertainty_breakdown[UncertaintyType.DISTRIBUTIONAL] = distributional
        
        # Temporal uncertainty (time-based factors)
        temporal = self._calculate_temporal_uncertainty(clinical_context)
        uncertainty_breakdown[UncertaintyType.TEMPORAL] = temporal
        
        return uncertainty_breakdown
    
    def _calculate_distributional_uncertainty(self,
                                            model_outputs: Dict[str, Any],
                                            clinical_context: Dict[str, Any]) -> float:
        """Calculate out-of-distribution uncertainty"""
        # Check for explicit OOD detection
        if 'ood_score' in model_outputs:
            return float(model_outputs['ood_score'])
        
        # Heuristic based on unusual clinical parameters
        unusual_factors = 0
        if clinical_context.get('patient_age', 0) > 90 or clinical_context.get('patient_age', 0) < 1:
            unusual_factors += 1
        
        if clinical_context.get('condition_severity') == 'CRITICAL':
            unusual_factors += 1
            
        if clinical_context.get('comorbidity_count', 0) > 5:
            unusual_factors += 1
        
        return min(unusual_factors * 0.2, 1.0)
    
    def _calculate_temporal_uncertainty(self, clinical_context: Dict[str, Any]) -> float:
        """Calculate temporal-based uncertainty factors"""
        temporal_uncertainty = 0.0
        
        # Recent symptom onset increases uncertainty
        if clinical_context.get('symptom_onset_hours', 0) < 6:
            temporal_uncertainty += 0.1
        
        # Rapid progression increases uncertainty
        if clinical_context.get('condition_progression') == 'RAPID':
            temporal_uncertainty += 0.2
        
        return min(temporal_uncertainty, 1.0)
    
    def _aggregate_uncertainty_scores(self, 
                                    uncertainty_breakdown: Dict[UncertaintyType, float]) -> float:
        """Aggregate individual uncertainty scores into overall uncertainty"""
        if not uncertainty_breakdown:
            return 0.0
        
        # Weighted combination of uncertainty types
        weights = {
            UncertaintyType.EPISTEMIC: 0.3,
            UncertaintyType.ALEATORIC: 0.2,
            UncertaintyType.DISTRIBUTIONAL: 0.3,
            UncertaintyType.TEMPORAL: 0.2
        }
        
        weighted_sum = sum(
            uncertainty_breakdown.get(unc_type, 0.0) * weight
            for unc_type, weight in weights.items()
        )
        
        return min(weighted_sum, 1.0)
    
    def _calibrate_confidence(self,
                            raw_confidence: float,
                            clinical_context: Dict[str, Any],
                            uncertainty_score: float) -> float:
        """Apply confidence calibration based on historical performance"""
        # Temperature scaling
        calibrated = raw_confidence / self.calibration_params['temperature']
        
        # Platt scaling
        a = self.calibration_params['platt_scaling_a']
        b = self.calibration_params['platt_scaling_b']
        calibrated = 1.0 / (1.0 + np.exp(a * calibrated + b))
        
        # Adjust based on uncertainty
        calibrated = calibrated * (1.0 - uncertainty_score * 0.5)
        
        return max(0.01, min(0.99, calibrated))
    
    def _calculate_prediction_entropy(self, model_outputs: Dict[str, Any]) -> float:
        """Calculate entropy of prediction distribution"""
        if 'class_probabilities' in model_outputs:
            probs = model_outputs['class_probabilities']
            if isinstance(probs, dict):
                probs = list(probs.values())
            elif isinstance(probs, (list, np.ndarray)):
                probs = np.array(probs)
            else:
                return 0.0
            
            # Calculate entropy
            probs = np.array(probs)
            probs = probs[probs > 0]  # Avoid log(0)
            entropy = -np.sum(probs * np.log2(probs))
            return float(entropy)
        
        return 0.0
    
    def _calculate_model_agreement(self, model_outputs: Dict[str, Any]) -> float:
        """Calculate agreement between ensemble models if available"""
        if 'ensemble_predictions' in model_outputs:
            predictions = model_outputs['ensemble_predictions']
            if isinstance(predictions, (list, np.ndarray)) and len(predictions) > 1:
                # Calculate standard deviation as disagreement measure
                disagreement = np.std(predictions)
                agreement = 1.0 - min(disagreement, 1.0)
                return agreement
        
        return 1.0  # Assume perfect agreement for single model
    
    def _get_historical_accuracy(self, clinical_context: Dict[str, Any]) -> float:
        """Get historical accuracy for similar cases"""
        # This would typically query a database of historical predictions
        # For now, return a default based on condition complexity
        
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
        """Adjust confidence based on clinical risk factors"""
        risk_adjustment = 1.0
        
        # Higher risk scenarios require higher confidence
        if clinical_context.get('condition_severity') == 'CRITICAL':
            risk_adjustment *= 0.8
        elif clinical_context.get('condition_severity') == 'HIGH':
            risk_adjustment *= 0.9
        
        # Age-based adjustments
        patient_age = clinical_context.get('patient_age', 50)
        if patient_age > 80 or patient_age < 18:
            risk_adjustment *= 0.95
        
        # Comorbidity adjustments
        comorbidities = clinical_context.get('comorbidity_count', 0)
        if comorbidities > 3:
            risk_adjustment *= 0.9
        
        adjusted_confidence = calibrated_confidence * risk_adjustment
        return max(0.01, min(0.99, adjusted_confidence))
    
    def _determine_confidence_level(self, confidence_score: float) -> ConfidenceLevel:
        """Map numerical confidence to categorical level"""
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
        """Load calibration parameters from historical data"""
        if 'temperature' in self.calibration_data:
            self.calibration_params['temperature'] = self.calibration_data['temperature']
        
        if 'platt_scaling' in self.calibration_data:
            platt = self.calibration_data['platt_scaling']
            self.calibration_params['platt_scaling_a'] = platt.get('a', 1.0)
            self.calibration_params['platt_scaling_b'] = platt.get('b', 0.0)
    
    def get_confidence_summary(self, hours_back: int = 24) -> Dict[str, Any]:
        """Get summary of recent confidence assessments"""
        cutoff = datetime.now(timezone.utc).timestamp() - (hours_back * 3600)
        recent_metrics = [
            m for m in self.confidence_history 
            if m.timestamp.timestamp() > cutoff
        ]
        
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
            'high_uncertainty_cases': sum(1 for m in recent_metrics if m.uncertainty_score > 0.7)
        }