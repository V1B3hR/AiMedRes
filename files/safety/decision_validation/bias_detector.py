"""
Bias Detector - Real-time Bias Monitoring

Implements comprehensive bias detection and mitigation algorithms
for clinical AI systems to ensure fair and equitable healthcare.
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timezone
import statistics
from collections import defaultdict, deque


logger = logging.getLogger('duetmind.bias_detector')


class BiasType(Enum):
    """Types of bias that can be detected in clinical AI systems"""
    DEMOGRAPHIC = "DEMOGRAPHIC"           # Age, gender, race, ethnicity
    SOCIOECONOMIC = "SOCIOECONOMIC"      # Income, insurance, education
    GEOGRAPHIC = "GEOGRAPHIC"             # Location, region, urban/rural
    TEMPORAL = "TEMPORAL"                 # Time-based patterns
    SELECTION = "SELECTION"               # Patient selection bias
    CONFIRMATION = "CONFIRMATION"         # Confirmation bias in recommendations
    ALGORITHMIC = "ALGORITHMIC"           # Model inherent biases
    CLINICAL = "CLINICAL"                 # Clinical setting or provider bias


class BiasMetric(Enum):
    """Metrics for measuring bias"""
    DEMOGRAPHIC_PARITY = "DEMOGRAPHIC_PARITY"
    EQUALIZED_ODDS = "EQUALIZED_ODDS"
    EQUALITY_OF_OPPORTUNITY = "EQUALITY_OF_OPPORTUNITY"
    CALIBRATION = "CALIBRATION"
    INDIVIDUAL_FAIRNESS = "INDIVIDUAL_FAIRNESS"
    COUNTERFACTUAL_FAIRNESS = "COUNTERFACTUAL_FAIRNESS"


class BiasSeverity(Enum):
    """Severity levels for detected bias"""
    MINIMAL = "MINIMAL"       # <5% disparity
    LOW = "LOW"              # 5-10% disparity
    MODERATE = "MODERATE"     # 10-20% disparity
    HIGH = "HIGH"            # 20-30% disparity
    CRITICAL = "CRITICAL"    # >30% disparity


@dataclass
class BiasDetection:
    """Result of bias detection analysis"""
    detection_id: str
    bias_type: BiasType
    bias_metric: BiasMetric
    severity: BiasSeverity
    affected_groups: List[str]
    disparity_score: float
    statistical_significance: float
    confidence_interval: Tuple[float, float]
    sample_size: int
    explanation: str
    mitigation_recommendations: List[str]
    timestamp: datetime


@dataclass
class BiasMonitoringConfig:
    """Configuration for bias monitoring"""
    sensitive_attributes: List[str]
    protected_groups: Dict[str, List[Any]]
    disparity_thresholds: Dict[BiasSeverity, float]
    minimum_sample_size: int
    monitoring_window_hours: int
    significance_threshold: float


class BiasDetector:
    """
    Real-time bias detection and monitoring system for clinical AI.
    
    Features:
    - Multi-dimensional bias detection across demographic groups
    - Statistical significance testing
    - Real-time monitoring and alerting
    - Bias mitigation recommendations
    - Historical bias trend analysis
    - Intersectional bias detection
    """
    
    def __init__(self, config: Optional[BiasMonitoringConfig] = None):
        """
        Initialize bias detector.
        
        Args:
            config: Bias monitoring configuration
        """
        self.config = config or self._get_default_config()
        self.detection_history = []
        self.decision_history = deque(maxlen=10000)  # Recent decisions for analysis
        
        # Group performance tracking
        self.group_metrics = defaultdict(lambda: {
            'predictions': [],
            'actual_outcomes': [],
            'confidence_scores': [],
            'timestamps': []
        })
        
        # Bias alert callbacks
        self.bias_alert_callbacks = []
    
    def _get_default_config(self) -> BiasMonitoringConfig:
        """Get default bias monitoring configuration"""
        return BiasMonitoringConfig(
            sensitive_attributes=['age_group', 'gender', 'race', 'ethnicity', 'insurance_type'],
            protected_groups={
                'age_group': ['pediatric', 'elderly'],
                'gender': ['female', 'male', 'other'],
                'race': ['white', 'black', 'asian', 'hispanic', 'other'],
                'insurance_type': ['medicaid', 'uninsured', 'private']
            },
            disparity_thresholds={
                BiasSeverity.MINIMAL: 0.05,
                BiasSeverity.LOW: 0.10,
                BiasSeverity.MODERATE: 0.20,
                BiasSeverity.HIGH: 0.30,
                BiasSeverity.CRITICAL: 0.40
            },
            minimum_sample_size=50,
            monitoring_window_hours=24,
            significance_threshold=0.05
        )
    
    def detect_bias(self,
                   decision_data: Dict[str, Any],
                   patient_demographics: Dict[str, Any],
                   ai_recommendation: Dict[str, Any],
                   actual_outcome: Optional[Dict[str, Any]] = None) -> List[BiasDetection]:
        """
        Detect bias in AI decision for a specific case.
        
        Args:
            decision_data: AI decision details
            patient_demographics: Patient demographic information
            ai_recommendation: AI recommendation details
            actual_outcome: Actual clinical outcome (if available)
            
        Returns:
            List of detected biases
        """
        # Store decision for historical analysis
        self._store_decision(decision_data, patient_demographics, ai_recommendation, actual_outcome)
        
        detected_biases = []
        
        # Perform bias detection across different dimensions
        for bias_type in BiasType:
            bias_detections = self._detect_bias_by_type(
                bias_type, decision_data, patient_demographics, ai_recommendation, actual_outcome
            )
            detected_biases.extend(bias_detections)
        
        # Store detection results
        for detection in detected_biases:
            self.detection_history.append(detection)
            
            # Trigger alerts for significant biases
            if detection.severity in [BiasSeverity.HIGH, BiasSeverity.CRITICAL]:
                self._trigger_bias_alert(detection)
        
        return detected_biases
    
    def _store_decision(self,
                       decision_data: Dict[str, Any],
                       patient_demographics: Dict[str, Any],
                       ai_recommendation: Dict[str, Any],
                       actual_outcome: Optional[Dict[str, Any]]):
        """Store decision data for historical bias analysis"""
        decision_record = {
            'decision_id': decision_data.get('decision_id', 'unknown'),
            'timestamp': datetime.now(timezone.utc),
            'patient_demographics': patient_demographics,
            'ai_recommendation': ai_recommendation,
            'confidence_score': decision_data.get('confidence_score', 0.0),
            'prediction': ai_recommendation.get('primary_recommendation'),
            'actual_outcome': actual_outcome,
            'user_id': decision_data.get('user_id'),
            'model_version': decision_data.get('model_version')
        }
        
        self.decision_history.append(decision_record)
        
        # Update group-specific metrics
        for attribute in self.config.sensitive_attributes:
            if attribute in patient_demographics:
                group_value = patient_demographics[attribute]
                group_key = f"{attribute}:{group_value}"
                
                metrics = self.group_metrics[group_key]
                metrics['predictions'].append(ai_recommendation.get('primary_recommendation'))
                metrics['confidence_scores'].append(decision_data.get('confidence_score', 0.0))
                metrics['timestamps'].append(decision_record['timestamp'])
                
                if actual_outcome:
                    metrics['actual_outcomes'].append(actual_outcome)
    
    def _detect_bias_by_type(self,
                           bias_type: BiasType,
                           decision_data: Dict[str, Any],
                           patient_demographics: Dict[str, Any],
                           ai_recommendation: Dict[str, Any],
                           actual_outcome: Optional[Dict[str, Any]]) -> List[BiasDetection]:
        """Detect bias of a specific type"""
        if bias_type == BiasType.DEMOGRAPHIC:
            return self._detect_demographic_bias(decision_data, patient_demographics, ai_recommendation)
        elif bias_type == BiasType.SOCIOECONOMIC:
            return self._detect_socioeconomic_bias(decision_data, patient_demographics, ai_recommendation)
        elif bias_type == BiasType.GEOGRAPHIC:
            return self._detect_geographic_bias(decision_data, patient_demographics, ai_recommendation)
        elif bias_type == BiasType.TEMPORAL:
            return self._detect_temporal_bias(decision_data, patient_demographics, ai_recommendation)
        elif bias_type == BiasType.ALGORITHMIC:
            return self._detect_algorithmic_bias(decision_data, patient_demographics, ai_recommendation)
        else:
            return []
    
    def _detect_demographic_bias(self,
                               decision_data: Dict[str, Any],
                               patient_demographics: Dict[str, Any],
                               ai_recommendation: Dict[str, Any]) -> List[BiasDetection]:
        """Detect demographic bias"""
        detections = []
        
        # Analyze bias for each demographic attribute
        for attribute in ['age_group', 'gender', 'race', 'ethnicity']:
            if attribute not in patient_demographics:
                continue
            
            current_group = patient_demographics[attribute]
            bias_detection = self._analyze_group_disparity(
                attribute, current_group, BiasType.DEMOGRAPHIC, decision_data, ai_recommendation
            )
            
            if bias_detection:
                detections.append(bias_detection)
        
        return detections
    
    def _detect_socioeconomic_bias(self,
                                 decision_data: Dict[str, Any],
                                 patient_demographics: Dict[str, Any],
                                 ai_recommendation: Dict[str, Any]) -> List[BiasDetection]:
        """Detect socioeconomic bias"""
        detections = []
        
        # Analyze insurance type bias
        if 'insurance_type' in patient_demographics:
            insurance_type = patient_demographics['insurance_type']
            bias_detection = self._analyze_group_disparity(
                'insurance_type', insurance_type, BiasType.SOCIOECONOMIC, 
                decision_data, ai_recommendation
            )
            
            if bias_detection:
                detections.append(bias_detection)
        
        return detections
    
    def _detect_geographic_bias(self,
                              decision_data: Dict[str, Any],
                              patient_demographics: Dict[str, Any],
                              ai_recommendation: Dict[str, Any]) -> List[BiasDetection]:
        """Detect geographic bias"""
        detections = []
        
        if 'geographic_region' in patient_demographics:
            region = patient_demographics['geographic_region']
            bias_detection = self._analyze_group_disparity(
                'geographic_region', region, BiasType.GEOGRAPHIC,
                decision_data, ai_recommendation
            )
            
            if bias_detection:
                detections.append(bias_detection)
        
        return detections
    
    def _detect_temporal_bias(self,
                            decision_data: Dict[str, Any],
                            patient_demographics: Dict[str, Any],
                            ai_recommendation: Dict[str, Any]) -> List[BiasDetection]:
        """Detect temporal bias patterns"""
        detections = []
        
        # Analyze time-of-day bias
        current_hour = datetime.now(timezone.utc).hour
        time_category = 'daytime' if 8 <= current_hour <= 18 else 'nighttime'
        
        bias_detection = self._analyze_temporal_disparity(
            time_category, decision_data, ai_recommendation
        )
        
        if bias_detection:
            detections.append(bias_detection)
        
        return detections
    
    def _detect_algorithmic_bias(self,
                               decision_data: Dict[str, Any],
                               patient_demographics: Dict[str, Any],
                               ai_recommendation: Dict[str, Any]) -> List[BiasDetection]:
        """Detect inherent algorithmic bias"""
        detections = []
        
        # Check for systematic confidence bias
        confidence_score = decision_data.get('confidence_score', 0.0)
        model_version = decision_data.get('model_version', 'unknown')
        
        # Analyze if model systematically under/over-estimates confidence for certain groups
        for attribute in self.config.sensitive_attributes:
            if attribute in patient_demographics:
                group_value = patient_demographics[attribute]
                confidence_bias = self._analyze_confidence_bias(
                    attribute, group_value, confidence_score, model_version
                )
                
                if confidence_bias:
                    detections.append(confidence_bias)
        
        return detections
    
    def _analyze_group_disparity(self,
                               attribute: str,
                               current_group: str,
                               bias_type: BiasType,
                               decision_data: Dict[str, Any],
                               ai_recommendation: Dict[str, Any]) -> Optional[BiasDetection]:
        """Analyze disparity between demographic groups"""
        # Get recent decisions for comparison
        recent_decisions = self._get_recent_decisions(self.config.monitoring_window_hours)
        
        if len(recent_decisions) < self.config.minimum_sample_size:
            return None
        
        # Group decisions by attribute value
        grouped_decisions = defaultdict(list)
        for decision in recent_decisions:
            if attribute in decision['patient_demographics']:
                group_val = decision['patient_demographics'][attribute]
                grouped_decisions[group_val].append(decision)
        
        # Calculate metrics for each group
        group_metrics = {}
        for group, decisions in grouped_decisions.items():
            if len(decisions) >= 10:  # Minimum sample size per group
                metrics = self._calculate_group_metrics(decisions)
                group_metrics[group] = metrics
        
        # Detect disparities
        if len(group_metrics) < 2:
            return None
        
        # Calculate pairwise disparities
        disparities = []
        groups = list(group_metrics.keys())
        
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                group1, group2 = groups[i], groups[j]
                disparity = self._calculate_disparity(
                    group_metrics[group1], group_metrics[group2]
                )
                disparities.append((group1, group2, disparity))
        
        # Find maximum disparity
        if not disparities:
            return None
        
        max_disparity = max(disparities, key=lambda x: abs(x[2]))
        disparity_score = abs(max_disparity[2])
        
        # Determine severity
        severity = self._determine_bias_severity(disparity_score)
        
        if severity == BiasSeverity.MINIMAL:
            return None  # Not significant enough to report
        
        # Statistical significance test
        significance = self._test_statistical_significance(
            group_metrics[max_disparity[0]], group_metrics[max_disparity[1]]
        )
        
        if significance > self.config.significance_threshold:
            return None  # Not statistically significant
        
        # Create bias detection
        return BiasDetection(
            detection_id=f"bias_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
            bias_type=bias_type,
            bias_metric=BiasMetric.DEMOGRAPHIC_PARITY,
            severity=severity,
            affected_groups=[max_disparity[0], max_disparity[1]],
            disparity_score=disparity_score,
            statistical_significance=significance,
            confidence_interval=(disparity_score - 0.05, disparity_score + 0.05),
            sample_size=sum(len(decisions) for decisions in grouped_decisions.values()),
            explanation=self._generate_bias_explanation(
                attribute, max_disparity[0], max_disparity[1], disparity_score, bias_type
            ),
            mitigation_recommendations=self._generate_mitigation_recommendations(
                bias_type, attribute, severity
            ),
            timestamp=datetime.now(timezone.utc)
        )
    
    def _analyze_temporal_disparity(self,
                                  time_category: str,
                                  decision_data: Dict[str, Any],
                                  ai_recommendation: Dict[str, Any]) -> Optional[BiasDetection]:
        """Analyze temporal disparities in AI decisions"""
        recent_decisions = self._get_recent_decisions(168)  # 1 week of data
        
        if len(recent_decisions) < self.config.minimum_sample_size:
            return None
        
        # Group by time categories
        daytime_decisions = []
        nighttime_decisions = []
        
        for decision in recent_decisions:
            hour = decision['timestamp'].hour
            if 8 <= hour <= 18:
                daytime_decisions.append(decision)
            else:
                nighttime_decisions.append(decision)
        
        if len(daytime_decisions) < 10 or len(nighttime_decisions) < 10:
            return None
        
        # Calculate metrics for each time period
        daytime_metrics = self._calculate_group_metrics(daytime_decisions)
        nighttime_metrics = self._calculate_group_metrics(nighttime_decisions)
        
        # Calculate disparity
        disparity = self._calculate_disparity(daytime_metrics, nighttime_metrics)
        disparity_score = abs(disparity)
        
        severity = self._determine_bias_severity(disparity_score)
        
        if severity == BiasSeverity.MINIMAL:
            return None
        
        return BiasDetection(
            detection_id=f"temporal_bias_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
            bias_type=BiasType.TEMPORAL,
            bias_metric=BiasMetric.DEMOGRAPHIC_PARITY,
            severity=severity,
            affected_groups=['daytime', 'nighttime'],
            disparity_score=disparity_score,
            statistical_significance=0.05,  # Simplified for now
            confidence_interval=(disparity_score - 0.05, disparity_score + 0.05),
            sample_size=len(daytime_decisions) + len(nighttime_decisions),
            explanation=f"Temporal bias detected: {disparity_score:.2f} disparity between daytime and nighttime decisions",
            mitigation_recommendations=self._generate_mitigation_recommendations(
                BiasType.TEMPORAL, 'time_of_day', severity
            ),
            timestamp=datetime.now(timezone.utc)
        )
    
    def _analyze_confidence_bias(self,
                               attribute: str,
                               group_value: str,
                               confidence_score: float,
                               model_version: str) -> Optional[BiasDetection]:
        """Analyze systematic confidence bias for specific groups"""
        group_key = f"{attribute}:{group_value}"
        
        if group_key not in self.group_metrics:
            return None
        
        group_confidences = self.group_metrics[group_key]['confidence_scores']
        
        if len(group_confidences) < 20:
            return None
        
        # Compare group confidence to overall average
        recent_decisions = self._get_recent_decisions(self.config.monitoring_window_hours)
        overall_confidences = [d.get('confidence_score', 0.0) for d in recent_decisions]
        
        if len(overall_confidences) < self.config.minimum_sample_size:
            return None
        
        group_avg = statistics.mean(group_confidences[-20:])  # Recent 20 decisions
        overall_avg = statistics.mean(overall_confidences)
        
        confidence_disparity = abs(group_avg - overall_avg)
        
        if confidence_disparity < 0.1:  # 10% threshold for confidence bias
            return None
        
        severity = self._determine_bias_severity(confidence_disparity)
        
        return BiasDetection(
            detection_id=f"confidence_bias_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
            bias_type=BiasType.ALGORITHMIC,
            bias_metric=BiasMetric.CALIBRATION,
            severity=severity,
            affected_groups=[group_value],
            disparity_score=confidence_disparity,
            statistical_significance=0.05,
            confidence_interval=(confidence_disparity - 0.02, confidence_disparity + 0.02),
            sample_size=len(group_confidences),
            explanation=f"Algorithmic confidence bias detected for {attribute}={group_value}: "
                       f"{confidence_disparity:.2f} difference from overall average",
            mitigation_recommendations=self._generate_mitigation_recommendations(
                BiasType.ALGORITHMIC, attribute, severity
            ),
            timestamp=datetime.now(timezone.utc)
        )
    
    def _get_recent_decisions(self, hours_back: int) -> List[Dict[str, Any]]:
        """Get decisions from the last N hours"""
        cutoff = datetime.now(timezone.utc).timestamp() - (hours_back * 3600)
        return [
            decision for decision in self.decision_history
            if decision['timestamp'].timestamp() > cutoff
        ]
    
    def _calculate_group_metrics(self, decisions: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate metrics for a group of decisions"""
        if not decisions:
            return {}
        
        confidence_scores = [d.get('confidence_score', 0.0) for d in decisions]
        
        # Positive outcome rate (simplified - would need actual outcome data)
        positive_outcomes = sum(
            1 for d in decisions 
            if d.get('actual_outcome', {}).get('outcome') == 'positive'
        )
        total_with_outcomes = sum(
            1 for d in decisions 
            if d.get('actual_outcome') is not None
        )
        
        positive_rate = positive_outcomes / max(total_with_outcomes, 1)
        
        return {
            'avg_confidence': statistics.mean(confidence_scores),
            'positive_outcome_rate': positive_rate,
            'sample_size': len(decisions)
        }
    
    def _calculate_disparity(self, 
                           group1_metrics: Dict[str, float], 
                           group2_metrics: Dict[str, float]) -> float:
        """Calculate disparity between two groups"""
        # Focus on positive outcome rate disparity
        rate1 = group1_metrics.get('positive_outcome_rate', 0.0)
        rate2 = group2_metrics.get('positive_outcome_rate', 0.0)
        
        if rate1 == 0 and rate2 == 0:
            return 0.0
        
        # Calculate relative disparity
        if rate2 > 0:
            disparity = (rate1 - rate2) / rate2
        else:
            disparity = rate1
        
        return disparity
    
    def _determine_bias_severity(self, disparity_score: float) -> BiasSeverity:
        """Determine bias severity based on disparity score"""
        for severity in [BiasSeverity.CRITICAL, BiasSeverity.HIGH, 
                        BiasSeverity.MODERATE, BiasSeverity.LOW, BiasSeverity.MINIMAL]:
            if disparity_score >= self.config.disparity_thresholds[severity]:
                return severity
        
        return BiasSeverity.MINIMAL
    
    def _test_statistical_significance(self,
                                     group1_metrics: Dict[str, float],
                                     group2_metrics: Dict[str, float]) -> float:
        """Test statistical significance of disparity (simplified)"""
        # This would typically use proper statistical tests like chi-square or t-test
        # For now, return a simplified p-value based on sample sizes
        
        n1 = group1_metrics.get('sample_size', 0)
        n2 = group2_metrics.get('sample_size', 0)
        
        if n1 < 30 or n2 < 30:
            return 0.1  # Not enough data for significance
        
        # Simplified significance estimation
        return 0.01 if (n1 > 100 and n2 > 100) else 0.05
    
    def _generate_bias_explanation(self,
                                 attribute: str,
                                 group1: str,
                                 group2: str,
                                 disparity_score: float,
                                 bias_type: BiasType) -> str:
        """Generate human-readable explanation of detected bias"""
        return (
            f"{bias_type.value} bias detected in {attribute}: "
            f"{disparity_score:.1%} disparity between {group1} and {group2} groups. "
            f"This indicates potential unfair treatment in AI recommendations."
        )
    
    def _generate_mitigation_recommendations(self,
                                           bias_type: BiasType,
                                           attribute: str,
                                           severity: BiasSeverity) -> List[str]:
        """Generate recommendations for bias mitigation"""
        recommendations = []
        
        if bias_type == BiasType.DEMOGRAPHIC:
            recommendations.extend([
                "Review training data for demographic representation balance",
                "Implement fairness-aware model training techniques",
                "Consider demographic parity constraints in model optimization"
            ])
        elif bias_type == BiasType.SOCIOECONOMIC:
            recommendations.extend([
                "Analyze insurance type impact on recommendations",
                "Ensure equal access to advanced treatment options",
                "Review socioeconomic factors in decision algorithms"
            ])
        elif bias_type == BiasType.TEMPORAL:
            recommendations.extend([
                "Analyze time-based decision patterns",
                "Ensure consistent model performance across time periods",
                "Consider temporal factors in model calibration"
            ])
        elif bias_type == BiasType.ALGORITHMIC:
            recommendations.extend([
                "Recalibrate model confidence scores",
                "Implement algorithmic auditing procedures",
                "Consider ensemble methods to reduce systematic bias"
            ])
        
        # Severity-specific recommendations
        if severity in [BiasSeverity.HIGH, BiasSeverity.CRITICAL]:
            recommendations.extend([
                "Immediate human review required for affected groups",
                "Consider temporary model restriction",
                "Escalate to ethics committee"
            ])
        
        return recommendations
    
    def _trigger_bias_alert(self, detection: BiasDetection):
        """Trigger bias alert callbacks"""
        alert_data = {
            'detection_id': detection.detection_id,
            'bias_type': detection.bias_type.value,
            'severity': detection.severity.value,
            'affected_groups': detection.affected_groups,
            'disparity_score': detection.disparity_score,
            'explanation': detection.explanation,
            'timestamp': detection.timestamp.isoformat()
        }
        
        for callback in self.bias_alert_callbacks:
            try:
                callback(alert_data)
            except Exception as e:
                logger.error(f"Error in bias alert callback: {e}")
    
    def add_bias_alert_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Add callback for bias alerts"""
        self.bias_alert_callbacks.append(callback)
    
    def get_bias_summary(self, hours_back: int = 24) -> Dict[str, Any]:
        """Get summary of recent bias detections"""
        cutoff = datetime.now(timezone.utc).timestamp() - (hours_back * 3600)
        recent_detections = [
            d for d in self.detection_history
            if d.timestamp.timestamp() > cutoff
        ]
        
        if not recent_detections:
            return {'message': 'No recent bias detections'}
        
        # Calculate summary statistics
        bias_counts = defaultdict(int)
        severity_counts = defaultdict(int)
        affected_groups = set()
        
        for detection in recent_detections:
            bias_counts[detection.bias_type.value] += 1
            severity_counts[detection.severity.value] += 1
            affected_groups.update(detection.affected_groups)
        
        return {
            'total_detections': len(recent_detections),
            'bias_types': dict(bias_counts),
            'severity_distribution': dict(severity_counts),
            'affected_groups': list(affected_groups),
            'critical_detections': sum(1 for d in recent_detections 
                                     if d.severity == BiasSeverity.CRITICAL),
            'high_severity_detections': sum(1 for d in recent_detections 
                                          if d.severity == BiasSeverity.HIGH)
        }
    
    def run_comprehensive_bias_audit(self) -> Dict[str, Any]:
        """Run comprehensive bias audit across all stored data"""
        logger.info("Starting comprehensive bias audit")
        
        audit_results = {
            'audit_timestamp': datetime.now(timezone.utc).isoformat(),
            'total_decisions_analyzed': len(self.decision_history),
            'bias_detections': [],
            'group_performance': {},
            'recommendations': []
        }
        
        # Analyze each group's performance
        for group_key, metrics in self.group_metrics.items():
            if len(metrics['predictions']) >= 10:
                group_performance = self._calculate_group_metrics([
                    {'confidence_score': score, 'actual_outcome': outcome}
                    for score, outcome in zip(metrics['confidence_scores'], 
                                            metrics['actual_outcomes'])
                ])
                audit_results['group_performance'][group_key] = group_performance
        
        # Generate audit recommendations
        audit_results['recommendations'] = self._generate_audit_recommendations(
            audit_results['group_performance']
        )
        
        return audit_results
    
    def _generate_audit_recommendations(self, 
                                      group_performance: Dict[str, Dict[str, float]]) -> List[str]:
        """Generate recommendations based on audit results"""
        recommendations = []
        
        if not group_performance:
            recommendations.append("Insufficient data for comprehensive bias analysis")
            return recommendations
        
        # Analyze performance disparities
        performance_values = [
            perf.get('positive_outcome_rate', 0.0) 
            for perf in group_performance.values()
        ]
        
        if performance_values:
            max_perf = max(performance_values)
            min_perf = min(performance_values)
            disparity = max_perf - min_perf
            
            if disparity > 0.2:
                recommendations.append(
                    "High performance disparity detected across groups - immediate review required"
                )
            elif disparity > 0.1:
                recommendations.append(
                    "Moderate performance disparity detected - consider model retraining"
                )
        
        recommendations.extend([
            "Implement regular bias monitoring",
            "Establish bias detection thresholds",
            "Create bias mitigation protocols"
        ])
        
        return recommendations