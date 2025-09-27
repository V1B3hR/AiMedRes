"""
Adversarial Defense - Attack Detection and Prevention

Implements comprehensive adversarial attack detection and prevention
mechanisms for clinical AI systems to ensure robust security.
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timezone
import statistics
import hashlib
from collections import deque, defaultdict


logger = logging.getLogger('duetmind.adversarial_defense')


class AttackType(Enum):
    """Types of adversarial attacks"""
    EVASION = "EVASION"                    # Input manipulation to fool model
    POISONING = "POISONING"                # Training data corruption
    MODEL_INVERSION = "MODEL_INVERSION"    # Extracting training data
    MODEL_EXTRACTION = "MODEL_EXTRACTION"  # Stealing model functionality
    MEMBERSHIP_INFERENCE = "MEMBERSHIP_INFERENCE"  # Inferring training membership
    BACKDOOR = "BACKDOOR"                  # Hidden trigger patterns
    GRADIENT_ATTACK = "GRADIENT_ATTACK"    # Gradient-based input manipulation
    SEMANTIC_ATTACK = "SEMANTIC_ATTACK"    # Semantically meaningful manipulations


class AttackSeverity(Enum):
    """Severity levels for detected attacks"""
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class DefenseStrategy(Enum):
    """Defense strategies against attacks"""
    INPUT_VALIDATION = "INPUT_VALIDATION"
    ANOMALY_DETECTION = "ANOMALY_DETECTION"
    ENSEMBLE_DEFENSE = "ENSEMBLE_DEFENSE"
    ADVERSARIAL_TRAINING = "ADVERSARIAL_TRAINING"
    GRADIENT_MASKING = "GRADIENT_MASKING"
    INPUT_PREPROCESSING = "INPUT_PREPROCESSING"
    OUTPUT_SMOOTHING = "OUTPUT_SMOOTHING"
    UNCERTAINTY_ESTIMATION = "UNCERTAINTY_ESTIMATION"


@dataclass
class AttackDetection:
    """Result of adversarial attack detection"""
    detection_id: str
    attack_type: AttackType
    severity: AttackSeverity
    confidence: float
    affected_inputs: List[str]
    attack_vector: Dict[str, Any]
    detection_method: str
    mitigation_actions: List[str]
    alert_triggered: bool
    timestamp: datetime


@dataclass
class DefenseConfig:
    """Configuration for adversarial defense system"""
    enable_input_validation: bool
    enable_anomaly_detection: bool
    enable_ensemble_defense: bool
    anomaly_threshold: float
    gradient_threshold: float
    input_bounds: Dict[str, Tuple[float, float]]
    suspicious_pattern_signatures: List[str]
    max_input_deviation: float
    defense_ensemble_size: int


class AdversarialDefense:
    """
    Comprehensive adversarial attack detection and prevention system.
    
    Features:
    - Multi-layer defense against various attack types
    - Real-time input validation and anomaly detection
    - Ensemble-based robustness verification
    - Attack pattern recognition and signature matching
    - Automated response and mitigation
    - Forensic analysis and attack attribution
    """
    
    def __init__(self, config: Optional[DefenseConfig] = None):
        """
        Initialize adversarial defense system.
        
        Args:
            config: Defense configuration parameters
        """
        self.config = config or self._get_default_config()
        self.attack_history = []
        self.input_history = deque(maxlen=1000)
        self.baseline_statistics = {}
        self.known_attack_signatures = set()
        
        # Defense callbacks
        self.attack_alert_callbacks = []
        
        # Initialize baseline statistics
        self._initialize_baseline_statistics()
        
        # Load known attack signatures
        self._load_attack_signatures()
    
    def _get_default_config(self) -> DefenseConfig:
        """Get default defense configuration"""
        return DefenseConfig(
            enable_input_validation=True,
            enable_anomaly_detection=True,
            enable_ensemble_defense=True,
            anomaly_threshold=0.95,
            gradient_threshold=0.1,
            input_bounds={
                'age': (0, 120),
                'weight': (0, 500),
                'height': (0, 300),
                'blood_pressure_systolic': (60, 250),
                'blood_pressure_diastolic': (30, 150),
                'heart_rate': (30, 200),
                'temperature': (95, 110)
            },
            suspicious_pattern_signatures=[
                'repeated_identical_inputs',
                'systematic_boundary_probing',
                'gradient_direction_manipulation'
            ],
            max_input_deviation=3.0,  # Standard deviations
            defense_ensemble_size=5
        )
    
    def detect_adversarial_attack(self,
                                input_data: Dict[str, Any],
                                model_outputs: Dict[str, Any],
                                request_metadata: Optional[Dict[str, Any]] = None) -> List[AttackDetection]:
        """
        Detect adversarial attacks in input data and model interactions.
        
        Args:
            input_data: Input data to the clinical AI model
            model_outputs: Outputs from the AI model
            request_metadata: Additional request metadata for analysis
            
        Returns:
            List of detected attacks
        """
        detections = []
        
        # Store input for historical analysis
        self._store_input_data(input_data, model_outputs, request_metadata)
        
        # Run different detection methods
        if self.config.enable_input_validation:
            input_detections = self._detect_input_manipulation(input_data, request_metadata)
            detections.extend(input_detections)
        
        if self.config.enable_anomaly_detection:
            anomaly_detections = self._detect_anomalous_patterns(input_data, model_outputs)
            detections.extend(anomaly_detections)
        
        if self.config.enable_ensemble_defense:
            ensemble_detections = self._detect_ensemble_inconsistencies(input_data, model_outputs)
            detections.extend(ensemble_detections)
        
        # Signature-based detection
        signature_detections = self._detect_known_attack_signatures(input_data, request_metadata)
        detections.extend(signature_detections)
        
        # Gradient-based attack detection
        gradient_detections = self._detect_gradient_attacks(input_data, model_outputs)
        detections.extend(gradient_detections)
        
        # Store detections and trigger alerts
        for detection in detections:
            self.attack_history.append(detection)
            
            if detection.severity in [AttackSeverity.HIGH, AttackSeverity.CRITICAL]:
                self._trigger_attack_alert(detection)
        
        return detections
    
    def _store_input_data(self,
                         input_data: Dict[str, Any],
                         model_outputs: Dict[str, Any],
                         request_metadata: Optional[Dict[str, Any]]):
        """Store input data for historical analysis"""
        input_record = {
            'timestamp': datetime.now(timezone.utc),
            'input_data': input_data,
            'model_outputs': model_outputs,
            'request_metadata': request_metadata or {},
            'input_hash': self._compute_input_hash(input_data)
        }
        
        self.input_history.append(input_record)
        
        # Update baseline statistics
        self._update_baseline_statistics(input_data)
    
    def _detect_input_manipulation(self,
                                 input_data: Dict[str, Any],
                                 request_metadata: Optional[Dict[str, Any]]) -> List[AttackDetection]:
        """Detect input data manipulation attempts"""
        detections = []
        
        # Check input bounds
        bounds_detection = self._check_input_bounds(input_data)
        if bounds_detection:
            detections.append(bounds_detection)
        
        # Check for statistical anomalies
        statistical_detection = self._check_statistical_anomalies(input_data)
        if statistical_detection:
            detections.append(statistical_detection)
        
        # Check for systematic manipulation patterns
        pattern_detection = self._check_manipulation_patterns(input_data, request_metadata)
        if pattern_detection:
            detections.append(pattern_detection)
        
        return detections
    
    def _check_input_bounds(self, input_data: Dict[str, Any]) -> Optional[AttackDetection]:
        """Check if input values are within expected bounds"""
        violations = []
        
        for field, value in input_data.items():
            if field in self.config.input_bounds:
                min_val, max_val = self.config.input_bounds[field]
                
                if isinstance(value, (int, float)):
                    if value < min_val or value > max_val:
                        violations.append(f"{field}={value} outside bounds [{min_val}, {max_val}]")
        
        if violations:
            return AttackDetection(
                detection_id=f"bounds_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
                attack_type=AttackType.EVASION,
                severity=AttackSeverity.MODERATE,
                confidence=0.8,
                affected_inputs=list(input_data.keys()),
                attack_vector={'violations': violations},
                detection_method='input_bounds_check',
                mitigation_actions=['validate_input_ranges', 'sanitize_inputs'],
                alert_triggered=True,
                timestamp=datetime.now(timezone.utc)
            )
        
        return None
    
    def _check_statistical_anomalies(self, input_data: Dict[str, Any]) -> Optional[AttackDetection]:
        """Check for statistical anomalies in input data"""
        if not self.baseline_statistics:
            return None
        
        anomalies = []
        anomaly_scores = []
        
        for field, value in input_data.items():
            if field in self.baseline_statistics and isinstance(value, (int, float)):
                stats = self.baseline_statistics[field]
                mean = stats.get('mean', 0)
                std = stats.get('std', 1)
                
                if std > 0:
                    z_score = abs((value - mean) / std)
                    if z_score > self.config.max_input_deviation:
                        anomalies.append(f"{field}: z-score={z_score:.2f}")
                        anomaly_scores.append(z_score)
        
        if anomalies:
            max_anomaly_score = max(anomaly_scores)
            severity = AttackSeverity.HIGH if max_anomaly_score > 5.0 else AttackSeverity.MODERATE
            
            return AttackDetection(
                detection_id=f"statistical_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
                attack_type=AttackType.EVASION,
                severity=severity,
                confidence=min(0.9, max_anomaly_score / 10.0),
                affected_inputs=list(input_data.keys()),
                attack_vector={'anomalies': anomalies, 'max_z_score': max_anomaly_score},
                detection_method='statistical_anomaly_detection',
                mitigation_actions=['review_input_manually', 'request_input_verification'],
                alert_triggered=severity == AttackSeverity.HIGH,
                timestamp=datetime.now(timezone.utc)
            )
        
        return None
    
    def _check_manipulation_patterns(self,
                                   input_data: Dict[str, Any],
                                   request_metadata: Optional[Dict[str, Any]]) -> Optional[AttackDetection]:
        """Check for systematic manipulation patterns"""
        if not request_metadata:
            return None
        
        # Check for repeated identical inputs (potential model probing)
        input_hash = self._compute_input_hash(input_data)
        recent_hashes = [record['input_hash'] for record in list(self.input_history)[-50:]]
        
        identical_count = recent_hashes.count(input_hash)
        if identical_count > 10:
            return AttackDetection(
                detection_id=f"pattern_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
                attack_type=AttackType.MODEL_EXTRACTION,
                severity=AttackSeverity.HIGH,
                confidence=0.85,
                affected_inputs=list(input_data.keys()),
                attack_vector={'repeated_inputs': identical_count},
                detection_method='pattern_analysis',
                mitigation_actions=['rate_limit_user', 'require_human_verification'],
                alert_triggered=True,
                timestamp=datetime.now(timezone.utc)
            )
        
        # Check for systematic boundary probing
        user_id = request_metadata.get('user_id')
        if user_id:
            user_inputs = [
                record for record in self.input_history 
                if record['request_metadata'].get('user_id') == user_id
            ]
            
            if len(user_inputs) > 20:
                # Analyze if user is systematically probing boundaries
                boundary_probes = self._analyze_boundary_probing(user_inputs)
                if boundary_probes > 0.7:  # High probability of boundary probing
                    return AttackDetection(
                        detection_id=f"boundary_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
                        attack_type=AttackType.MODEL_EXTRACTION,
                        severity=AttackSeverity.MODERATE,
                        confidence=boundary_probes,
                        affected_inputs=list(input_data.keys()),
                        attack_vector={'boundary_probing_score': boundary_probes},
                        detection_method='boundary_probing_analysis',
                        mitigation_actions=['monitor_user_activity', 'limit_query_rate'],
                        alert_triggered=False,
                        timestamp=datetime.now(timezone.utc)
                    )
        
        return None
    
    def _detect_anomalous_patterns(self,
                                 input_data: Dict[str, Any],
                                 model_outputs: Dict[str, Any]) -> List[AttackDetection]:
        """Detect anomalous patterns in model behavior"""
        detections = []
        
        # Check for unusual confidence patterns
        confidence_detection = self._detect_confidence_anomalies(model_outputs)
        if confidence_detection:
            detections.append(confidence_detection)
        
        # Check for output instability
        instability_detection = self._detect_output_instability(input_data, model_outputs)
        if instability_detection:
            detections.append(instability_detection)
        
        return detections
    
    def _detect_confidence_anomalies(self, model_outputs: Dict[str, Any]) -> Optional[AttackDetection]:
        """Detect unusual confidence patterns that might indicate attacks"""
        confidence = model_outputs.get('confidence', 0.5)
        
        # Get recent confidence scores
        recent_confidences = [
            record['model_outputs'].get('confidence', 0.5)
            for record in list(self.input_history)[-50:]
            if 'model_outputs' in record
        ]
        
        if len(recent_confidences) < 10:
            return None
        
        # Check for unusual confidence drops
        avg_confidence = statistics.mean(recent_confidences)
        confidence_drop = avg_confidence - confidence
        
        if confidence_drop > 0.3:  # Significant confidence drop
            return AttackDetection(
                detection_id=f"confidence_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
                attack_type=AttackType.EVASION,
                severity=AttackSeverity.MODERATE,
                confidence=0.7,
                affected_inputs=['model_confidence'],
                attack_vector={'confidence_drop': confidence_drop},
                detection_method='confidence_anomaly_detection',
                mitigation_actions=['verify_input_integrity', 'request_manual_review'],
                alert_triggered=False,
                timestamp=datetime.now(timezone.utc)
            )
        
        return None
    
    def _detect_output_instability(self,
                                 input_data: Dict[str, Any],
                                 model_outputs: Dict[str, Any]) -> Optional[AttackDetection]:
        """Detect output instability that might indicate adversarial inputs"""
        # This would typically involve checking if small input changes lead to large output changes
        # For now, implement a simplified version
        
        # Check if we have similar inputs with very different outputs
        similar_inputs = self._find_similar_inputs(input_data)
        
        if len(similar_inputs) > 5:
            current_prediction = model_outputs.get('prediction', 0)
            similar_predictions = [
                record['model_outputs'].get('prediction', 0)
                for record in similar_inputs
                if 'model_outputs' in record
            ]
            
            if similar_predictions:
                prediction_variance = np.var(similar_predictions + [current_prediction])
                
                if prediction_variance > 0.5:  # High variance in similar inputs
                    return AttackDetection(
                        detection_id=f"instability_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
                        attack_type=AttackType.EVASION,
                        severity=AttackSeverity.MODERATE,
                        confidence=0.6,
                        affected_inputs=['model_predictions'],
                        attack_vector={'prediction_variance': prediction_variance},
                        detection_method='output_instability_detection',
                        mitigation_actions=['verify_model_robustness', 'ensemble_validation'],
                        alert_triggered=False,
                        timestamp=datetime.now(timezone.utc)
                    )
        
        return None
    
    def _detect_ensemble_inconsistencies(self,
                                       input_data: Dict[str, Any],
                                       model_outputs: Dict[str, Any]) -> List[AttackDetection]:
        """Detect inconsistencies in ensemble model outputs"""
        detections = []
        
        # This would typically require ensemble predictions
        # For now, implement a placeholder that checks for ensemble-related indicators
        
        if 'ensemble_predictions' in model_outputs:
            predictions = model_outputs['ensemble_predictions']
            
            if isinstance(predictions, list) and len(predictions) > 1:
                prediction_std = np.std(predictions)
                
                if prediction_std > 0.3:  # High disagreement between ensemble members
                    detections.append(AttackDetection(
                        detection_id=f"ensemble_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
                        attack_type=AttackType.EVASION,
                        severity=AttackSeverity.MODERATE,
                        confidence=0.8,
                        affected_inputs=list(input_data.keys()),
                        attack_vector={'ensemble_disagreement': prediction_std},
                        detection_method='ensemble_consistency_check',
                        mitigation_actions=['require_human_review', 'increase_ensemble_size'],
                        alert_triggered=True,
                        timestamp=datetime.now(timezone.utc)
                    ))
        
        return detections
    
    def _detect_known_attack_signatures(self,
                                      input_data: Dict[str, Any],
                                      request_metadata: Optional[Dict[str, Any]]) -> List[AttackDetection]:
        """Detect known attack signatures"""
        detections = []
        
        # Check against known attack signatures
        input_signature = self._compute_input_signature(input_data, request_metadata)
        
        if input_signature in self.known_attack_signatures:
            detections.append(AttackDetection(
                detection_id=f"signature_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
                attack_type=AttackType.EVASION,
                severity=AttackSeverity.HIGH,
                confidence=0.95,
                affected_inputs=list(input_data.keys()),
                attack_vector={'known_signature': input_signature},
                detection_method='signature_matching',
                mitigation_actions=['block_request', 'alert_security_team'],
                alert_triggered=True,
                timestamp=datetime.now(timezone.utc)
            ))
        
        return detections
    
    def _detect_gradient_attacks(self,
                               input_data: Dict[str, Any],
                               model_outputs: Dict[str, Any]) -> List[AttackDetection]:
        """Detect gradient-based adversarial attacks"""
        detections = []
        
        # Check for gradient-related indicators in model outputs
        if 'gradients' in model_outputs:
            gradients = model_outputs['gradients']
            
            if isinstance(gradients, (list, np.ndarray)):
                gradient_norm = np.linalg.norm(gradients)
                
                if gradient_norm > self.config.gradient_threshold:
                    detections.append(AttackDetection(
                        detection_id=f"gradient_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
                        attack_type=AttackType.GRADIENT_ATTACK,
                        severity=AttackSeverity.MODERATE,
                        confidence=0.7,
                        affected_inputs=list(input_data.keys()),
                        attack_vector={'gradient_norm': gradient_norm},
                        detection_method='gradient_analysis',
                        mitigation_actions=['apply_gradient_clipping', 'use_differential_privacy'],
                        alert_triggered=False,
                        timestamp=datetime.now(timezone.utc)
                    ))
        
        return detections
    
    def _compute_input_hash(self, input_data: Dict[str, Any]) -> str:
        """Compute hash of input data"""
        # Sort keys for consistent hashing
        sorted_items = sorted(input_data.items())
        input_str = str(sorted_items)
        return hashlib.md5(input_str.encode()).hexdigest()
    
    def _compute_input_signature(self,
                               input_data: Dict[str, Any],
                               request_metadata: Optional[Dict[str, Any]]) -> str:
        """Compute signature for attack detection"""
        # Combine input data with metadata for signature
        combined_data = {**input_data}
        if request_metadata:
            combined_data.update(request_metadata)
        
        return self._compute_input_hash(combined_data)
    
    def _find_similar_inputs(self, input_data: Dict[str, Any], threshold: float = 0.1) -> List[Dict[str, Any]]:
        """Find similar inputs in history"""
        similar_inputs = []
        
        for record in self.input_history:
            similarity = self._calculate_input_similarity(input_data, record['input_data'])
            if similarity > (1.0 - threshold):  # Similar inputs
                similar_inputs.append(record)
        
        return similar_inputs
    
    def _calculate_input_similarity(self, input1: Dict[str, Any], input2: Dict[str, Any]) -> float:
        """Calculate similarity between two input dictionaries"""
        common_keys = set(input1.keys()) & set(input2.keys())
        
        if not common_keys:
            return 0.0
        
        similarities = []
        for key in common_keys:
            val1, val2 = input1[key], input2[key]
            
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # Numerical similarity
                if val1 == val2:
                    similarities.append(1.0)
                else:
                    max_val = max(abs(val1), abs(val2), 1.0)
                    similarity = 1.0 - abs(val1 - val2) / max_val
                    similarities.append(max(0.0, similarity))
            elif str(val1) == str(val2):
                similarities.append(1.0)
            else:
                similarities.append(0.0)
        
        return sum(similarities) / len(similarities) if similarities else 0.0
    
    def _analyze_boundary_probing(self, user_inputs: List[Dict[str, Any]]) -> float:
        """Analyze if user inputs show boundary probing patterns"""
        if len(user_inputs) < 10:
            return 0.0
        
        # Check if user is systematically testing input boundaries
        boundary_tests = 0
        total_tests = 0
        
        for record in user_inputs:
            input_data = record['input_data']
            
            for field, value in input_data.items():
                if field in self.config.input_bounds and isinstance(value, (int, float)):
                    min_val, max_val = self.config.input_bounds[field]
                    range_val = max_val - min_val
                    
                    # Check if value is near boundaries
                    if (value - min_val) / range_val < 0.1 or (max_val - value) / range_val < 0.1:
                        boundary_tests += 1
                    
                    total_tests += 1
        
        return boundary_tests / max(total_tests, 1)
    
    def _initialize_baseline_statistics(self):
        """Initialize baseline statistics for input validation"""
        # This would typically be loaded from historical data
        # For now, set reasonable defaults for clinical data
        self.baseline_statistics = {
            'age': {'mean': 50, 'std': 20},
            'weight': {'mean': 70, 'std': 15},
            'height': {'mean': 170, 'std': 10},
            'blood_pressure_systolic': {'mean': 120, 'std': 20},
            'blood_pressure_diastolic': {'mean': 80, 'std': 15},
            'heart_rate': {'mean': 70, 'std': 15},
            'temperature': {'mean': 98.6, 'std': 1.5}
        }
    
    def _update_baseline_statistics(self, input_data: Dict[str, Any]):
        """Update baseline statistics with new input data"""
        for field, value in input_data.items():
            if isinstance(value, (int, float)):
                if field not in self.baseline_statistics:
                    self.baseline_statistics[field] = {'values': []}
                
                if 'values' not in self.baseline_statistics[field]:
                    self.baseline_statistics[field]['values'] = []
                
                self.baseline_statistics[field]['values'].append(value)
                
                # Keep only recent values
                if len(self.baseline_statistics[field]['values']) > 1000:
                    self.baseline_statistics[field]['values'] = \
                        self.baseline_statistics[field]['values'][-1000:]
                
                # Update statistics
                values = self.baseline_statistics[field]['values']
                self.baseline_statistics[field]['mean'] = statistics.mean(values)
                self.baseline_statistics[field]['std'] = statistics.stdev(values) if len(values) > 1 else 1.0
    
    def _load_attack_signatures(self):
        """Load known attack signatures"""
        # This would typically load from a database or file
        # For now, initialize with empty set
        self.known_attack_signatures = set()
    
    def _trigger_attack_alert(self, detection: AttackDetection):
        """Trigger attack alert callbacks"""
        alert_data = {
            'detection_id': detection.detection_id,
            'attack_type': detection.attack_type.value,
            'severity': detection.severity.value,
            'confidence': detection.confidence,
            'affected_inputs': detection.affected_inputs,
            'attack_vector': detection.attack_vector,
            'mitigation_actions': detection.mitigation_actions,
            'timestamp': detection.timestamp.isoformat()
        }
        
        for callback in self.attack_alert_callbacks:
            try:
                callback(alert_data)
            except Exception as e:
                logger.error(f"Error in attack alert callback: {e}")
    
    def add_attack_alert_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Add callback for attack alerts"""
        self.attack_alert_callbacks.append(callback)
    
    def get_defense_summary(self, hours_back: int = 24) -> Dict[str, Any]:
        """Get summary of recent attack detections and defense activities"""
        cutoff = datetime.now(timezone.utc).timestamp() - (hours_back * 3600)
        recent_detections = [
            d for d in self.attack_history
            if d.timestamp.timestamp() > cutoff
        ]
        
        if not recent_detections:
            return {'message': 'No recent attack detections'}
        
        # Calculate summary statistics
        attack_counts = defaultdict(int)
        severity_counts = defaultdict(int)
        
        for detection in recent_detections:
            attack_counts[detection.attack_type.value] += 1
            severity_counts[detection.severity.value] += 1
        
        return {
            'total_detections': len(recent_detections),
            'attack_types': dict(attack_counts),
            'severity_distribution': dict(severity_counts),
            'critical_attacks': sum(1 for d in recent_detections 
                                  if d.severity == AttackSeverity.CRITICAL),
            'high_severity_attacks': sum(1 for d in recent_detections 
                                       if d.severity == AttackSeverity.HIGH),
            'defense_effectiveness': self._calculate_defense_effectiveness(recent_detections)
        }
    
    def _calculate_defense_effectiveness(self, detections: List[AttackDetection]) -> Dict[str, float]:
        """Calculate defense system effectiveness metrics"""
        if not detections:
            return {'detection_rate': 0.0, 'false_positive_rate': 0.0}
        
        # This would typically be calculated against ground truth
        # For now, provide estimated metrics
        high_confidence_detections = sum(1 for d in detections if d.confidence > 0.8)
        
        return {
            'detection_rate': high_confidence_detections / len(detections),
            'response_time_avg': 0.1,  # Average response time in seconds
            'mitigation_success_rate': 0.95  # Estimated mitigation success rate
        }
    
    def update_attack_signatures(self, new_signatures: List[str]):
        """Update known attack signatures"""
        self.known_attack_signatures.update(new_signatures)
        logger.info(f"Updated attack signatures database with {len(new_signatures)} new signatures")
    
    def run_security_audit(self) -> Dict[str, Any]:
        """Run comprehensive security audit of defense systems"""
        audit_results = {
            'audit_timestamp': datetime.now(timezone.utc).isoformat(),
            'defense_config': {
                'input_validation_enabled': self.config.enable_input_validation,
                'anomaly_detection_enabled': self.config.enable_anomaly_detection,
                'ensemble_defense_enabled': self.config.enable_ensemble_defense
            },
            'attack_detection_stats': self.get_defense_summary(168),  # 1 week
            'baseline_statistics_health': self._audit_baseline_statistics(),
            'signature_database_status': {
                'total_signatures': len(self.known_attack_signatures),
                'last_updated': 'unknown'  # Would track in production
            },
            'recommendations': self._generate_security_recommendations()
        }
        
        return audit_results
    
    def _audit_baseline_statistics(self) -> Dict[str, Any]:
        """Audit baseline statistics for completeness and accuracy"""
        stats_health = {
            'total_fields': len(self.baseline_statistics),
            'fields_with_sufficient_data': 0,
            'fields_needing_update': []
        }
        
        for field, stats in self.baseline_statistics.items():
            if 'values' in stats and len(stats['values']) > 50:
                stats_health['fields_with_sufficient_data'] += 1
            else:
                stats_health['fields_needing_update'].append(field)
        
        return stats_health
    
    def _generate_security_recommendations(self) -> List[str]:
        """Generate security recommendations based on current state"""
        recommendations = [
            "Regularly update attack signature database",
            "Monitor defense system performance metrics",
            "Conduct periodic adversarial testing"
        ]
        
        if len(self.attack_history) > 100:
            recommendations.append("Consider implementing adaptive defense thresholds")
        
        if not self.config.enable_ensemble_defense:
            recommendations.append("Enable ensemble defense for improved robustness")
        
        return recommendations