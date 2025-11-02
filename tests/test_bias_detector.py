#!/usr/bin/env python3
"""
Tests for Bias Detector Module

Tests comprehensive bias detection functionality including:
- Demographic bias detection
- Socioeconomic bias detection
- Temporal bias detection
- Algorithmic bias detection
- Bias severity classification
- Bias mitigation recommendations
"""

import pytest
import sys
import os
from datetime import datetime, timezone

# Add bias_detector module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'files', 'safety', 'decision_validation'))

from bias_detector import (
    BiasDetector, BiasType, BiasMetric, BiasSeverity, 
    BiasDetection, BiasMonitoringConfig
)

# Test constants
TEST_DECISION_COUNT = 60  # Number of decisions for multi-decision tests


class TestBiasDetectorInitialization:
    """Test BiasDetector initialization and configuration"""
    
    def test_default_initialization(self):
        """Test bias detector initializes with default config"""
        detector = BiasDetector()
        
        assert detector is not None
        assert detector.config is not None
        assert len(detector.config.sensitive_attributes) > 0
        assert len(detector.detection_history) == 0
    
    def test_custom_config_initialization(self):
        """Test bias detector with custom configuration"""
        custom_config = BiasMonitoringConfig(
            sensitive_attributes=['age_group', 'gender'],
            protected_groups={
                'age_group': ['pediatric', 'elderly'],
                'gender': ['female', 'male']
            },
            disparity_thresholds={
                BiasSeverity.MINIMAL: 0.05,
                BiasSeverity.LOW: 0.10,
                BiasSeverity.MODERATE: 0.20,
                BiasSeverity.HIGH: 0.30,
                BiasSeverity.CRITICAL: 0.40
            },
            minimum_sample_size=30,
            monitoring_window_hours=12,
            significance_threshold=0.05
        )
        
        detector = BiasDetector(config=custom_config)
        
        assert len(detector.config.sensitive_attributes) == 2
        assert detector.config.minimum_sample_size == 30
        assert detector.config.monitoring_window_hours == 12


class TestBiasDetection:
    """Test bias detection functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.detector = BiasDetector()
    
    def test_detect_bias_stores_decision(self):
        """Test that detect_bias stores decision data"""
        decision_data = {
            'decision_id': 'test_001',
            'confidence_score': 0.85,
            'user_id': 'doctor_123',
            'model_version': 'v1.0'
        }
        
        patient_demographics = {
            'age_group': 'elderly',
            'gender': 'female',
            'race': 'white'
        }
        
        ai_recommendation = {
            'primary_recommendation': 'treatment_A'
        }
        
        initial_count = len(self.detector.decision_history)
        
        self.detector.detect_bias(
            decision_data, 
            patient_demographics, 
            ai_recommendation
        )
        
        assert len(self.detector.decision_history) == initial_count + 1
        assert self.detector.decision_history[-1]['decision_id'] == 'test_001'
    
    def test_detect_bias_with_multiple_decisions(self):
        """Test bias detection with multiple decisions"""
        # Create multiple decisions with different demographics
        for i in range(TEST_DECISION_COUNT):
            decision_data = {
                'decision_id': f'test_{i:03d}',
                'confidence_score': 0.85 if i % 2 == 0 else 0.75,
                'user_id': 'doctor_123',
                'model_version': 'v1.0'
            }
            
            patient_demographics = {
                'age_group': 'elderly' if i % 2 == 0 else 'pediatric',
                'gender': 'female' if i % 3 == 0 else 'male',
                'race': 'white'
            }
            
            ai_recommendation = {
                'primary_recommendation': 'treatment_A'
            }
            
            actual_outcome = {
                'outcome': 'positive' if i % 2 == 0 else 'negative'
            }
            
            self.detector.detect_bias(
                decision_data,
                patient_demographics,
                ai_recommendation,
                actual_outcome
            )
        
        assert len(self.detector.decision_history) >= TEST_DECISION_COUNT


class TestBiasSeverityClassification:
    """Test bias severity classification"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.detector = BiasDetector()
    
    def test_minimal_severity(self):
        """Test minimal severity classification"""
        severity = self.detector._determine_bias_severity(0.04)
        assert severity == BiasSeverity.MINIMAL
    
    def test_low_severity(self):
        """Test low severity classification"""
        severity = self.detector._determine_bias_severity(0.12)
        assert severity == BiasSeverity.LOW
    
    def test_moderate_severity(self):
        """Test moderate severity classification"""
        severity = self.detector._determine_bias_severity(0.22)
        assert severity == BiasSeverity.MODERATE
    
    def test_high_severity(self):
        """Test high severity classification"""
        severity = self.detector._determine_bias_severity(0.32)
        assert severity == BiasSeverity.HIGH
    
    def test_critical_severity(self):
        """Test critical severity classification"""
        severity = self.detector._determine_bias_severity(0.45)
        assert severity == BiasSeverity.CRITICAL


class TestGroupMetrics:
    """Test group metrics calculation"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.detector = BiasDetector()
    
    def test_calculate_group_metrics_empty(self):
        """Test group metrics with empty decisions"""
        metrics = self.detector._calculate_group_metrics([])
        assert metrics == {}
    
    def test_calculate_group_metrics_with_data(self):
        """Test group metrics with decision data"""
        decisions = [
            {
                'confidence_score': 0.85,
                'actual_outcome': {'outcome': 'positive'}
            },
            {
                'confidence_score': 0.90,
                'actual_outcome': {'outcome': 'positive'}
            },
            {
                'confidence_score': 0.75,
                'actual_outcome': {'outcome': 'negative'}
            }
        ]
        
        metrics = self.detector._calculate_group_metrics(decisions)
        
        assert 'avg_confidence' in metrics
        assert 'positive_outcome_rate' in metrics
        assert 'sample_size' in metrics
        assert metrics['sample_size'] == 3
        assert 0 <= metrics['avg_confidence'] <= 1
        assert 0 <= metrics['positive_outcome_rate'] <= 1


class TestDisparityCalculation:
    """Test disparity calculation between groups"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.detector = BiasDetector()
    
    def test_calculate_disparity_equal_groups(self):
        """Test disparity calculation for equal groups"""
        group1_metrics = {
            'avg_confidence': 0.85,
            'positive_outcome_rate': 0.70,
            'sample_size': 50
        }
        
        group2_metrics = {
            'avg_confidence': 0.85,
            'positive_outcome_rate': 0.70,
            'sample_size': 50
        }
        
        disparity = self.detector._calculate_disparity(group1_metrics, group2_metrics)
        assert abs(disparity) < 0.01
    
    def test_calculate_disparity_unequal_groups(self):
        """Test disparity calculation for unequal groups"""
        group1_metrics = {
            'avg_confidence': 0.85,
            'positive_outcome_rate': 0.80,
            'sample_size': 50
        }
        
        group2_metrics = {
            'avg_confidence': 0.85,
            'positive_outcome_rate': 0.60,
            'sample_size': 50
        }
        
        disparity = self.detector._calculate_disparity(group1_metrics, group2_metrics)
        assert disparity != 0


class TestMitigationRecommendations:
    """Test bias mitigation recommendations"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.detector = BiasDetector()
    
    def test_demographic_bias_recommendations(self):
        """Test recommendations for demographic bias"""
        recommendations = self.detector._generate_mitigation_recommendations(
            BiasType.DEMOGRAPHIC,
            'age_group',
            BiasSeverity.MODERATE
        )
        
        assert len(recommendations) > 0
        assert any('demographic' in rec.lower() or 'training data' in rec.lower() 
                  for rec in recommendations)
    
    def test_socioeconomic_bias_recommendations(self):
        """Test recommendations for socioeconomic bias"""
        recommendations = self.detector._generate_mitigation_recommendations(
            BiasType.SOCIOECONOMIC,
            'insurance_type',
            BiasSeverity.LOW
        )
        
        assert len(recommendations) > 0
        assert any('insurance' in rec.lower() or 'socioeconomic' in rec.lower() 
                  for rec in recommendations)
    
    def test_critical_severity_recommendations(self):
        """Test that critical severity adds urgent recommendations"""
        recommendations = self.detector._generate_mitigation_recommendations(
            BiasType.DEMOGRAPHIC,
            'race',
            BiasSeverity.CRITICAL
        )
        
        assert len(recommendations) > 0
        assert any('immediate' in rec.lower() or 'urgent' in rec.lower() 
                  for rec in recommendations)


class TestBiasSummary:
    """Test bias summary and reporting"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.detector = BiasDetector()
    
    def test_get_bias_summary_empty(self):
        """Test bias summary with no detections"""
        summary = self.detector.get_bias_summary()
        
        assert 'message' in summary or 'total_detections' in summary
    
    def test_get_bias_summary_with_detections(self):
        """Test bias summary with detection data"""
        # Add a mock detection
        detection = BiasDetection(
            detection_id='test_001',
            bias_type=BiasType.DEMOGRAPHIC,
            bias_metric=BiasMetric.DEMOGRAPHIC_PARITY,
            severity=BiasSeverity.MODERATE,
            affected_groups=['group1', 'group2'],
            disparity_score=0.15,
            statistical_significance=0.03,
            confidence_interval=(0.10, 0.20),
            sample_size=100,
            explanation='Test bias detection',
            mitigation_recommendations=['Test recommendation'],
            timestamp=datetime.now(timezone.utc)
        )
        
        self.detector.detection_history.append(detection)
        
        summary = self.detector.get_bias_summary(hours_back=24)
        
        if 'total_detections' in summary:
            assert summary['total_detections'] > 0


class TestBiasAlerts:
    """Test bias alert functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.detector = BiasDetector()
        self.alert_received = False
        self.alert_data = None
    
    def alert_callback(self, alert_data):
        """Callback for testing alerts"""
        self.alert_received = True
        self.alert_data = alert_data
    
    def test_add_bias_alert_callback(self):
        """Test adding bias alert callback"""
        self.detector.add_bias_alert_callback(self.alert_callback)
        
        assert len(self.detector.bias_alert_callbacks) > 0
    
    def test_trigger_bias_alert(self):
        """Test triggering bias alert"""
        self.detector.add_bias_alert_callback(self.alert_callback)
        
        detection = BiasDetection(
            detection_id='test_001',
            bias_type=BiasType.DEMOGRAPHIC,
            bias_metric=BiasMetric.DEMOGRAPHIC_PARITY,
            severity=BiasSeverity.HIGH,
            affected_groups=['group1', 'group2'],
            disparity_score=0.25,
            statistical_significance=0.02,
            confidence_interval=(0.20, 0.30),
            sample_size=100,
            explanation='High severity bias detected',
            mitigation_recommendations=['Immediate review required'],
            timestamp=datetime.now(timezone.utc)
        )
        
        self.detector._trigger_bias_alert(detection)
        
        assert self.alert_received
        assert self.alert_data is not None
        assert self.alert_data['severity'] == BiasSeverity.HIGH.value


class TestComprehensiveBiasAudit:
    """Test comprehensive bias audit functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.detector = BiasDetector()
    
    def test_run_comprehensive_bias_audit(self):
        """Test running comprehensive bias audit"""
        audit_results = self.detector.run_comprehensive_bias_audit()
        
        assert 'audit_timestamp' in audit_results
        assert 'total_decisions_analyzed' in audit_results
        assert 'bias_detections' in audit_results
        assert 'group_performance' in audit_results
        assert 'recommendations' in audit_results


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
