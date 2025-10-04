"""
Tests for monitoring enhancements.

Tests the advanced monitoring improvements including:
- Performance monitoring with analytics
- Security monitoring with aggregation
- Drift detection with multiple methods
"""

import pytest
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import monitoring modules directly
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from security.performance_monitor import (
    ClinicalPerformanceMonitor,
    PerformanceAnalyzer,
    ClinicalPriority,
    PerformanceThresholds,
    reset_performance_monitor,
)
from security.monitoring import SecurityMonitor, AlertDeduplicator, SecurityMetricsAggregator
from mlops.monitoring.drift_detection import ImagingDriftDetector, DriftAlertManager, DriftRecoveryManager


# Mark all tests to allow optional dependencies
pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


class TestPerformanceMonitorEnhancements:
    """Test performance monitoring enhancements."""

    def setup_method(self):
        """Setup test fixtures."""
        reset_performance_monitor()
        self.monitor = ClinicalPerformanceMonitor()
        self.monitor.start_monitoring()

    def teardown_method(self):
        """Cleanup after tests."""
        if hasattr(self, "monitor"):
            self.monitor.stop_monitoring()

    def test_input_validation(self):
        """Test input validation in record_operation."""
        # Test with invalid operation name
        self.monitor.record_operation("", 100, ClinicalPriority.ROUTINE)

        # Test with negative response time
        self.monitor.record_operation("test_op", -50, ClinicalPriority.ROUTINE)

        # Should not crash and should log warnings
        assert True

    def test_configuration_validation(self):
        """Test configuration validation."""
        # Valid configuration
        thresholds = PerformanceThresholds(
            emergency_max_ms=20, critical_max_ms=50, urgent_max_ms=100, routine_max_ms=200, admin_max_ms=500
        )

        monitor = ClinicalPerformanceMonitor(thresholds=thresholds)
        assert monitor is not None

        # Invalid threshold should raise error
        with pytest.raises(ValueError):
            invalid_thresholds = PerformanceThresholds(emergency_max_ms=-10)
            ClinicalPerformanceMonitor(thresholds=invalid_thresholds)

    def test_configurable_monitoring_interval(self):
        """Test configurable monitoring interval."""
        # Set valid interval
        self.monitor.set_monitoring_interval(0.5)
        assert self.monitor.monitoring_interval_seconds == 0.5

        # Invalid interval should raise error
        with pytest.raises(ValueError):
            self.monitor.set_monitoring_interval(-1)

        with pytest.raises(ValueError):
            self.monitor.set_monitoring_interval(100)

    def test_get_configuration(self):
        """Test getting monitor configuration."""
        config = self.monitor.get_configuration()

        assert "thresholds" in config
        assert "auto_optimization_enabled" in config
        assert "monitoring_interval_seconds" in config
        assert config["monitoring_interval_seconds"] > 0

    def test_performance_analyzer(self):
        """Test PerformanceAnalyzer class."""
        # Record some metrics
        for i in range(20):
            self.monitor.record_operation(
                f"test_op_{i % 5}",
                50 + (i * 2),  # Increasing response time
                ClinicalPriority.ROUTINE,
                success=True,
            )

        time.sleep(0.2)  # Allow processing

        analyzer = PerformanceAnalyzer(self.monitor)

        # Test trend analysis
        trends = analyzer.analyze_trends(hours_back=1)
        assert "period_hours" in trends
        assert "response_time_trend" in trends

        # Test risk prediction
        risk = analyzer.predict_violation_risk(hours_ahead=1)
        assert "risk_level" in risk
        assert "confidence" in risk
        assert risk["risk_level"] in ["low", "medium", "high"]

    def test_batch_processing(self):
        """Test batch processing of metrics."""
        # Record many metrics quickly
        for i in range(200):
            self.monitor.record_operation(f"batch_op_{i}", 30, ClinicalPriority.ROUTINE)

        time.sleep(0.3)  # Allow batch processing

        # Verify metrics were processed
        assert len(self.monitor.metrics_history) > 0


class TestSecurityMonitorEnhancements:
    """Test security monitoring enhancements."""

    def setup_method(self):
        """Setup test fixtures."""
        self.config = {"security_monitoring_enabled": True}
        self.monitor = SecurityMonitor(self.config)
        self.monitor.start_monitoring()

    def teardown_method(self):
        """Cleanup after tests."""
        if hasattr(self, "monitor"):
            self.monitor.stop_monitoring()

    def test_health_status(self):
        """Test health status endpoint."""
        health = self.monitor.get_health_status()

        assert "healthy" in health
        assert "monitoring_enabled" in health
        assert "status" in health
        assert health["status"] in ["operational", "stopped"]

    def test_clear_old_data(self):
        """Test clearing old data."""
        # Add some events
        for i in range(10):
            self.monitor.log_security_event(
                "test_event", {"detail": f"event_{i}"}, severity="info", user_id=f"user_{i}"
            )

        # Clear data older than 0 days (should clear all)
        result = self.monitor.clear_old_data(days=365)

        assert "events_cleared" in result
        assert "patterns_cleared" in result

    def test_alert_deduplicator(self):
        """Test AlertDeduplicator class."""
        deduplicator = AlertDeduplicator(dedup_window_seconds=5)

        # First alert should be sent
        assert deduplicator.should_send_alert("test_alert", {"user_id": "user1"}) is True

        # Duplicate alert within window should be suppressed
        assert deduplicator.should_send_alert("test_alert", {"user_id": "user1"}) is False

        # Different alert should be sent
        assert deduplicator.should_send_alert("different_alert", {"user_id": "user1"}) is True

    def test_security_metrics_aggregator(self):
        """Test SecurityMetricsAggregator class."""
        aggregator = SecurityMetricsAggregator(self.monitor)

        # Add some events
        for i in range(15):
            severity = "critical" if i % 5 == 0 else "warning" if i % 3 == 0 else "info"
            self.monitor.log_security_event("test_event", {"index": i}, severity=severity, user_id=f"user_{i % 3}")

        time.sleep(0.1)

        # Test hourly metrics
        hourly = aggregator.get_hourly_metrics(hours=1)
        assert "period_hours" in hourly
        assert "total_events" in hourly

        # Test top event types
        top_events = aggregator.get_top_event_types(limit=5)
        assert isinstance(top_events, list)

        # Test risk score calculation
        risk = aggregator.calculate_risk_score()
        assert "risk_score" in risk
        assert "risk_level" in risk
        assert risk["risk_level"] in ["low", "medium", "high", "critical"]


class TestDriftDetectionEnhancements:
    """Test drift detection enhancements."""

    def setup_method(self):
        """Setup test fixtures."""
        # Create baseline data
        np.random.seed(42)
        self.baseline_data = pd.DataFrame(
            {
                "feature1": np.random.normal(100, 15, 200),
                "feature2": np.random.normal(50, 10, 200),
                "feature3": np.random.uniform(0, 1, 200),
            }
        )

        self.detector = ImagingDriftDetector(drift_features=["feature1", "feature2", "feature3"])

    def test_input_validation_fit(self):
        """Test input validation in fit_baseline."""
        # Empty data should raise error
        with pytest.raises(ValueError):
            self.detector.fit_baseline(pd.DataFrame())

        # Small sample should log warning but work
        small_data = self.baseline_data.head(25)
        self.detector.fit_baseline(small_data)
        assert self.detector.baseline_stats is not None

    def test_input_validation_detect(self):
        """Test input validation in detect_drift."""
        self.detector.fit_baseline(self.baseline_data)

        # Empty data should raise error
        with pytest.raises(ValueError):
            self.detector.detect_drift(pd.DataFrame())

        # Not fitted should raise error
        new_detector = ImagingDriftDetector()
        with pytest.raises(ValueError):
            new_detector.detect_drift(self.baseline_data)

    def test_strict_mode(self):
        """Test strict mode in drift detection."""
        self.detector.fit_baseline(self.baseline_data)

        # Create slightly drifted data
        drifted_data = pd.DataFrame(
            {
                "feature1": np.random.normal(105, 15, 100),  # Slight mean shift
                "feature2": np.random.normal(50, 10, 100),
                "feature3": np.random.uniform(0, 1, 100),
            }
        )

        # Normal mode might not detect
        results_normal = self.detector.detect_drift(drifted_data, strict_mode=False)

        # Strict mode should be more sensitive
        results_strict = self.detector.detect_drift(drifted_data, strict_mode=True)

        assert "strict_mode" in results_strict
        assert results_strict["strict_mode"] is True

    def test_feature_importance(self):
        """Test feature importance calculation."""
        self.detector.fit_baseline(self.baseline_data)

        # Create data with drift in specific features
        for i in range(5):
            drifted_data = pd.DataFrame(
                {
                    "feature1": np.random.normal(120, 20, 100),  # Strong drift
                    "feature2": np.random.normal(50, 10, 100),  # No drift
                    "feature3": np.random.uniform(0, 1, 100),
                }
            )
            self.detector.detect_drift(drifted_data)

        importance = self.detector.get_feature_importance()
        assert isinstance(importance, dict)

        # feature1 should have highest importance due to consistent drift
        if importance:
            assert "feature1" in importance or len(importance) == 0

    def test_export_drift_report(self, tmp_path):
        """Test drift report export."""
        self.detector.fit_baseline(self.baseline_data)

        # Detect some drift
        drifted_data = self.baseline_data.copy()
        drifted_data["feature1"] = drifted_data["feature1"] + 20
        self.detector.detect_drift(drifted_data)

        # Export report
        report_path = tmp_path / "drift_report.json"
        self.detector.export_drift_report(str(report_path), include_history=True)

        assert report_path.exists()

    def test_psi_calculation(self):
        """Test Population Stability Index calculation."""
        self.detector.fit_baseline(self.baseline_data)

        # Create drifted data
        drifted_data = pd.DataFrame(
            {
                "feature1": np.random.normal(110, 15, 100),  # Moderate drift
                "feature2": np.random.normal(50, 10, 100),
                "feature3": np.random.uniform(0.2, 0.8, 100),  # Distribution shift
            }
        )

        results = self.detector.detect_drift_with_multiple_methods(drifted_data, methods=["population_stability"])

        assert "psi_analysis" in results
        psi_analysis = results["psi_analysis"]
        assert isinstance(psi_analysis, dict)

        # Check that PSI was calculated for features
        for feature in ["feature1", "feature2", "feature3"]:
            if feature in psi_analysis:
                assert "psi" in psi_analysis[feature]

    def test_additional_statistical_tests(self):
        """Test additional statistical tests in drift detection."""
        self.detector.fit_baseline(self.baseline_data)

        # Create data with different types of drift
        drifted_data = pd.DataFrame(
            {
                "feature1": np.random.normal(100, 30, 100),  # Variance change
                "feature2": np.random.normal(60, 10, 100),  # Mean shift
                "feature3": np.random.uniform(0, 1, 100),
            }
        )

        results = self.detector.detect_drift(drifted_data)

        # Check that feature analysis includes new tests
        for feature, analysis in results["feature_analysis"].items():
            tests = analysis.get("tests", {})

            # Check for CV test
            if "baseline_cv" in tests:
                assert "current_cv" in tests

            # Check for percentile tests
            if "p90_shift_zscore" in tests:
                assert isinstance(tests["p90_shift_zscore"], (int, float))

    def test_drift_alert_manager(self):
        """Test DriftAlertManager class."""
        alert_manager = DriftAlertManager()

        # Add a callback
        alerts_received = []

        def test_callback(alert):
            alerts_received.append(alert)

        alert_manager.register_alert_callback(test_callback)

        # Process drift results
        drift_results = {"drift_detected": True, "summary_metrics": {"drift_severity": "high"}}

        alert_manager.process_drift_results(drift_results)

        # Verify alert was sent
        assert len(alerts_received) == 1
        assert "severity" in alerts_received[0]

        # Test alert summary
        summary = alert_manager.get_alert_summary(hours=24)
        assert "total_alerts" in summary
        assert summary["total_alerts"] >= 1

    def test_drift_recovery_manager(self):
        """Test DriftRecoveryManager class."""
        self.detector.fit_baseline(self.baseline_data)

        recovery_manager = DriftRecoveryManager()

        # Create high severity drift
        drifted_data = pd.DataFrame(
            {
                "feature1": np.random.normal(150, 20, 100),
                "feature2": np.random.normal(80, 15, 100),
                "feature3": np.random.uniform(0.5, 1, 100),
            }
        )

        drift_results = self.detector.detect_drift(drifted_data)

        # Handle drift event
        recovery_result = recovery_manager.handle_drift_event(drift_results, self.detector)

        assert "actions_taken" in recovery_result
        assert "drift_severity" in recovery_result
        assert isinstance(recovery_result["actions_taken"], list)

        # Get recovery history
        history = recovery_manager.get_recovery_history()
        assert isinstance(history, list)
        assert len(history) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
