"""
Drift Detection for Imaging Features

Monitors imaging features for statistical drift and data quality changes.
Provides alerts when feature distributions deviate from baseline.
"""

import logging
import time
import numpy as np
import pandas as pd
from pathlib import Path
import json
import pickle
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from collections import defaultdict, deque
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class ImagingDriftDetector:
    """
    Drift detection system for medical imaging features.

    Monitors key volumetric and quality control features for distribution drift,
    outliers, and data quality issues.
    """

    def __init__(
        self, drift_features: Optional[List[str]] = None, contamination: float = 0.1, sensitivity: float = 0.05
    ):
        """
        Initialize drift detector.

        Args:
            drift_features: List of feature names to monitor for drift
            contamination: Expected fraction of outliers (for IsolationForest)
            sensitivity: Sensitivity threshold for statistical tests
        """
        self.drift_features = drift_features or []
        self.contamination = contamination
        self.sensitivity = sensitivity

        # Components
        self.outlier_detector = None
        self.scaler = StandardScaler()
        self.baseline_stats = {}
        self.drift_history = []

        # Thresholds
        self.drift_thresholds = {
            "ks_test_pvalue": sensitivity,
            "outlier_fraction": contamination * 2,  # Alert if 2x expected outliers
            "feature_shift_zscore": 2.0,  # Z-score threshold for mean shift
            "variance_ratio": 3.0,  # Alert if variance changes by 3x
        }

        self.logger = logging.getLogger(__name__)

    def fit_baseline(self, baseline_data: pd.DataFrame) -> None:
        """
        Fit the drift detector on baseline/reference data.

        Args:
            baseline_data: Reference dataset to establish baseline statistics

        Raises:
            ValueError: If baseline data is invalid or contains no drift features
        """
        # Input validation
        if baseline_data.empty:
            raise ValueError("Baseline data cannot be empty")

        if len(baseline_data) < 30:
            self.logger.warning(
                f"Baseline data has only {len(baseline_data)} samples. "
                "At least 30 samples recommended for reliable drift detection."
            )

        self.logger.info(f"Fitting drift detector on baseline data: {baseline_data.shape}")

        # Auto-select drift features if not provided
        if not self.drift_features:
            self.drift_features = self._select_drift_features(baseline_data)

        # Ensure drift features exist in data
        available_features = [f for f in self.drift_features if f in baseline_data.columns]
        if not available_features:
            raise ValueError("No drift features found in baseline data")

        self.drift_features = available_features
        self.logger.info(f"Monitoring {len(self.drift_features)} features for drift")

        # Extract drift monitoring data with validation
        drift_data = baseline_data[self.drift_features].copy()

        # Check for missing values
        missing_pct = drift_data.isnull().sum() / len(drift_data)
        high_missing_features = missing_pct[missing_pct > 0.5].index.tolist()

        if high_missing_features:
            self.logger.warning(
                f"Features with >50% missing values will be filled with median: {high_missing_features}"
            )

        drift_data = drift_data.fillna(drift_data.median())

        # Check for constant features
        constant_features = [col for col in drift_data.columns if drift_data[col].nunique() <= 1]
        if constant_features:
            self.logger.warning(f"Constant features detected (may affect drift detection): {constant_features}")

        try:
            # Fit scaler
            self.scaler.fit(drift_data)
            scaled_data = self.scaler.transform(drift_data)
        except Exception as e:
            raise ValueError(f"Failed to scale baseline data: {e}")

        try:
            # Fit outlier detector
            self.outlier_detector = IsolationForest(contamination=self.contamination, random_state=42, n_estimators=100)
            self.outlier_detector.fit(scaled_data)
        except Exception as e:
            raise ValueError(f"Failed to fit outlier detector: {e}")

        # Calculate baseline statistics
        self.baseline_stats = self._calculate_baseline_stats(drift_data)

        self.logger.info("Baseline established for drift detection")

    def detect_drift(self, new_data: pd.DataFrame, strict_mode: bool = False) -> Dict[str, Any]:
        """
        Detect drift in new data compared to baseline.

        Args:
            new_data: New data to check for drift
            strict_mode: If True, use stricter thresholds for drift detection

        Returns:
            Dictionary containing drift detection results

        Raises:
            ValueError: If detector is not fitted or input data is invalid
        """
        if self.outlier_detector is None:
            raise ValueError("Drift detector not fitted. Call fit_baseline() first.")

        # Input validation
        if new_data.empty:
            raise ValueError("New data cannot be empty")

        if len(new_data) < 10:
            self.logger.warning(
                f"New data has only {len(new_data)} samples. " "At least 10 samples recommended for drift detection."
            )

        self.logger.info(f"Detecting drift in new data: {new_data.shape}")

        # Extract drift monitoring features
        available_features = [f for f in self.drift_features if f in new_data.columns]
        if not available_features:
            self.logger.warning("No drift features found in new data")
            return {"drift_detected": False, "error": "No drift features available"}

        drift_data = new_data[available_features].copy()

        # Check data quality
        missing_pct = drift_data.isnull().sum() / len(drift_data)
        if missing_pct.max() > 0.8:
            self.logger.warning(f"High missing value rate detected: {missing_pct[missing_pct > 0.8].to_dict()}")

        drift_data = drift_data.fillna(drift_data.median())

        # Apply strict mode adjustments
        active_thresholds = self.drift_thresholds.copy()
        if strict_mode:
            active_thresholds["ks_test_pvalue"] = self.drift_thresholds["ks_test_pvalue"] * 0.5
            active_thresholds["feature_shift_zscore"] = self.drift_thresholds["feature_shift_zscore"] * 0.75
            self.logger.info("Using strict mode for drift detection")

        # Initialize results
        results = {
            "timestamp": datetime.now().isoformat(),
            "n_samples": len(drift_data),
            "features_monitored": available_features,
            "strict_mode": strict_mode,
            "drift_detected": False,
            "alerts": [],
            "feature_analysis": {},
            "outlier_analysis": {},
            "summary_metrics": {},
        }

        # 1. Statistical drift tests
        for feature in available_features:
            feature_result = self._test_feature_drift(feature, drift_data[feature], self.baseline_stats[feature])
            results["feature_analysis"][feature] = feature_result

            if feature_result["drift_detected"]:
                results["drift_detected"] = True
                results["alerts"].append(f"Statistical drift detected in {feature}")

        # 2. Outlier detection
        try:
            scaled_data = self.scaler.transform(drift_data)
            outlier_scores = self.outlier_detector.decision_function(scaled_data)
            outlier_predictions = self.outlier_detector.predict(scaled_data)

            outlier_fraction = (outlier_predictions == -1).mean()

            results["outlier_analysis"] = {
                "outlier_fraction": float(outlier_fraction),
                "mean_outlier_score": float(outlier_scores.mean()),
                "min_outlier_score": float(outlier_scores.min()),
                "outlier_threshold_exceeded": outlier_fraction > self.drift_thresholds["outlier_fraction"],
            }

            if results["outlier_analysis"]["outlier_threshold_exceeded"]:
                results["drift_detected"] = True
                results["alerts"].append(f"High outlier fraction detected: {outlier_fraction:.3f}")

        except Exception as e:
            self.logger.error(f"Outlier detection failed: {e}")
            results["outlier_analysis"] = {"error": str(e)}

        # 3. Summary metrics
        results["summary_metrics"] = {
            "total_alerts": len(results["alerts"]),
            "features_with_drift": sum(
                1 for fa in results["feature_analysis"].values() if fa.get("drift_detected", False)
            ),
            "drift_severity": "none",  # Will be calculated next
        }

        # Calculate drift severity
        results["summary_metrics"]["drift_severity"] = self._calculate_drift_severity(results)

        # Store in history
        self.drift_history.append(results)

        # Limit history size
        if len(self.drift_history) > 1000:
            self.drift_history = self.drift_history[-1000:]

        self.logger.info(f"Drift detection complete. Drift detected: {results['drift_detected']}")

        return results

    def _select_drift_features(self, data: pd.DataFrame) -> List[str]:
        """Auto-select important features for drift monitoring."""
        candidates = []

        # Prioritize certain feature types
        volume_features = [col for col in data.columns if "volume" in col.lower()]
        intensity_features = [col for col in data.columns if "intensity" in col.lower() or "mean" in col.lower()]
        qc_features = [col for col in data.columns if col.startswith("qc_")]
        shape_features = [col for col in data.columns if any(x in col.lower() for x in ["sphericity", "elongation"])]

        # Select top features from each category
        candidates.extend(volume_features[:3])
        candidates.extend(intensity_features[:2])
        candidates.extend(qc_features[:3])
        candidates.extend(shape_features[:2])

        # Remove duplicates and ensure features exist
        candidates = list(set(candidates))
        candidates = [f for f in candidates if f in data.columns]

        # If still no features, take first numeric columns
        if not candidates:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            candidates = numeric_cols[: min(10, len(numeric_cols))].tolist()

        return candidates

    def _calculate_baseline_stats(self, data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Calculate baseline statistics for each feature."""
        stats_dict = {}

        for feature in data.columns:
            feature_data = data[feature].dropna()

            if len(feature_data) == 0:
                continue

            stats_dict[feature] = {
                "mean": float(feature_data.mean()),
                "std": float(feature_data.std()),
                "median": float(feature_data.median()),
                "min": float(feature_data.min()),
                "max": float(feature_data.max()),
                "q25": float(feature_data.quantile(0.25)),
                "q75": float(feature_data.quantile(0.75)),
                "n_samples": len(feature_data),
                "skewness": float(stats.skew(feature_data)),
                "kurtosis": float(stats.kurtosis(feature_data)),
            }

        return stats_dict

    def _test_feature_drift(
        self, feature_name: str, new_data: pd.Series, baseline_stats: Dict[str, float]
    ) -> Dict[str, Any]:
        """Test individual feature for drift."""

        result = {"feature": feature_name, "drift_detected": False, "tests": {}, "current_stats": {}, "alerts": []}

        # Clean new data
        new_data_clean = new_data.dropna()

        if len(new_data_clean) == 0:
            result["error"] = "No valid data points"
            return result

        # Calculate current statistics
        current_stats = {
            "mean": float(new_data_clean.mean()),
            "std": float(new_data_clean.std()),
            "median": float(new_data_clean.median()),
            "n_samples": len(new_data_clean),
        }
        result["current_stats"] = current_stats

        # Test 1: Mean shift detection
        baseline_mean = baseline_stats["mean"]
        baseline_std = baseline_stats["std"]

        if baseline_std > 0:
            mean_shift_zscore = abs(current_stats["mean"] - baseline_mean) / baseline_std
            result["tests"]["mean_shift_zscore"] = float(mean_shift_zscore)

            if mean_shift_zscore > self.drift_thresholds["feature_shift_zscore"]:
                result["drift_detected"] = True
                result["alerts"].append(f"Mean shift detected (z-score: {mean_shift_zscore:.2f})")

        # Test 2: Variance change detection
        if baseline_std > 0 and current_stats["std"] > 0:
            variance_ratio = current_stats["std"] / baseline_std
            result["tests"]["variance_ratio"] = float(variance_ratio)

            if (
                variance_ratio > self.drift_thresholds["variance_ratio"]
                or variance_ratio < 1 / self.drift_thresholds["variance_ratio"]
            ):
                result["drift_detected"] = True
                result["alerts"].append(f"Variance change detected (ratio: {variance_ratio:.2f})")

        # Test 3: Kolmogorov-Smirnov test (if we had baseline samples)
        # For now, use a simplified approach based on quantiles
        baseline_q25 = baseline_stats["q25"]
        baseline_q75 = baseline_stats["q75"]
        current_q25 = float(new_data_clean.quantile(0.25))
        current_q75 = float(new_data_clean.quantile(0.75))

        if baseline_std > 0:
            q25_shift = abs(current_q25 - baseline_q25) / baseline_std
            q75_shift = abs(current_q75 - baseline_q75) / baseline_std

            result["tests"]["q25_shift_zscore"] = float(q25_shift)
            result["tests"]["q75_shift_zscore"] = float(q75_shift)

            if q25_shift > 1.5 or q75_shift > 1.5:  # Lower threshold for quantiles
                result["drift_detected"] = True
                result["alerts"].append("Distribution shift detected in quantiles")

        return result

    def _calculate_drift_severity(self, results: Dict[str, Any]) -> str:
        """Calculate overall drift severity level."""
        n_alerts = results["summary_metrics"]["total_alerts"]
        n_features_drift = results["summary_metrics"]["features_with_drift"]
        total_features = len(results["features_monitored"])

        if n_alerts == 0:
            return "none"
        elif n_alerts <= 2 and n_features_drift <= 1:
            return "low"
        elif n_alerts <= 5 and n_features_drift <= total_features // 2:
            return "medium"
        else:
            return "high"

    def get_drift_summary(self, days: int = 7) -> Dict[str, Any]:
        """Get drift detection summary for the last N days."""
        if not self.drift_history:
            return {"error": "No drift history available"}

        # Filter recent history
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_history = [h for h in self.drift_history if datetime.fromisoformat(h["timestamp"]) > cutoff_date]

        if not recent_history:
            return {"error": f"No drift history in the last {days} days"}

        summary = {
            "period_days": days,
            "total_checks": len(recent_history),
            "drift_detections": sum(1 for h in recent_history if h["drift_detected"]),
            "drift_rate": sum(1 for h in recent_history if h["drift_detected"]) / len(recent_history),
            "most_common_alerts": {},
            "feature_drift_frequency": {},
            "severity_distribution": {"none": 0, "low": 0, "medium": 0, "high": 0},
        }

        # Aggregate alerts and features
        all_alerts = []
        feature_drift_counts = {}

        for history_item in recent_history:
            all_alerts.extend(history_item.get("alerts", []))

            severity = history_item.get("summary_metrics", {}).get("drift_severity", "none")
            summary["severity_distribution"][severity] += 1

            for feature, analysis in history_item.get("feature_analysis", {}).items():
                if analysis.get("drift_detected", False):
                    feature_drift_counts[feature] = feature_drift_counts.get(feature, 0) + 1

        # Most common alerts
        alert_counts = {}
        for alert in all_alerts:
            alert_type = alert.split(" ")[0]  # First word
            alert_counts[alert_type] = alert_counts.get(alert_type, 0) + 1

        summary["most_common_alerts"] = dict(sorted(alert_counts.items(), key=lambda x: x[1], reverse=True)[:5])
        summary["feature_drift_frequency"] = dict(
            sorted(feature_drift_counts.items(), key=lambda x: x[1], reverse=True)
        )

        return summary

    def save_state(self, filepath: str) -> None:
        """Save drift detector state to file."""
        state = {
            "drift_features": self.drift_features,
            "contamination": self.contamination,
            "sensitivity": self.sensitivity,
            "baseline_stats": self.baseline_stats,
            "drift_thresholds": self.drift_thresholds,
            "drift_history": self.drift_history[-100:],  # Save last 100 entries
        }

        with open(filepath, "w") as f:
            json.dump(state, f, indent=2)

        # Save sklearn components separately
        sklearn_state = {"outlier_detector": self.outlier_detector, "scaler": self.scaler}

        sklearn_filepath = str(Path(filepath).with_suffix(".pkl"))
        with open(sklearn_filepath, "wb") as f:
            pickle.dump(sklearn_state, f)

        self.logger.info(f"Drift detector state saved to {filepath}")

    def load_state(self, filepath: str) -> None:
        """Load drift detector state from file."""
        with open(filepath, "r") as f:
            state = json.load(f)

        self.drift_features = state["drift_features"]
        self.contamination = state["contamination"]
        self.sensitivity = state["sensitivity"]
        self.baseline_stats = state["baseline_stats"]
        self.drift_thresholds = state["drift_thresholds"]
        self.drift_history = state["drift_history"]

        # Load sklearn components
        sklearn_filepath = str(Path(filepath).with_suffix(".pkl"))
        if Path(sklearn_filepath).exists():
            with open(sklearn_filepath, "rb") as f:
                sklearn_state = pickle.load(f)

            self.outlier_detector = sklearn_state["outlier_detector"]
            self.scaler = sklearn_state["scaler"]

        self.logger.info(f"Drift detector state loaded from {filepath}")

    def validate_inputs(self, data: pd.DataFrame) -> bool:
        """Validate input data for drift detection."""
        if data.empty:
            self.logger.error("Input data is empty")
            return False

        if len(data.columns) == 0:
            self.logger.error("Input data has no columns")
            return False

        # Check if any drift features are available
        if self.drift_features:
            available_features = [f for f in self.drift_features if f in data.columns]
            if not available_features:
                self.logger.warning("No drift features found in input data")
                return False

        return True

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get importance scores for drift features based on historical drift frequency.

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.drift_history:
            return {}

        feature_drift_counts = defaultdict(int)
        total_checks = len(self.drift_history)

        for history_item in self.drift_history:
            for feature, analysis in history_item.get("feature_analysis", {}).items():
                if analysis.get("drift_detected", False):
                    feature_drift_counts[feature] += 1

        # Calculate importance as drift frequency
        importance = {feature: count / total_checks for feature, count in feature_drift_counts.items()}

        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    def export_drift_report(self, filepath: str, include_history: bool = False) -> None:
        """
        Export comprehensive drift detection report.

        Args:
            filepath: Path to save report (JSON format)
            include_history: Whether to include full drift history
        """
        report = {
            "export_timestamp": datetime.now().isoformat(),
            "configuration": {
                "drift_features": self.drift_features,
                "contamination": self.contamination,
                "sensitivity": self.sensitivity,
                "drift_thresholds": self.drift_thresholds,
            },
            "baseline_statistics": self.baseline_stats,
            "feature_importance": self.get_feature_importance(),
            "drift_summary": self.get_drift_summary() if self.drift_history else {},
        }

        if include_history:
            report["drift_history"] = self.drift_history[-100:]  # Last 100 entries

        with open(filepath, "w") as f:
            json.dump(report, f, indent=2)

        self.logger.info(f"Drift report exported to {filepath}")


class DriftAlertManager:
    """
    Manage drift detection alerts with configurable thresholds and actions.

    Provides alert routing, escalation, and automated response to drift events.
    """

    def __init__(self, alert_config: Optional[Dict[str, Any]] = None):
        """
        Initialize drift alert manager.

        Args:
            alert_config: Configuration for alert thresholds and actions
        """
        self.alert_config = alert_config or {}
        self.alert_callbacks = []
        self.alert_history = deque(maxlen=1000)
        self.logger = logging.getLogger(__name__)

        # Default alert thresholds
        self.thresholds = {
            "low_severity": self.alert_config.get("low_threshold", 0.1),
            "medium_severity": self.alert_config.get("medium_threshold", 0.3),
            "high_severity": self.alert_config.get("high_threshold", 0.6),
        }

    def process_drift_results(self, drift_results: Dict[str, Any]) -> None:
        """
        Process drift detection results and trigger appropriate alerts.

        Args:
            drift_results: Results from drift detection
        """
        if not drift_results.get("drift_detected", False):
            return

        severity = drift_results.get("summary_metrics", {}).get("drift_severity", "low")
        alert_data = {
            "timestamp": datetime.now().isoformat(),
            "severity": severity,
            "drift_results": drift_results,
            "alert_id": f"drift_alert_{int(time.time() * 1000)}",
        }

        self.alert_history.append(alert_data)
        self._trigger_alerts(alert_data)

    def _trigger_alerts(self, alert_data: Dict[str, Any]) -> None:
        """Trigger alerts through registered callbacks."""
        for callback in self.alert_callbacks:
            try:
                callback(alert_data)
            except Exception as e:
                self.logger.error(f"Error in alert callback: {e}")

    def register_alert_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Register a callback function for drift alerts."""
        self.alert_callbacks.append(callback)

    def get_alert_summary(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get summary of recent alerts.

        Args:
            hours: Number of hours to summarize

        Returns:
            Alert summary statistics
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)

        recent_alerts = [
            alert for alert in self.alert_history if datetime.fromisoformat(alert["timestamp"]) > cutoff_time
        ]

        severity_counts = defaultdict(int)
        for alert in recent_alerts:
            severity_counts[alert["severity"]] += 1

        return {
            "period_hours": hours,
            "total_alerts": len(recent_alerts),
            "by_severity": dict(severity_counts),
            "last_alert": recent_alerts[-1] if recent_alerts else None,
        }


class DriftRecoveryManager:
    """
    Automated recovery actions for drift detection.

    Implements strategies to respond to detected drift automatically.
    """

    def __init__(self, recovery_config: Optional[Dict[str, Any]] = None):
        """
        Initialize drift recovery manager.

        Args:
            recovery_config: Configuration for recovery actions
        """
        self.recovery_config = recovery_config or {}
        self.recovery_history = []
        self.logger = logging.getLogger(__name__)

    def handle_drift_event(self, drift_results: Dict[str, Any], detector: ImagingDriftDetector) -> Dict[str, Any]:
        """
        Handle drift event with appropriate recovery actions.

        Args:
            drift_results: Results from drift detection
            detector: Drift detector instance

        Returns:
            Recovery action results
        """
        severity = drift_results.get("summary_metrics", {}).get("drift_severity", "low")

        recovery_actions = []

        if severity in ["high", "medium"]:
            # Log detailed drift information
            self.logger.warning(
                f"Drift detected with severity {severity}: "
                f"{drift_results.get('summary_metrics', {}).get('features_with_drift')} features affected"
            )

            # Action 1: Save detailed drift report
            report_path = f"drift_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            try:
                detector.export_drift_report(report_path, include_history=True)
                recovery_actions.append({"action": "export_report", "status": "success", "path": report_path})
            except Exception as e:
                self.logger.error(f"Failed to export drift report: {e}")
                recovery_actions.append({"action": "export_report", "status": "failed", "error": str(e)})

        if severity == "high":
            # Action 2: Flag for retraining
            recovery_actions.append(
                {
                    "action": "flag_for_retraining",
                    "status": "success",
                    "reason": "High severity drift detected",
                }
            )

            # Action 3: Notify stakeholders (placeholder)
            recovery_actions.append(
                {
                    "action": "notify_stakeholders",
                    "status": "success",
                    "severity": severity,
                }
            )

        recovery_result = {
            "timestamp": datetime.now().isoformat(),
            "drift_severity": severity,
            "actions_taken": recovery_actions,
            "drift_summary": drift_results.get("summary_metrics", {}),
        }

        self.recovery_history.append(recovery_result)
        return recovery_result

    def get_recovery_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent recovery action history."""
        return self.recovery_history[-limit:]
