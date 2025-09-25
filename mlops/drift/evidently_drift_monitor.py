#!/usr/bin/env python3
"""
Data and Model Drift Monitoring using Evidently AI for DuetMind Adaptive MLOps.
Comprehensive drift detection with statistical tests and reporting.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import yaml
import json
import os
from scipy import stats

# Import drift detection constants
try:
    from constants import (
        DRIFT_DETECTION_KS_THRESHOLD, DRIFT_DETECTION_WASSERSTEIN_THRESHOLD,
        DRIFT_DETECTION_TV_THRESHOLD, DRIFT_DETECTION_JS_THRESHOLD, 
        DRIFT_DETECTION_CHI2_THRESHOLD
    )
except ImportError:
    # Fallback constants if import fails
    DRIFT_DETECTION_KS_THRESHOLD = 0.05
    DRIFT_DETECTION_WASSERSTEIN_THRESHOLD = 0.1
    DRIFT_DETECTION_TV_THRESHOLD = 0.2
    DRIFT_DETECTION_JS_THRESHOLD = 0.1
    DRIFT_DETECTION_CHI2_THRESHOLD = 0.05

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Evidently imports - with fallback handling
try:
    from evidently import ColumnMapping
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, DataQualityPreset
    from evidently.metrics import DatasetDriftMetric, ColumnDriftMetric
    from evidently.test_suite import TestSuite
    from evidently.tests import TestNumberOfColumnsWithMissingValues, TestNumberOfRowsWithMissingValues
    from evidently.tests import TestColumnDrift, TestDatasetDrift
    EVIDENTLY_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Evidently not available: {e}. Falling back to statistical methods.")
    EVIDENTLY_AVAILABLE = False
    
    # Mock classes for compatibility
    class ColumnMapping:
        def __init__(self, **kwargs):
            pass
    
    class Report:
        def __init__(self, *args, **kwargs):
            pass
        def run(self, *args, **kwargs):
            pass
        def as_dict(self):
            return {'metrics': []}
        def save_html(self, path):
            pass


class DriftMonitor:
    """
    Advanced drift monitoring class using Evidently AI framework.
    Provides comprehensive data and model drift detection capabilities.
    """
    
    def __init__(self, reference_data: pd.DataFrame, drift_threshold: float = 0.1,
                 numerical_features: List[str] = None, categorical_features: List[str] = None):
        """
        Initialize drift monitor.
        
        Args:
            reference_data: Reference dataset for comparison
            drift_threshold: Threshold for drift detection (0-1)
            numerical_features: List of numerical feature names
            categorical_features: List of categorical feature names
        """
        self.reference_data = reference_data
        self.drift_threshold = drift_threshold
        
        # Auto-detect feature types if not provided
        if numerical_features is None:
            self.numerical_features = reference_data.select_dtypes(include=[np.number]).columns.tolist()
        else:
            self.numerical_features = numerical_features
            
        if categorical_features is None:
            self.categorical_features = reference_data.select_dtypes(include=['object', 'category']).columns.tolist()
        else:
            self.categorical_features = categorical_features
        
        # Create column mapping for Evidently
        self.column_mapping = ColumnMapping(
            numerical_features=self.numerical_features,
            categorical_features=self.categorical_features
        )
        
        logger.info(f"Initialized DriftMonitor with {len(self.numerical_features)} numerical and {len(self.categorical_features)} categorical features")
    
    def detect_data_drift(self, current_data: pd.DataFrame, generate_report: bool = True) -> Dict[str, Any]:
        """
        Detect data drift using Evidently framework with fallback.
        
        Args:
            current_data: Current dataset to compare
            generate_report: Whether to generate detailed HTML report
            
        Returns:
            Dictionary with comprehensive drift detection results
        """
        if not EVIDENTLY_AVAILABLE:
            logger.info("Using fallback drift detection (Evidently not available)")
            return self._fallback_drift_detection(current_data)
        
        try:
            # Create Evidently data drift report
            report = Report(metrics=[
                DataDriftPreset(),
                DataQualityPreset(),
                DatasetDriftMetric(),
            ])
            
            report.run(
                reference_data=self.reference_data,
                current_data=current_data,
                column_mapping=self.column_mapping
            )
            
            # Extract results from Evidently report
            result_dict = report.as_dict()
            
            # Parse Evidently results into our format
            drift_results = self._parse_evidently_results(result_dict)
            
            # Generate HTML report if requested
            if generate_report:
                report_path = f"drift_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                report.save_html(report_path)
                drift_results['report_path'] = report_path
                logger.info(f"Drift report saved to {report_path}")
            
            return drift_results
            
        except Exception as e:
            logger.error(f"Error in Evidently drift detection: {e}")
            # Fallback to simple drift detection
            return self._fallback_drift_detection(current_data)
    
    def _parse_evidently_results(self, result_dict: Dict) -> Dict[str, Any]:
        """
        Parse Evidently results into standardized format.
        
        Args:
            result_dict: Raw results from Evidently report
            
        Returns:
            Parsed drift results
        """
        try:
            metrics = result_dict.get('metrics', [])
            
            # Find dataset drift metric
            dataset_drift_metric = None
            feature_drift_metrics = {}
            
            for metric in metrics:
                if metric.get('metric') == 'DatasetDriftMetric':
                    dataset_drift_metric = metric.get('result', {})
                elif metric.get('metric') == 'ColumnDriftMetric':
                    column_name = metric.get('result', {}).get('column_name')
                    if column_name:
                        feature_drift_metrics[column_name] = metric.get('result', {})
            
            # Build standardized results
            drift_results = {
                'timestamp': datetime.now().isoformat(),
                'overall_drift_detected': False,
                'drift_score': 0.0,
                'feature_drift': {},
                'summary': {
                    'total_features': len(self.reference_data.columns),
                    'drifted_features': 0,
                    'drift_threshold': self.drift_threshold
                },
                'evidently_results': {
                    'dataset_drift': dataset_drift_metric,
                    'feature_drift': feature_drift_metrics
                }
            }
            
            if dataset_drift_metric:
                drift_results['overall_drift_detected'] = dataset_drift_metric.get('dataset_drift', False)
                drift_results['drift_score'] = dataset_drift_metric.get('drift_score', 0.0)
                
                # Count drifted features
                drifted_features = dataset_drift_metric.get('drifted_features_count', 0)
                drift_results['summary']['drifted_features'] = drifted_features
            
            # Process individual feature drift results
            for feature_name, feature_result in feature_drift_metrics.items():
                drift_results['feature_drift'][feature_name] = {
                    'drift_detected': feature_result.get('drift_detected', False),
                    'drift_score': feature_result.get('drift_score', 0.0),
                    'method': 'evidently_statistical_test',
                    'p_value': feature_result.get('stattest_name', 'unknown'),
                    'statistics': feature_result
                }
            
            logger.info(f"Parsed Evidently results: {drift_results['summary']['drifted_features']} drifted features")
            return drift_results
            
        except Exception as e:
            logger.error(f"Error parsing Evidently results: {e}")
            return self._create_empty_results()
    
    def _fallback_drift_detection(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Fallback drift detection using simple statistical methods.
        
        Args:
            current_data: Current dataset to compare
            
        Returns:
            Drift detection results using simple methods
        """
        logger.warning("Using fallback drift detection due to Evidently error")
        
        drift_results = {
            'timestamp': datetime.now().isoformat(),
            'overall_drift_detected': False,
            'drift_score': 0.0,
            'feature_drift': {},
            'summary': {
                'total_features': len(current_data.columns),
                'drifted_features': 0,
                'drift_threshold': self.drift_threshold
            },
            'method': 'fallback_statistical'
        }
        
        # Check numerical features
        for col in self.numerical_features:
            if col in current_data.columns:
                drift_result = self._detect_numerical_drift(
                    self.reference_data[col], 
                    current_data[col],
                    col
                )
                drift_results['feature_drift'][col] = drift_result
                
                if drift_result['drift_detected']:
                    drift_results['summary']['drifted_features'] += 1
        
        # Check categorical features
        for col in self.categorical_features:
            if col in current_data.columns:
                drift_result = self._detect_categorical_drift(
                    self.reference_data[col],
                    current_data[col],
                    col
                )
                drift_results['feature_drift'][col] = drift_result
                
                if drift_result['drift_detected']:
                    drift_results['summary']['drifted_features'] += 1
        
        # Calculate overall drift score
        total_features = len(drift_results['feature_drift'])
        if total_features > 0:
            drift_results['drift_score'] = drift_results['summary']['drifted_features'] / total_features
            drift_results['overall_drift_detected'] = drift_results['drift_score'] > self.drift_threshold
        
        return drift_results
    
    def _create_empty_results(self) -> Dict[str, Any]:
        """Create empty results structure for error cases."""
        return {
            'timestamp': datetime.now().isoformat(),
            'overall_drift_detected': False,
            'drift_score': 0.0,
            'feature_drift': {},
            'summary': {
                'total_features': 0,
                'drifted_features': 0,
                'drift_threshold': self.drift_threshold
            },
            'error': 'Failed to process drift detection'
        }
    
    def run_drift_tests(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Run comprehensive drift test suite using Evidently with fallback.
        
        Args:
            current_data: Current dataset to test
            
        Returns:
            Test suite results
        """
        if not EVIDENTLY_AVAILABLE:
            logger.info("Using simplified drift tests (Evidently not available)")
            return self._fallback_drift_tests(current_data)
        
        try:
            # Create test suite
            test_suite = TestSuite(tests=[
                TestDatasetDrift(),
                TestNumberOfColumnsWithMissingValues(),
                TestNumberOfRowsWithMissingValues(),
            ])
            
            # Add individual column drift tests
            for col in self.numerical_features + self.categorical_features:
                if col in current_data.columns:
                    test_suite.tests.append(TestColumnDrift(column_name=col))
            
            test_suite.run(
                reference_data=self.reference_data,
                current_data=current_data,
                column_mapping=self.column_mapping
            )
            
            # Get test results
            test_results = test_suite.as_dict()
            
            return {
                'timestamp': datetime.now().isoformat(),
                'test_results': test_results,
                'passed_tests': sum(1 for test in test_results.get('tests', []) if test.get('status') == 'SUCCESS'),
                'total_tests': len(test_results.get('tests', [])),
                'overall_status': 'PASS' if all(test.get('status') == 'SUCCESS' for test in test_results.get('tests', [])) else 'FAIL'
            }
            
        except Exception as e:
            logger.error(f"Error running drift test suite: {e}")
            return self._fallback_drift_tests(current_data)
    
    def _fallback_drift_tests(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """Fallback drift tests using statistical methods."""
        drift_results = self._fallback_drift_detection(current_data)
        
        # Convert drift results to test format
        tests_passed = sum(1 for result in drift_results['feature_drift'].values() if not result['drift_detected'])
        total_tests = len(drift_results['feature_drift']) + 2  # +2 for data quality checks
        
        return {
            'timestamp': datetime.now().isoformat(),
            'test_results': {
                'drift_summary': drift_results,
                'method': 'statistical_fallback'
            },
            'passed_tests': tests_passed,
            'total_tests': total_tests,
            'overall_status': 'PASS' if not drift_results['overall_drift_detected'] else 'FAIL'
        }
    
    def _detect_numerical_drift(self, reference: pd.Series, current: pd.Series, feature_name: str) -> Dict:
        """
        Detect drift in numerical features using statistical tests.
        
        Args:
            reference: Reference feature values
            current: Current feature values
            feature_name: Name of the feature
            
        Returns:
            Dictionary with drift results for this feature
        """
        # Remove NaN values
        ref_clean = reference.dropna()
        cur_clean = current.dropna()
        
        if len(ref_clean) == 0 or len(cur_clean) == 0:
            return {
                'drift_detected': False,
                'drift_score': 0.0,
                'method': 'insufficient_data',
                'statistics': {}
            }
        
        # Calculate basic statistics
        ref_stats = {
            'mean': ref_clean.mean(),
            'std': ref_clean.std(),
            'min': ref_clean.min(),
            'max': ref_clean.max(),
            'median': ref_clean.median()
        }
        
        cur_stats = {
            'mean': cur_clean.mean(),
            'std': cur_clean.std(),
            'min': cur_clean.min(),
            'max': cur_clean.max(),
            'median': cur_clean.median()
        }
        
        # Simple drift detection using mean shift
        mean_shift = abs(cur_stats['mean'] - ref_stats['mean']) / (ref_stats['std'] + 1e-8)
        mean_shift_drift = mean_shift > 2.0  # 2 standard deviations
        
        # Implement sophisticated statistical tests
        ks_stat, ks_p_value = stats.ks_2samp(ref_clean, cur_clean)
        ks_drift = ks_p_value < DRIFT_DETECTION_KS_THRESHOLD  # Significant if p < threshold
        
        # Wasserstein distance (Earth Mover's Distance)
        wasserstein_distance = stats.wasserstein_distance(ref_clean, cur_clean)
        # Normalize by the range for threshold comparison
        data_range = max(ref_clean.max(), cur_clean.max()) - min(ref_clean.min(), cur_clean.min())
        normalized_wasserstein = wasserstein_distance / (data_range + 1e-8)
        wasserstein_drift = normalized_wasserstein > DRIFT_DETECTION_WASSERSTEIN_THRESHOLD
        
        # Combined drift decision using majority vote
        drift_tests = [mean_shift_drift, ks_drift, wasserstein_drift]
        drift_detected = sum(drift_tests) >= 2  # Majority vote
        
        # Combined drift score as weighted average
        drift_score = (mean_shift * 0.3 + (1 - ks_p_value) * 0.4 + normalized_wasserstein * 0.3)
        
        return {
            'drift_detected': drift_detected,
            'drift_score': drift_score,
            'method': 'combined_statistical',
            'statistics': {
                'reference': ref_stats,
                'current': cur_stats,
                'mean_shift': mean_shift,
                'ks_statistic': ks_stat,
                'ks_p_value': ks_p_value,
                'wasserstein_distance': wasserstein_distance,
                'normalized_wasserstein': normalized_wasserstein,
                'individual_tests': {
                    'mean_shift_drift': mean_shift_drift,
                    'ks_drift': ks_drift,
                    'wasserstein_drift': wasserstein_drift
                }
            }
        }
    
    def _detect_categorical_drift(self, reference: pd.Series, current: pd.Series, feature_name: str) -> Dict:
        """
        Detect drift in categorical features using distribution comparison.
        
        Args:
            reference: Reference feature values
            current: Current feature values
            feature_name: Name of the feature
            
        Returns:
            Dictionary with drift results for this feature
        """
        # Remove NaN values
        ref_clean = reference.dropna()
        cur_clean = current.dropna()
        
        if len(ref_clean) == 0 or len(cur_clean) == 0:
            return {
                'drift_detected': False,
                'drift_score': 0.0,
                'method': 'insufficient_data',
                'statistics': {}
            }
        
        # Calculate distributions
        ref_dist = ref_clean.value_counts(normalize=True).sort_index()
        cur_dist = cur_clean.value_counts(normalize=True).sort_index()
        
        # Align distributions (handle missing categories)
        all_categories = set(ref_dist.index) | set(cur_dist.index)
        ref_aligned = ref_dist.reindex(all_categories, fill_value=0)
        cur_aligned = cur_dist.reindex(all_categories, fill_value=0)
        
        # Calculate Total Variation Distance
        tv_distance = 0.5 * np.sum(np.abs(ref_aligned - cur_aligned))
        tv_drift = tv_distance > DRIFT_DETECTION_TV_THRESHOLD  # Threshold from constants
        
        # Implement chi-square test for categorical drift
        try:
            # Convert distributions to counts for chi-square test
            # Use minimum sample size for proper chi-square test
            min_sample_size = min(len(ref_clean), len(cur_clean))
            ref_counts = (ref_aligned * min_sample_size).round().astype(int)
            cur_counts = (cur_aligned * min_sample_size).round().astype(int)
            
            # Ensure no zero counts (add 1 to avoid division by zero)
            ref_counts = ref_counts + 1
            cur_counts = cur_counts + 1
            
            # Create contingency table
            contingency_table = np.array([ref_counts, cur_counts])
            
            # Perform chi-square test
            chi2_stat, chi2_p_value, dof, expected = stats.chi2_contingency(contingency_table)
            chi2_drift = chi2_p_value < DRIFT_DETECTION_CHI2_THRESHOLD  # Significant if p < threshold
            
        except (ValueError, ZeroDivisionError) as e:
            logger.warning(f"Chi-square test failed for {feature_name}: {e}")
            chi2_stat, chi2_p_value, chi2_drift = 0, 1, False
        
        # Jensen-Shannon divergence as additional test
        try:
            # Calculate JS divergence
            js_div = self._jensen_shannon_divergence(ref_aligned.values, cur_aligned.values)
            js_drift = js_div > DRIFT_DETECTION_JS_THRESHOLD  # Threshold from constants
        except Exception as e:
            logger.warning(f"JS divergence calculation failed for {feature_name}: {e}")
            js_div, js_drift = 0, False
        
        # Combined drift decision using majority vote
        drift_tests = [tv_drift, chi2_drift, js_drift]
        drift_detected = sum(drift_tests) >= 2  # Majority vote
        
        # Combined drift score as weighted average
        drift_score = (tv_distance * 0.4 + (1 - chi2_p_value) * 0.3 + js_div * 0.3)
        
        return {
            'drift_detected': drift_detected,
            'drift_score': drift_score,
            'method': 'combined_categorical',
            'statistics': {
                'reference_distribution': ref_dist.to_dict(),
                'current_distribution': cur_dist.to_dict(),
                'tv_distance': tv_distance,
                'chi2_statistic': chi2_stat,
                'chi2_p_value': chi2_p_value,
                'js_divergence': js_div,
                'individual_tests': {
                    'tv_drift': tv_drift,
                    'chi2_drift': chi2_drift,
                    'js_drift': js_drift
                },
                'new_categories': list(set(cur_dist.index) - set(ref_dist.index)),
                'missing_categories': list(set(ref_dist.index) - set(cur_dist.index))
            }
        }
    
    def _jensen_shannon_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """
        Calculate Jensen-Shannon divergence between two probability distributions.
        
        Args:
            p: First probability distribution
            q: Second probability distribution
            
        Returns:
            JS divergence value (0 to 1, where 0 means identical distributions)
        """
        # Ensure arrays sum to 1
        p = p / (np.sum(p) + 1e-8)
        q = q / (np.sum(q) + 1e-8)
        
        # Calculate average distribution
        m = 0.5 * (p + q)
        
        # Calculate KL divergences with small epsilon to avoid log(0)
        eps = 1e-8
        kl_pm = np.sum(p * np.log((p + eps) / (m + eps)))
        kl_qm = np.sum(q * np.log((q + eps) / (m + eps)))
        
        # JS divergence is the average of the two KL divergences
        js_div = 0.5 * (kl_pm + kl_qm)
        
        # Convert to JS distance (square root of JS divergence)
        return np.sqrt(js_div)


class ModelDriftMonitor:
    """
    Model performance drift monitoring.
    """
    
    def __init__(self, baseline_metrics: Dict):
        """
        Initialize model drift monitor.
        
        Args:
            baseline_metrics: Reference performance metrics
        """
        self.baseline_metrics = baseline_metrics
        logger.info("Initialized ModelDriftMonitor")
    
    def detect_performance_drift(self, current_metrics: Dict, threshold: float = 0.05) -> Dict:
        """
        Detect performance drift in model metrics.
        
        Args:
            current_metrics: Current model performance metrics
            threshold: Threshold for performance degradation (0-1)
            
        Returns:
            Dictionary with performance drift results
        """
        drift_results = {
            'timestamp': datetime.now().isoformat(),
            'performance_drift_detected': False,
            'degraded_metrics': [],
            'metric_comparisons': {}
        }
        
        for metric_name, baseline_value in self.baseline_metrics.items():
            if metric_name in current_metrics:
                current_value = current_metrics[metric_name]
                
                # Calculate relative change (assuming higher is better for most metrics)
                if baseline_value > 0:
                    relative_change = (current_value - baseline_value) / baseline_value
                else:
                    relative_change = 0
                
                # Check for degradation
                degraded = relative_change < -threshold
                
                drift_results['metric_comparisons'][metric_name] = {
                    'baseline': baseline_value,
                    'current': current_value,
                    'relative_change': relative_change,
                    'degraded': degraded
                }
                
                if degraded:
                    drift_results['degraded_metrics'].append(metric_name)
        
        drift_results['performance_drift_detected'] = len(drift_results['degraded_metrics']) > 0
        
        logger.info(f"Performance drift check: {len(drift_results['degraded_metrics'])} degraded metrics")
        
        return drift_results


def load_reference_data(params: dict) -> pd.DataFrame:
    """Load reference dataset for drift monitoring."""
    reference_file = params['data']['features']
    
    if os.path.exists(reference_file):
        return pd.read_parquet(reference_file)
    else:
        logger.warning(f"Reference file not found: {reference_file}")
        return pd.DataFrame()


def simulate_current_data(reference_data: pd.DataFrame, drift_factor: float = 0.1) -> pd.DataFrame:
    """
    Simulate current data with some drift for testing.
    
    Args:
        reference_data: Original reference data
        drift_factor: Amount of drift to introduce (0-1)
        
    Returns:
        Simulated current data with drift
    """
    current_data = reference_data.copy()
    
    # Add drift to numerical columns
    for col in reference_data.select_dtypes(include=[np.number]).columns:
        if col in current_data.columns:
            # Add systematic shift
            shift = drift_factor * current_data[col].std()
            current_data[col] = current_data[col] + np.random.normal(shift, shift * 0.1, len(current_data))
    
    # Add drift to categorical columns
    for col in reference_data.select_dtypes(include=['object', 'category']).columns:
        if col in current_data.columns and len(current_data[col].unique()) > 1:
            # Randomly change some values
            mask = np.random.random(len(current_data)) < drift_factor * 0.1
            if mask.any():
                unique_values = current_data[col].unique()
                current_data.loc[mask, col] = np.random.choice(unique_values, mask.sum())
    
    logger.info(f"Simulated current data with drift factor {drift_factor}")
    return current_data


def main():
    """Demonstrate advanced drift monitoring capabilities using Evidently."""
    logger.info("Starting advanced drift monitoring demonstration with Evidently...")
    
    # Load parameters
    with open("params.yaml", 'r') as f:
        params = yaml.safe_load(f)
    
    # Load reference data
    reference_data = load_reference_data(params)
    
    if reference_data.empty:
        logger.error("No reference data available. Run the ML pipeline first.")
        return
    
    # Get feature lists from params
    numerical_features = params.get('features', {}).get('numerical_features', [])
    categorical_features = params.get('features', {}).get('categorical_features', [])
    
    # Initialize advanced drift monitor
    drift_monitor = DriftMonitor(
        reference_data, 
        drift_threshold=params.get('drift', {}).get('drift_threshold', 0.15),
        numerical_features=numerical_features,
        categorical_features=categorical_features
    )
    
    # Simulate current data with drift
    current_data = simulate_current_data(reference_data, drift_factor=0.2)
    
    # Detect data drift using Evidently
    print("\n=== Advanced Data Drift Detection with Evidently ===")
    data_drift_results = drift_monitor.detect_data_drift(current_data, generate_report=True)
    
    # Print results
    print(f"Overall drift detected: {data_drift_results['overall_drift_detected']}")
    print(f"Drift score: {data_drift_results['drift_score']:.3f}")
    print(f"Drifted features: {data_drift_results['summary']['drifted_features']}")
    
    if 'report_path' in data_drift_results:
        print(f"Detailed HTML report: {data_drift_results['report_path']}")
    
    print("\nDrifted Features:")
    for feature, result in data_drift_results['feature_drift'].items():
        if result.get('drift_detected', False):
            print(f"  - {feature}: {result['method']} = {result['drift_score']:.3f}")
    
    # Run comprehensive drift test suite
    print("\n=== Drift Test Suite Results ===")
    test_results = drift_monitor.run_drift_tests(current_data)
    print(f"Test status: {test_results['overall_status']}")
    print(f"Passed tests: {test_results.get('passed_tests', 0)}/{test_results.get('total_tests', 0)}")
    
    # Demonstrate model drift monitoring
    baseline_metrics = {"accuracy": 0.85, "roc_auc": 0.80, "precision": 0.82, "recall": 0.78}
    current_metrics = {"accuracy": 0.82, "roc_auc": 0.78, "precision": 0.79, "recall": 0.75}  # Simulated degradation
    
    model_drift_monitor = ModelDriftMonitor(baseline_metrics)
    performance_drift_results = model_drift_monitor.detect_performance_drift(current_metrics)
    
    print("\n=== Model Performance Drift Results ===")
    print(f"Performance drift detected: {performance_drift_results['performance_drift_detected']}")
    print(f"Degraded metrics: {performance_drift_results['degraded_metrics']}")
    
    for metric, comparison in performance_drift_results['metric_comparisons'].items():
        print(f"  {metric}: {comparison['baseline']:.3f} → {comparison['current']:.3f} "
              f"({comparison['relative_change']:+.2%})")
    
    # Save comprehensive drift report
    drift_report = {
        'data_drift': data_drift_results,
        'test_suite': test_results,
        'model_drift': performance_drift_results
    }
    
    report_file = f"comprehensive_drift_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w') as f:
        json.dump(drift_report, f, indent=2, default=str)
    
    print(f"\nComprehensive drift report saved to: {report_file}")
    logger.info("Advanced drift monitoring demonstration completed!")


if __name__ == "__main__":
    main()