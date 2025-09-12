#!/usr/bin/env python3
"""
Data and Model Drift Monitoring Scaffold for DuetMind Adaptive MLOps.
Provides framework for Evidently or custom drift detection logic.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import yaml
import json
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DriftMonitor:
    """
    Base drift monitoring class with scaffold implementation.
    Can be extended with Evidently or custom drift detection methods.
    """
    
    def __init__(self, reference_data: pd.DataFrame, drift_threshold: float = 0.1):
        """
        Initialize drift monitor.
        
        Args:
            reference_data: Reference dataset for comparison
            drift_threshold: Threshold for drift detection (0-1)
        """
        self.reference_data = reference_data
        self.drift_threshold = drift_threshold
        self.numerical_columns = reference_data.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = reference_data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        logger.info(f"Initialized DriftMonitor with {len(self.numerical_columns)} numerical and {len(self.categorical_columns)} categorical features")
    
    def detect_data_drift(self, current_data: pd.DataFrame) -> Dict:
        """
        Detect data drift between reference and current data.
        
        Args:
            current_data: Current dataset to compare
            
        Returns:
            Dictionary with drift detection results
        """
        drift_results = {
            'timestamp': datetime.now().isoformat(),
            'overall_drift_detected': False,
            'drift_score': 0.0,
            'feature_drift': {},
            'summary': {
                'total_features': len(current_data.columns),
                'drifted_features': 0,
                'drift_threshold': self.drift_threshold
            }
        }
        
        # Check numerical features
        for col in self.numerical_columns:
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
        for col in self.categorical_columns:
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
        
        logger.info(f"Drift detection completed: {drift_results['summary']['drifted_features']} of {total_features} features drifted")
        
        return drift_results
    
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
        drift_detected = mean_shift > 2.0  # 2 standard deviations
        
        # TODO: Implement more sophisticated tests (KS test, Wasserstein distance, etc.)
        
        return {
            'drift_detected': drift_detected,
            'drift_score': mean_shift,
            'method': 'mean_shift',
            'statistics': {
                'reference': ref_stats,
                'current': cur_stats,
                'mean_shift': mean_shift
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
        drift_detected = tv_distance > 0.2  # 20% threshold
        
        # TODO: Implement chi-square test or other categorical drift tests
        
        return {
            'drift_detected': drift_detected,
            'drift_score': tv_distance,
            'method': 'total_variation',
            'statistics': {
                'reference_distribution': ref_dist.to_dict(),
                'current_distribution': cur_dist.to_dict(),
                'tv_distance': tv_distance,
                'new_categories': list(set(cur_dist.index) - set(ref_dist.index)),
                'missing_categories': list(set(ref_dist.index) - set(cur_dist.index))
            }
        }


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
    """Demonstrate drift monitoring capabilities."""
    logger.info("Starting drift monitoring demonstration...")
    
    # Load parameters
    with open("params.yaml", 'r') as f:
        params = yaml.safe_load(f)
    
    # Load reference data
    reference_data = load_reference_data(params)
    
    if reference_data.empty:
        logger.error("No reference data available. Run the ML pipeline first.")
        return
    
    # Initialize drift monitor
    drift_monitor = DriftMonitor(reference_data, drift_threshold=0.15)
    
    # Simulate current data with drift
    current_data = simulate_current_data(reference_data, drift_factor=0.2)
    
    # Detect data drift
    data_drift_results = drift_monitor.detect_data_drift(current_data)
    
    # Print results
    print("\n=== Data Drift Detection Results ===")
    print(f"Overall drift detected: {data_drift_results['overall_drift_detected']}")
    print(f"Drift score: {data_drift_results['drift_score']:.3f}")
    print(f"Drifted features: {data_drift_results['summary']['drifted_features']}")
    
    print("\nDrifted Features:")
    for feature, result in data_drift_results['feature_drift'].items():
        if result['drift_detected']:
            print(f"  - {feature}: {result['method']} = {result['drift_score']:.3f}")
    
    # Demonstrate model drift monitoring
    baseline_metrics = {"accuracy": 0.85, "roc_auc": 0.80}
    current_metrics = {"accuracy": 0.82, "roc_auc": 0.78}  # Simulated degradation
    
    model_drift_monitor = ModelDriftMonitor(baseline_metrics)
    performance_drift_results = model_drift_monitor.detect_performance_drift(current_metrics)
    
    print("\n=== Model Performance Drift Results ===")
    print(f"Performance drift detected: {performance_drift_results['performance_drift_detected']}")
    print(f"Degraded metrics: {performance_drift_results['degraded_metrics']}")
    
    for metric, comparison in performance_drift_results['metric_comparisons'].items():
        print(f"  {metric}: {comparison['baseline']:.3f} → {comparison['current']:.3f} "
              f"({comparison['relative_change']:+.2%})")
    
    logger.info("Drift monitoring demonstration completed!")


if __name__ == "__main__":
    main()