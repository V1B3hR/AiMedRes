"""
Production Readiness and MLOps Testing
Tests for deployment readiness, monitoring, and operational concerns
"""

import pytest
import time
import threading
import multiprocessing
import sys
import os
import tempfile
import json
import pickle
import logging
import subprocess
from typing import Dict, List, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import psutil
import sqlite3
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training import AlzheimerTrainer
from clinical_decision_support import RiskStratificationEngine
from data_quality_monitor import DataQualityMonitor


class TestModelDeployment:
    """Test model deployment and versioning"""
    
    def test_model_serialization_deserialization(self):
        """Test model serialization for deployment"""
        # Train a model
        trainer = AlzheimerTrainer()
        training_data = pd.DataFrame([
            {'age': 65, 'gender': 'M', 'education_level': 12, 'mmse_score': 28, 
             'cdr_score': 0.0, 'apoe_genotype': 'E3/E3', 'diagnosis': 'Normal'},
            {'age': 70, 'gender': 'F', 'education_level': 16, 'mmse_score': 24, 
             'cdr_score': 0.5, 'apoe_genotype': 'E3/E4', 'diagnosis': 'MCI'},
            {'age': 75, 'gender': 'M', 'education_level': 14, 'mmse_score': 20, 
             'cdr_score': 1.0, 'apoe_genotype': 'E4/E4', 'diagnosis': 'Dementia'}
        ] * 30)
        
        trainer.train_model(training_data)
        
        # Test serialization
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            model_path = tmp_file.name
        
        try:
            # Save model
            trainer.save_model(model_path)
            assert os.path.exists(model_path)
            
            # Load model into new trainer
            new_trainer = AlzheimerTrainer()
            new_trainer.load_model(model_path)
            
            # Test that loaded model works
            test_data = training_data.drop('diagnosis', axis=1).head(5)
            original_predictions = trainer.predict(test_data)
            loaded_predictions = new_trainer.predict(test_data)
            
            # Predictions should be identical
            assert original_predictions == loaded_predictions
            
        finally:
            if os.path.exists(model_path):
                os.unlink(model_path)

    def test_model_versioning(self):
        """Test model versioning and metadata tracking"""
        class ModelVersionManager:
            def __init__(self):
                self.versions = {}
                self.current_version = None
            
            def register_model(self, model, version: str, metadata: Dict[str, Any]):
                """Register a model version with metadata"""
                self.versions[version] = {
                    'model': model,
                    'metadata': metadata,
                    'registered_at': time.time()
                }
                self.current_version = version
            
            def get_model(self, version: str = None):
                """Get model by version (default: current)"""
                version = version or self.current_version
                return self.versions.get(version, {}).get('model')
            
            def get_metadata(self, version: str = None):
                """Get model metadata"""
                version = version or self.current_version
                return self.versions.get(version, {}).get('metadata')
            
            def list_versions(self) -> List[str]:
                """List all registered versions"""
                return list(self.versions.keys())
        
        version_manager = ModelVersionManager()
        
        # Register different model versions
        for i, params in enumerate([{'n_estimators': 50}, {'n_estimators': 100}]):
            trainer = AlzheimerTrainer()
            trainer.config.n_estimators = params['n_estimators']
            
            version = f"v1.{i}"
            metadata = {
                'version': version,
                'parameters': params,
                'training_date': datetime.now().isoformat(),
                'performance_metrics': {'accuracy': 0.85 + (i * 0.02)}
            }
            
            version_manager.register_model(trainer, version, metadata)
        
        # Test version management
        assert len(version_manager.list_versions()) == 2
        assert version_manager.current_version == "v1.1"
        
        # Test retrieving specific version
        v0_model = version_manager.get_model("v1.0")
        v0_metadata = version_manager.get_metadata("v1.0")
        
        assert v0_model is not None
        assert v0_metadata['parameters']['n_estimators'] == 50

    def test_model_validation_pipeline(self):
        """Test model validation before deployment"""
        class ModelValidator:
            def __init__(self):
                self.validation_criteria = {
                    'min_accuracy': 0.8,
                    'max_prediction_time': 1.0,
                    'required_features': ['age', 'gender', 'mmse_score'],
                    'valid_predictions': ['Normal', 'MCI', 'Dementia']
                }
            
            def validate_model(self, model, test_data: pd.DataFrame) -> Dict[str, Any]:
                """Validate model against deployment criteria"""
                validation_results = {
                    'passed': True,
                    'issues': [],
                    'metrics': {}
                }
                
                try:
                    # Test prediction functionality
                    start_time = time.time()
                    predictions = model.predict(test_data.drop('diagnosis', axis=1))
                    prediction_time = time.time() - start_time
                    
                    validation_results['metrics']['prediction_time'] = prediction_time
                    validation_results['metrics']['prediction_count'] = len(predictions)
                    
                    # Validate prediction time
                    if prediction_time > self.validation_criteria['max_prediction_time']:
                        validation_results['issues'].append(
                            f"Prediction time too slow: {prediction_time:.3f}s"
                        )
                        validation_results['passed'] = False
                    
                    # Validate predictions format
                    invalid_predictions = [
                        p for p in predictions 
                        if p not in self.validation_criteria['valid_predictions']
                    ]
                    
                    if invalid_predictions:
                        validation_results['issues'].append(
                            f"Invalid predictions: {set(invalid_predictions)}"
                        )
                        validation_results['passed'] = False
                    
                    # Calculate accuracy if true labels available
                    if 'diagnosis' in test_data.columns:
                        correct = sum(1 for i, pred in enumerate(predictions) 
                                    if pred == test_data.iloc[i]['diagnosis'])
                        accuracy = correct / len(predictions)
                        validation_results['metrics']['accuracy'] = accuracy
                        
                        if accuracy < self.validation_criteria['min_accuracy']:
                            validation_results['issues'].append(
                                f"Accuracy too low: {accuracy:.3f}"
                            )
                            validation_results['passed'] = False
                
                except Exception as e:
                    validation_results['passed'] = False
                    validation_results['issues'].append(f"Validation error: {str(e)}")
                
                return validation_results
        
        # Create and train model
        trainer = AlzheimerTrainer()
        training_data = pd.DataFrame([
            {'age': 65, 'gender': 'M', 'education_level': 12, 'mmse_score': 28, 
             'cdr_score': 0.0, 'apoe_genotype': 'E3/E3', 'diagnosis': 'Normal'},
            {'age': 70, 'gender': 'F', 'education_level': 16, 'mmse_score': 24, 
             'cdr_score': 0.5, 'apoe_genotype': 'E3/E4', 'diagnosis': 'MCI'}
        ] * 50)
        
        trainer.train_model(training_data)
        
        # Validate model
        validator = ModelValidator()
        test_data = training_data.head(10)
        
        validation_results = validator.validate_model(trainer, test_data)
        
        # Check validation results
        assert 'passed' in validation_results
        assert 'metrics' in validation_results
        assert 'prediction_time' in validation_results['metrics']
        
        if not validation_results['passed']:
            print(f"Validation issues: {validation_results['issues']}")

    def test_a_b_testing_framework(self):
        """Test A/B testing framework for model comparison"""
        class ABTestingFramework:
            def __init__(self):
                self.experiments = {}
            
            def create_experiment(self, experiment_id: str, model_a, model_b, 
                                traffic_split: float = 0.5):
                """Create A/B test experiment"""
                self.experiments[experiment_id] = {
                    'model_a': model_a,
                    'model_b': model_b,
                    'traffic_split': traffic_split,
                    'results_a': [],
                    'results_b': [],
                    'created_at': time.time()
                }
            
            def route_prediction(self, experiment_id: str, user_id: str, 
                               input_data: pd.DataFrame):
                """Route prediction to A or B model based on user ID"""
                if experiment_id not in self.experiments:
                    raise ValueError(f"Experiment {experiment_id} not found")
                
                experiment = self.experiments[experiment_id]
                
                # Simple hash-based routing for consistent assignment
                user_hash = hash(user_id) % 100
                use_model_a = user_hash < (experiment['traffic_split'] * 100)
                
                if use_model_a:
                    model = experiment['model_a']
                    model_version = 'A'
                else:
                    model = experiment['model_b']
                    model_version = 'B'
                
                # Make prediction
                prediction = model.predict(input_data)
                
                # Record result
                result = {
                    'user_id': user_id,
                    'prediction': prediction,
                    'timestamp': time.time(),
                    'model_version': model_version
                }
                
                if use_model_a:
                    experiment['results_a'].append(result)
                else:
                    experiment['results_b'].append(result)
                
                return prediction, model_version
            
            def get_experiment_stats(self, experiment_id: str) -> Dict[str, Any]:
                """Get experiment statistics"""
                if experiment_id not in self.experiments:
                    return {}
                
                experiment = self.experiments[experiment_id]
                
                return {
                    'total_predictions_a': len(experiment['results_a']),
                    'total_predictions_b': len(experiment['results_b']),
                    'traffic_split_actual': (
                        len(experiment['results_a']) / 
                        (len(experiment['results_a']) + len(experiment['results_b']))
                        if experiment['results_a'] or experiment['results_b'] else 0
                    ),
                    'experiment_duration': time.time() - experiment['created_at']
                }
        
        # Create two different models
        trainer_a = AlzheimerTrainer()
        trainer_a.config.n_estimators = 50
        
        trainer_b = AlzheimerTrainer()
        trainer_b.config.n_estimators = 100
        
        # Train both models
        training_data = pd.DataFrame([
            {'age': 65, 'gender': 'M', 'education_level': 12, 'mmse_score': 28, 
             'cdr_score': 0.0, 'apoe_genotype': 'E3/E3', 'diagnosis': 'Normal'}
        ] * 20)
        
        trainer_a.train_model(training_data)
        trainer_b.train_model(training_data)
        
        # Set up A/B test
        ab_framework = ABTestingFramework()
        ab_framework.create_experiment('alzheimer_model_test', trainer_a, trainer_b, 0.5)
        
        # Simulate user predictions
        test_input = training_data.drop('diagnosis', axis=1).head(1)
        
        predictions = []
        for i in range(20):
            user_id = f"user_{i}"
            prediction, model_version = ab_framework.route_prediction(
                'alzheimer_model_test', user_id, test_input
            )
            predictions.append((prediction, model_version))
        
        # Check experiment stats
        stats = ab_framework.get_experiment_stats('alzheimer_model_test')
        
        assert stats['total_predictions_a'] + stats['total_predictions_b'] == 20
        assert 0.3 <= stats['traffic_split_actual'] <= 0.7  # Should be roughly 50/50


class TestMonitoringAndAlerting:
    """Test production monitoring and alerting systems"""
    
    def test_performance_monitoring(self):
        """Test performance monitoring system"""
        class PerformanceMonitor:
            def __init__(self):
                self.metrics = {
                    'prediction_times': [],
                    'prediction_counts': [],
                    'error_counts': [],
                    'memory_usage': []
                }
                self.thresholds = {
                    'max_prediction_time': 2.0,
                    'max_error_rate': 0.05,
                    'max_memory_usage': 1024  # MB
                }
            
            def record_prediction(self, prediction_time: float, success: bool = True):
                """Record prediction metrics"""
                self.metrics['prediction_times'].append(prediction_time)
                self.metrics['prediction_counts'].append(1)
                if not success:
                    self.metrics['error_counts'].append(1)
                else:
                    self.metrics['error_counts'].append(0)
                
                # Record memory usage
                memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
                self.metrics['memory_usage'].append(memory_mb)
            
            def get_current_metrics(self) -> Dict[str, float]:
                """Get current performance metrics"""
                if not self.metrics['prediction_times']:
                    return {}
                
                recent_window = 100  # Last 100 predictions
                
                recent_times = self.metrics['prediction_times'][-recent_window:]
                recent_errors = self.metrics['error_counts'][-recent_window:]
                recent_memory = self.metrics['memory_usage'][-recent_window:]
                
                return {
                    'avg_prediction_time': np.mean(recent_times),
                    'max_prediction_time': np.max(recent_times),
                    'error_rate': np.mean(recent_errors),
                    'avg_memory_usage': np.mean(recent_memory),
                    'peak_memory_usage': np.max(recent_memory)
                }
            
            def check_alerts(self) -> List[Dict[str, Any]]:
                """Check for alert conditions"""
                alerts = []
                metrics = self.get_current_metrics()
                
                if not metrics:
                    return alerts
                
                # Check prediction time alert
                if metrics['max_prediction_time'] > self.thresholds['max_prediction_time']:
                    alerts.append({
                        'type': 'performance',
                        'severity': 'warning',
                        'message': f"Slow predictions detected: {metrics['max_prediction_time']:.2f}s",
                        'timestamp': time.time()
                    })
                
                # Check error rate alert
                if metrics['error_rate'] > self.thresholds['max_error_rate']:
                    alerts.append({
                        'type': 'reliability',
                        'severity': 'critical',
                        'message': f"High error rate: {metrics['error_rate']:.2%}",
                        'timestamp': time.time()
                    })
                
                # Check memory usage alert
                if metrics['peak_memory_usage'] > self.thresholds['max_memory_usage']:
                    alerts.append({
                        'type': 'resource',
                        'severity': 'warning',
                        'message': f"High memory usage: {metrics['peak_memory_usage']:.1f}MB",
                        'timestamp': time.time()
                    })
                
                return alerts
        
        # Test performance monitoring
        monitor = PerformanceMonitor()
        
        # Simulate predictions with varying performance
        prediction_times = [0.1, 0.2, 0.15, 2.5, 0.3]  # One slow prediction
        success_rates = [True, True, True, False, True]  # One error
        
        for time_taken, success in zip(prediction_times, success_rates):
            monitor.record_prediction(time_taken, success)
        
        # Check metrics
        metrics = monitor.get_current_metrics()
        assert 'avg_prediction_time' in metrics
        assert 'error_rate' in metrics
        assert metrics['error_rate'] == 0.2  # 1 error out of 5
        
        # Check alerts
        alerts = monitor.check_alerts()
        assert len(alerts) >= 1  # Should have at least performance alert
        
        # Verify alert structure
        for alert in alerts:
            assert 'type' in alert
            assert 'severity' in alert
            assert 'message' in alert
            assert 'timestamp' in alert

    def test_data_drift_detection(self):
        """Test data drift detection for model inputs"""
        class DataDriftDetector:
            def __init__(self, reference_data: pd.DataFrame):
                self.reference_stats = self._calculate_stats(reference_data)
                self.drift_threshold = 0.1  # 10% change threshold
            
            def _calculate_stats(self, data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
                """Calculate statistical properties of data"""
                stats = {}
                
                for column in data.select_dtypes(include=[np.number]).columns:
                    stats[column] = {
                        'mean': float(data[column].mean()),
                        'std': float(data[column].std()),
                        'min': float(data[column].min()),
                        'max': float(data[column].max())
                    }
                
                for column in data.select_dtypes(include=['object']).columns:
                    value_counts = data[column].value_counts(normalize=True)
                    stats[column] = dict(value_counts)
                
                return stats
            
            def detect_drift(self, new_data: pd.DataFrame) -> Dict[str, Any]:
                """Detect drift in new data compared to reference"""
                new_stats = self._calculate_stats(new_data)
                drift_results = {
                    'has_drift': False,
                    'drifted_features': [],
                    'drift_scores': {}
                }
                
                # Check numerical features
                for column in new_data.select_dtypes(include=[np.number]).columns:
                    if column in self.reference_stats:
                        ref_mean = self.reference_stats[column]['mean']
                        new_mean = new_stats[column]['mean']
                        
                        # Calculate relative change
                        if ref_mean != 0:
                            drift_score = abs(new_mean - ref_mean) / abs(ref_mean)
                        else:
                            drift_score = abs(new_mean)
                        
                        drift_results['drift_scores'][column] = drift_score
                        
                        if drift_score > self.drift_threshold:
                            drift_results['has_drift'] = True
                            drift_results['drifted_features'].append(column)
                
                # Check categorical features
                for column in new_data.select_dtypes(include=['object']).columns:
                    if column in self.reference_stats:
                        ref_dist = self.reference_stats[column]
                        new_counts = new_data[column].value_counts(normalize=True)
                        
                        # Calculate distribution drift using simple metric
                        total_drift = 0
                        for value in set(list(ref_dist.keys()) + list(new_counts.keys())):
                            ref_prob = ref_dist.get(value, 0)
                            new_prob = new_counts.get(value, 0)
                            total_drift += abs(new_prob - ref_prob)
                        
                        drift_score = total_drift / 2  # Normalize
                        drift_results['drift_scores'][column] = drift_score
                        
                        if drift_score > self.drift_threshold:
                            drift_results['has_drift'] = True
                            drift_results['drifted_features'].append(column)
                
                return drift_results
        
        # Create reference dataset
        reference_data = pd.DataFrame({
            'age': np.random.normal(70, 10, 1000),
            'mmse_score': np.random.normal(25, 3, 1000),
            'gender': np.random.choice(['M', 'F'], 1000, p=[0.4, 0.6])
        })
        
        # Initialize drift detector
        drift_detector = DataDriftDetector(reference_data)
        
        # Test 1: No drift (similar data)
        similar_data = pd.DataFrame({
            'age': np.random.normal(70, 10, 100),
            'mmse_score': np.random.normal(25, 3, 100),
            'gender': np.random.choice(['M', 'F'], 100, p=[0.4, 0.6])
        })
        
        drift_results = drift_detector.detect_drift(similar_data)
        assert drift_results['has_drift'] is False
        
        # Test 2: Drift detected (different distribution)
        drifted_data = pd.DataFrame({
            'age': np.random.normal(80, 5, 100),  # Older population
            'mmse_score': np.random.normal(20, 2, 100),  # Lower scores
            'gender': np.random.choice(['M', 'F'], 100, p=[0.8, 0.2])  # Different gender ratio
        })
        
        drift_results = drift_detector.detect_drift(drifted_data)
        assert drift_results['has_drift'] is True
        assert len(drift_results['drifted_features']) > 0

    def test_model_degradation_detection(self):
        """Test detection of model performance degradation"""
        class ModelDegradationMonitor:
            def __init__(self, baseline_accuracy: float = 0.85):
                self.baseline_accuracy = baseline_accuracy
                self.performance_history = []
                self.degradation_threshold = 0.05  # 5% drop
                self.window_size = 100
            
            def record_prediction_outcome(self, predicted: str, actual: str):
                """Record prediction outcome for accuracy tracking"""
                is_correct = predicted == actual
                self.performance_history.append({
                    'correct': is_correct,
                    'timestamp': time.time(),
                    'predicted': predicted,
                    'actual': actual
                })
                
                # Keep only recent history
                if len(self.performance_history) > self.window_size:
                    self.performance_history = self.performance_history[-self.window_size:]
            
            def get_current_accuracy(self) -> float:
                """Calculate current accuracy from recent predictions"""
                if not self.performance_history:
                    return 0.0
                
                correct_predictions = sum(1 for pred in self.performance_history if pred['correct'])
                return correct_predictions / len(self.performance_history)
            
            def check_degradation(self) -> Dict[str, Any]:
                """Check for model performance degradation"""
                current_accuracy = self.get_current_accuracy()
                
                degradation_amount = self.baseline_accuracy - current_accuracy
                is_degraded = degradation_amount > self.degradation_threshold
                
                return {
                    'is_degraded': is_degraded,
                    'current_accuracy': current_accuracy,
                    'baseline_accuracy': self.baseline_accuracy,
                    'degradation_amount': degradation_amount,
                    'samples_count': len(self.performance_history)
                }
        
        # Test degradation monitoring
        monitor = ModelDegradationMonitor(baseline_accuracy=0.90)
        
        # Simulate good performance initially
        for _ in range(50):
            monitor.record_prediction_outcome('Normal', 'Normal')  # All correct
        
        degradation_check = monitor.check_degradation()
        assert degradation_check['is_degraded'] is False
        assert degradation_check['current_accuracy'] == 1.0
        
        # Simulate performance degradation
        for _ in range(50):
            # Mix of correct and incorrect predictions (worse performance)
            predicted = 'Normal' if _ % 3 == 0 else 'MCI'
            actual = 'Normal'
            monitor.record_prediction_outcome(predicted, actual)
        
        degradation_check = monitor.check_degradation()
        assert degradation_check['is_degraded'] is True
        assert degradation_check['current_accuracy'] < 0.85

    def test_system_health_monitoring(self):
        """Test overall system health monitoring"""
        class SystemHealthMonitor:
            def __init__(self):
                self.health_checks = {}
                self.last_check_time = {}
            
            def register_health_check(self, name: str, check_function: callable, 
                                    interval_seconds: int = 60):
                """Register a health check function"""
                self.health_checks[name] = {
                    'function': check_function,
                    'interval': interval_seconds,
                    'last_result': None,
                    'last_error': None
                }
                self.last_check_time[name] = 0
            
            def run_health_checks(self) -> Dict[str, Any]:
                """Run all registered health checks"""
                current_time = time.time()
                results = {}
                
                for name, check_config in self.health_checks.items():
                    # Check if it's time to run this health check
                    if (current_time - self.last_check_time[name]) >= check_config['interval']:
                        try:
                            result = check_config['function']()
                            check_config['last_result'] = result
                            check_config['last_error'] = None
                            results[name] = {
                                'status': 'healthy',
                                'result': result,
                                'timestamp': current_time
                            }
                        except Exception as e:
                            check_config['last_error'] = str(e)
                            results[name] = {
                                'status': 'unhealthy',
                                'error': str(e),
                                'timestamp': current_time
                            }
                        
                        self.last_check_time[name] = current_time
                    else:
                        # Use last result
                        results[name] = {
                            'status': 'cached',
                            'result': check_config['last_result'],
                            'timestamp': self.last_check_time[name]
                        }
                
                return results
            
            def get_overall_health(self) -> Dict[str, Any]:
                """Get overall system health status"""
                health_results = self.run_health_checks()
                
                healthy_checks = sum(1 for result in health_results.values() 
                                   if result['status'] == 'healthy')
                total_checks = len(health_results)
                
                overall_status = 'healthy' if healthy_checks == total_checks else 'degraded'
                if healthy_checks == 0:
                    overall_status = 'critical'
                
                return {
                    'overall_status': overall_status,
                    'healthy_checks': healthy_checks,
                    'total_checks': total_checks,
                    'health_score': healthy_checks / total_checks if total_checks > 0 else 0,
                    'details': health_results
                }
        
        # Define health check functions
        def database_health_check():
            """Mock database connectivity check"""
            # Simulate database check
            time.sleep(0.01)  # Small delay to simulate check
            return {'connection': 'active', 'response_time': 0.01}
        
        def model_health_check():
            """Mock model availability check"""
            trainer = AlzheimerTrainer()
            # Simple prediction test
            test_data = pd.DataFrame([{
                'age': 70, 'gender': 'M', 'education_level': 14,
                'mmse_score': 25, 'cdr_score': 0.5, 'apoe_genotype': 'E3/E4'
            }])
            
            # Create a simple trained model for testing
            training_data = pd.DataFrame([
                {'age': 65, 'gender': 'M', 'education_level': 12, 'mmse_score': 28, 
                 'cdr_score': 0.0, 'apoe_genotype': 'E3/E3', 'diagnosis': 'Normal'}
            ] * 10)
            trainer.train_model(training_data)
            
            prediction = trainer.predict(test_data)
            return {'model_responsive': True, 'prediction_count': len(prediction)}
        
        def memory_health_check():
            """Check system memory usage"""
            memory_info = psutil.virtual_memory()
            return {
                'memory_usage_percent': memory_info.percent,
                'available_gb': memory_info.available / (1024**3)
            }
        
        # Set up health monitoring
        health_monitor = SystemHealthMonitor()
        health_monitor.register_health_check('database', database_health_check, 30)
        health_monitor.register_health_check('model', model_health_check, 60)
        health_monitor.register_health_check('memory', memory_health_check, 10)
        
        # Run health checks
        overall_health = health_monitor.get_overall_health()
        
        # Verify health monitoring
        assert 'overall_status' in overall_health
        assert 'health_score' in overall_health
        assert 'details' in overall_health
        assert len(overall_health['details']) == 3
        
        # Check individual health check results
        for check_name, result in overall_health['details'].items():
            assert 'status' in result
            assert 'timestamp' in result


class TestDisasterRecovery:
    """Test disaster recovery and backup procedures"""
    
    def test_data_backup_and_restore(self):
        """Test data backup and restore procedures"""
        class BackupManager:
            def __init__(self, backup_dir: str):
                self.backup_dir = Path(backup_dir)
                self.backup_dir.mkdir(exist_ok=True)
            
            def create_backup(self, data: pd.DataFrame, backup_name: str) -> str:
                """Create a backup of data"""
                backup_path = self.backup_dir / f"{backup_name}_{int(time.time())}.csv"
                data.to_csv(backup_path, index=False)
                
                # Create metadata file
                metadata_path = backup_path.with_suffix('.meta.json')
                metadata = {
                    'backup_name': backup_name,
                    'created_at': time.time(),
                    'row_count': len(data),
                    'columns': list(data.columns),
                    'file_size_bytes': backup_path.stat().st_size
                }
                
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                return str(backup_path)
            
            def restore_backup(self, backup_path: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
                """Restore data from backup"""
                # Load data
                data = pd.read_csv(backup_path)
                
                # Load metadata
                metadata_path = Path(backup_path).with_suffix('.meta.json')
                if metadata_path.exists():
                    with open(metadata_path) as f:
                        metadata = json.load(f)
                else:
                    metadata = {}
                
                return data, metadata
            
            def list_backups(self) -> List[Dict[str, Any]]:
                """List available backups"""
                backups = []
                for csv_file in self.backup_dir.glob('*.csv'):
                    metadata_file = csv_file.with_suffix('.meta.json')
                    if metadata_file.exists():
                        with open(metadata_file) as f:
                            metadata = json.load(f)
                        metadata['backup_path'] = str(csv_file)
                        backups.append(metadata)
                
                return sorted(backups, key=lambda x: x['created_at'], reverse=True)
            
            def cleanup_old_backups(self, retention_days: int = 30):
                """Clean up backups older than retention period"""
                cutoff_time = time.time() - (retention_days * 24 * 3600)
                
                for backup_info in self.list_backups():
                    if backup_info['created_at'] < cutoff_time:
                        backup_path = Path(backup_info['backup_path'])
                        metadata_path = backup_path.with_suffix('.meta.json')
                        
                        backup_path.unlink()
                        if metadata_path.exists():
                            metadata_path.unlink()
        
        # Test backup functionality
        with tempfile.TemporaryDirectory() as temp_dir:
            backup_manager = BackupManager(temp_dir)
            
            # Create test data
            test_data = pd.DataFrame({
                'patient_id': ['P001', 'P002', 'P003'],
                'age': [65, 70, 75],
                'diagnosis': ['Normal', 'MCI', 'Dementia']
            })
            
            # Create backup
            backup_path = backup_manager.create_backup(test_data, 'patient_data')
            assert os.path.exists(backup_path)
            
            # List backups
            backups = backup_manager.list_backups()
            assert len(backups) == 1
            assert backups[0]['backup_name'] == 'patient_data'
            assert backups[0]['row_count'] == 3
            
            # Restore backup
            restored_data, metadata = backup_manager.restore_backup(backup_path)
            
            # Verify restoration
            assert len(restored_data) == len(test_data)
            assert list(restored_data.columns) == list(test_data.columns)
            assert restored_data.equals(test_data)
            assert metadata['backup_name'] == 'patient_data'

    def test_model_backup_and_recovery(self):
        """Test model backup and recovery procedures"""
        class ModelBackupManager:
            def __init__(self, backup_dir: str):
                self.backup_dir = Path(backup_dir)
                self.backup_dir.mkdir(exist_ok=True)
            
            def backup_model(self, model, model_name: str, metadata: Dict[str, Any] = None) -> str:
                """Backup a trained model"""
                timestamp = int(time.time())
                backup_path = self.backup_dir / f"{model_name}_{timestamp}.pkl"
                
                # Save model
                with open(backup_path, 'wb') as f:
                    pickle.dump(model, f)
                
                # Save metadata
                metadata = metadata or {}
                metadata.update({
                    'model_name': model_name,
                    'backed_up_at': timestamp,
                    'backup_path': str(backup_path),
                    'file_size_bytes': backup_path.stat().st_size
                })
                
                metadata_path = backup_path.with_suffix('.meta.json')
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                return str(backup_path)
            
            def restore_model(self, backup_path: str) -> Tuple[Any, Dict[str, Any]]:
                """Restore model from backup"""
                # Load model
                with open(backup_path, 'rb') as f:
                    model = pickle.load(f)
                
                # Load metadata
                metadata_path = Path(backup_path).with_suffix('.meta.json')
                if metadata_path.exists():
                    with open(metadata_path) as f:
                        metadata = json.load(f)
                else:
                    metadata = {}
                
                return model, metadata
        
        # Test model backup
        with tempfile.TemporaryDirectory() as temp_dir:
            model_backup_manager = ModelBackupManager(temp_dir)
            
            # Train a model
            trainer = AlzheimerTrainer()
            training_data = pd.DataFrame([
                {'age': 65, 'gender': 'M', 'education_level': 12, 'mmse_score': 28, 
                 'cdr_score': 0.0, 'apoe_genotype': 'E3/E3', 'diagnosis': 'Normal'}
            ] * 20)
            trainer.train_model(training_data)
            
            # Backup model
            model_metadata = {
                'version': '1.0',
                'training_samples': len(training_data),
                'performance_metrics': {'accuracy': 0.85}
            }
            
            backup_path = model_backup_manager.backup_model(
                trainer.model, 'alzheimer_model', model_metadata
            )
            
            assert os.path.exists(backup_path)
            
            # Restore model
            restored_model, metadata = model_backup_manager.restore_model(backup_path)
            
            # Verify restoration
            assert restored_model is not None
            assert metadata['model_name'] == 'alzheimer_model'
            assert metadata['version'] == '1.0'
            
            # Test that restored model works
            test_data = training_data.drop('diagnosis', axis=1).head(1)
            
            # Create new trainer with restored model
            new_trainer = AlzheimerTrainer()
            new_trainer.model = restored_model
            
            prediction = new_trainer.predict(test_data)
            assert len(prediction) == 1

    def test_failover_mechanisms(self):
        """Test automatic failover mechanisms"""
        class FailoverManager:
            def __init__(self):
                self.primary_service = None
                self.backup_services = []
                self.current_service = None
                self.health_check_interval = 1  # seconds
                self.last_health_check = 0
            
            def register_primary(self, service):
                """Register primary service"""
                self.primary_service = service
                self.current_service = service
            
            def register_backup(self, service):
                """Register backup service"""
                self.backup_services.append(service)
            
            def health_check(self, service) -> bool:
                """Check if service is healthy"""
                try:
                    # Simple health check - try to call a method
                    if hasattr(service, 'health_check'):
                        return service.health_check()
                    else:
                        # Default: try to make a prediction
                        test_data = pd.DataFrame([{
                            'age': 70, 'gender': 'M', 'education_level': 14,
                            'mmse_score': 25, 'cdr_score': 0.5, 'apoe_genotype': 'E3/E4'
                        }])
                        service.predict(test_data)
                        return True
                except Exception:
                    return False
            
            def get_active_service(self):
                """Get currently active service with automatic failover"""
                current_time = time.time()
                
                # Check if it's time for health check
                if (current_time - self.last_health_check) >= self.health_check_interval:
                    # Check current service health
                    if not self.health_check(self.current_service):
                        # Current service is down, try to failover
                        failover_successful = False
                        
                        # Try primary service first (if not current)
                        if (self.current_service != self.primary_service and 
                            self.health_check(self.primary_service)):
                            self.current_service = self.primary_service
                            failover_successful = True
                        else:
                            # Try backup services
                            for backup_service in self.backup_services:
                                if (backup_service != self.current_service and 
                                    self.health_check(backup_service)):
                                    self.current_service = backup_service
                                    failover_successful = True
                                    break
                        
                        if not failover_successful:
                            raise RuntimeError("All services are down - failover failed")
                    
                    self.last_health_check = current_time
                
                return self.current_service
        
        # Create mock services
        class MockService:
            def __init__(self, name: str, should_fail: bool = False):
                self.name = name
                self.should_fail = should_fail
                self.call_count = 0
            
            def predict(self, data):
                self.call_count += 1
                if self.should_fail:
                    raise RuntimeError(f"Service {self.name} is down")
                return [f"prediction_from_{self.name}"]
            
            def health_check(self):
                return not self.should_fail
        
        # Test failover
        primary_service = MockService("primary", should_fail=False)
        backup_service = MockService("backup", should_fail=False)
        
        failover_manager = FailoverManager()
        failover_manager.register_primary(primary_service)
        failover_manager.register_backup(backup_service)
        
        # Test normal operation
        active_service = failover_manager.get_active_service()
        assert active_service == primary_service
        
        # Simulate primary service failure
        primary_service.should_fail = True
        
        # Should failover to backup
        active_service = failover_manager.get_active_service()
        assert active_service == backup_service
        
        # Test prediction works on backup
        test_data = pd.DataFrame([{'age': 70, 'gender': 'M', 'education_level': 14,
                                 'mmse_score': 25, 'cdr_score': 0.5, 'apoe_genotype': 'E3/E4'}])
        prediction = active_service.predict(test_data)
        assert prediction == ["prediction_from_backup"]