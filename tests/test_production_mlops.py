#!/usr/bin/env python3
"""
Tests for Production MLOps Features.
Comprehensive tests for monitoring, A/B testing, and data-driven retraining.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
import json
import sqlite3
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import time

# Import the modules to test
from mlops.monitoring.production_monitor import ProductionMonitor, MonitoringMetrics, AlertConfig
from mlops.monitoring.data_trigger import DataDrivenRetrainingTrigger, RetrainingTriggerConfig
from mlops.monitoring.ab_testing import ABTestingManager, ABTestConfig, ExperimentStatus
from mlops.serving.model_loader import ModelLoader
from mlops.serving.production_server import ProductionServer


class TestProductionMonitor:
    """Test production monitoring functionality."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(0, 1, 100),
            'feature3': np.random.normal(0, 1, 100)
        })
        labels = pd.Series(np.random.choice([0, 1], 100))
        return data, labels
    
    @pytest.fixture
    def production_monitor(self, sample_data):
        """Create a production monitor for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            
            # Use a non-existent Redis host to test graceful fallback
            monitor = ProductionMonitor(
                model_name="test_model",
                mlflow_tracking_uri="sqlite:///test_mlflow.db",
                redis_host="non_existent_host",  # Will fail gracefully
                redis_port=6379
            )
            
            # Set baseline metrics
            baseline_data, baseline_labels = sample_data
            monitor.set_baseline_metrics(baseline_data, baseline_labels)
            
            yield monitor
    
    def test_monitor_initialization(self, production_monitor):
        """Test monitor initialization."""
        assert production_monitor.model_name == "test_model"
        assert production_monitor.baseline_metrics is not None
        assert production_monitor.drift_monitor is not None
    
    def test_log_prediction_batch(self, production_monitor, sample_data):
        """Test logging prediction batches."""
        baseline_data, baseline_labels = sample_data
        current_data = baseline_data.iloc[:50]
        predictions = np.random.choice([0, 1], 50)
        true_labels = baseline_labels.iloc[:50].values
        
        result = production_monitor.log_prediction_batch(
            features=current_data,
            predictions=predictions,
            true_labels=true_labels,
            model_version="v1.0.0",
            latency_ms=100.0
        )
        
        assert 'metrics' in result
        assert result['metrics']['prediction_count'] == 50
        assert result['metrics']['model_version'] == "v1.0.0"
        assert result['metrics']['latency_ms'] == 100.0
        assert 0 <= result['metrics']['accuracy'] <= 1
    
    def test_alert_triggering(self, production_monitor, sample_data):
        """Test alert triggering based on thresholds."""
        # Create alert config with low thresholds
        alert_config = AlertConfig(
            critical_accuracy_drop=0.01,
            critical_latency_ms=50.0
        )
        production_monitor.alert_config = alert_config
        
        baseline_data, _ = sample_data
        current_data = baseline_data.iloc[:20]
        
        # Log predictions with high latency (should trigger alert)
        predictions = np.zeros(20)  # Poor predictions
        
        result = production_monitor.log_prediction_batch(
            features=current_data,
            predictions=predictions,
            true_labels=np.ones(20),  # All true labels are 1, predictions are 0
            latency_ms=200.0  # High latency
        )
        
        # Check if alerts were triggered
        recent_alerts = production_monitor._get_recent_alerts(minutes=1)
        assert len(recent_alerts) > 0
    
    def test_monitoring_summary(self, production_monitor, sample_data):
        """Test monitoring summary generation."""
        baseline_data, baseline_labels = sample_data
        current_data = baseline_data.iloc[:30]
        predictions = np.random.choice([0, 1], 30)
        
        # Log some predictions
        production_monitor.log_prediction_batch(
            features=current_data,
            predictions=predictions,
            true_labels=baseline_labels.iloc[:30].values,
            latency_ms=75.0
        )
        
        # Get summary
        summary = production_monitor.get_monitoring_summary(hours=1)
        
        assert summary['total_predictions'] == 30
        assert 'avg_accuracy' in summary
        assert 'avg_latency_ms' in summary
        assert 'status' in summary
        assert summary['status'] in ['HEALTHY', 'DEGRADED', 'CRITICAL']


class TestDataDrivenRetrainingTrigger:
    """Test automated retraining trigger functionality."""
    
    @pytest.fixture
    def trigger_config(self):
        """Create trigger configuration for testing."""
        return RetrainingTriggerConfig(
            min_new_samples=50,
            max_days_without_retrain=1,  # Short for testing
            accuracy_degradation_threshold=0.05,
            min_hours_between_retrains=0,  # No delay for testing
            max_retrains_per_day=10,
            require_manual_approval=False
        )
    
    @pytest.fixture
    def retraining_trigger(self, trigger_config):
        """Create retraining trigger for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            
            trigger = DataDrivenRetrainingTrigger(
                model_name="test_model",
                config=trigger_config,
                mlflow_tracking_uri="sqlite:///test_mlflow.db"
            )
            
            yield trigger
    
    def test_trigger_initialization(self, retraining_trigger):
        """Test trigger initialization."""
        assert retraining_trigger.model_name == "test_model" 
        assert retraining_trigger.config.min_new_samples == 50
        assert not retraining_trigger.is_active
    
    def test_trigger_constraints(self, retraining_trigger):
        """Test retraining constraints."""
        # Test that retraining is allowed initially
        assert retraining_trigger._can_retrain()
        
        # Simulate recent retrain
        retraining_trigger.last_retrain_time = datetime.now()
        retraining_trigger.config.min_hours_between_retrains = 24
        
        # Should not allow retrain too soon
        assert not retraining_trigger._can_retrain()
        
        # Test daily limit
        retraining_trigger.config.min_hours_between_retrains = 0
        retraining_trigger.retrains_today = 10
        retraining_trigger.config.max_retrains_per_day = 5
        
        assert not retraining_trigger._can_retrain()
    
    @patch('subprocess.run')
    def test_force_retrain(self, mock_subprocess, retraining_trigger):
        """Test force retraining."""
        # Mock successful training pipeline
        mock_subprocess.return_value.returncode = 0
        mock_subprocess.return_value.stderr = ""
        
        # Force retrain should work regardless of constraints
        result = retraining_trigger.force_retrain("Test force retrain")
        
        assert result is True
        
        # Check history
        history = retraining_trigger.get_trigger_history(days=1)
        assert len(history) >= 1
        assert history[0]['trigger_type'] == 'manual'
    
    def test_trigger_monitoring_lifecycle(self, retraining_trigger):
        """Test starting and stopping monitoring."""
        assert not retraining_trigger.is_active
        
        # Start monitoring
        retraining_trigger.start_monitoring()
        assert retraining_trigger.is_active
        
        # Stop monitoring
        retraining_trigger.stop_monitoring()
        assert not retraining_trigger.is_active


class TestABTestingManager:
    """Test A/B testing functionality."""
    
    @pytest.fixture
    def ab_config(self):
        """Create A/B test configuration."""
        return ABTestConfig(
            experiment_name="test_experiment",
            model_a_version="v1.0.0",
            model_b_version="v2.0.0",
            traffic_split=0.5,
            duration_days=7,
            min_samples_per_variant=20
        )
    
    @pytest.fixture
    def ab_manager(self):
        """Create A/B testing manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            
            manager = ABTestingManager(
                mlflow_tracking_uri="sqlite:///test_mlflow.db",
                experiment_db="test_ab.db"
            )
            
            yield manager
    
    def test_ab_manager_initialization(self, ab_manager):
        """Test A/B manager initialization."""
        assert ab_manager.experiment_db == "test_ab.db"
        assert ab_manager.active_experiments == {}
    
    def test_create_experiment(self, ab_manager, ab_config):
        """Test creating A/B test experiment."""
        experiment_name = ab_manager.create_experiment(ab_config)
        
        assert experiment_name == "test_experiment"
        
        # Check database
        with sqlite3.connect(ab_manager.experiment_db) as conn:
            cursor = conn.execute(
                "SELECT experiment_name, status FROM ab_experiments WHERE experiment_name = ?",
                (experiment_name,)
            )
            row = cursor.fetchone()
            
            assert row is not None
            assert row[0] == experiment_name
            assert row[1] == ExperimentStatus.DRAFT.value
    
    def test_start_experiment(self, ab_manager, ab_config):
        """Test starting an experiment."""
        # Create experiment
        experiment_name = ab_manager.create_experiment(ab_config)
        
        # Start experiment
        success = ab_manager.start_experiment(experiment_name)
        assert success
        
        # Check that experiment is now active
        assert experiment_name in ab_manager.active_experiments
        
        # Check database status
        with sqlite3.connect(ab_manager.experiment_db) as conn:
            cursor = conn.execute(
                "SELECT status FROM ab_experiments WHERE experiment_name = ?",
                (experiment_name,)
            )
            row = cursor.fetchone()
            assert row[0] == ExperimentStatus.RUNNING.value
    
    def test_make_prediction_with_ab_testing(self, ab_manager, ab_config):
        """Test making predictions through A/B testing."""
        # Create and start experiment
        experiment_name = ab_manager.create_experiment(ab_config)
        ab_manager.start_experiment(experiment_name)
        
        # Mock models
        class MockModel:
            def predict(self, X):
                return np.array([1])
        
        models = {
            "v1.0.0": MockModel(),
            "v2.0.0": MockModel()
        }
        
        # Make predictions for multiple users
        results = []
        for i in range(100):
            result = ab_manager.make_prediction(
                experiment_name=experiment_name,
                user_id=f"user_{i}",
                features={'age': 65, 'mmse': 25},
                models=models
            )
            results.append(result)
        
        # Check that we got both variants
        variants = [r['variant'] for r in results]
        assert 'control' in variants
        assert 'treatment' in variants
        
        # Check predictions were made
        for result in results[:5]:  # Check first 5
            assert 'prediction' in result
            assert 'model_version' in result
            assert result['model_version'] in ["v1.0.0", "v2.0.0"]
    
    def test_experiment_analysis(self, ab_manager, ab_config):
        """Test experiment analysis and results."""
        # Create and start experiment
        experiment_name = ab_manager.create_experiment(ab_config)
        ab_manager.start_experiment(experiment_name)
        
        # Mock models with different accuracies
        class MockModel:
            def __init__(self, accuracy):
                self.accuracy = accuracy
            
            def predict(self, X):
                return np.array([1 if np.random.random() < self.accuracy else 0])
        
        models = {
            "v1.0.0": MockModel(0.7),  # 70% accuracy
            "v2.0.0": MockModel(0.8)   # 80% accuracy
        }
        
        # Generate predictions and outcomes
        user_predictions = []
        for i in range(100):
            result = ab_manager.make_prediction(
                experiment_name=experiment_name,
                user_id=f"user_{i}",
                features={'age': 65, 'mmse': 25},
                models=models
            )
            
            # Simulate true outcome
            true_label = 1 if np.random.random() < 0.75 else 0
            user_predictions.append((f"user_{i}", true_label))
        
        # Update with true outcomes
        ab_manager.update_prediction_outcomes(experiment_name, user_predictions)
        
        # Analyze experiment
        results = ab_manager.analyze_experiment(experiment_name)
        
        assert results.experiment_name == experiment_name
        assert results.samples_a > 0
        assert results.samples_b > 0
        assert 'accuracy' in results.metrics_a
        assert 'accuracy' in results.metrics_b
        assert results.recommendation is not None
    
    def test_stop_experiment(self, ab_manager, ab_config):
        """Test stopping an experiment."""
        # Create, start experiment, add some data
        experiment_name = ab_manager.create_experiment(ab_config)
        ab_manager.start_experiment(experiment_name)
        
        # Add minimal data for analysis
        with sqlite3.connect(ab_manager.experiment_db) as conn:
            conn.execute("""
                INSERT INTO ab_predictions (
                    timestamp, experiment_name, user_id, model_version,
                    variant, features, prediction, true_label, correct
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                experiment_name,
                "test_user",
                "v1.0.0",
                "control",
                json.dumps({'age': 65}),
                json.dumps(1),
                json.dumps(1),
                True
            ))
            conn.commit()
        
        # Stop experiment
        results = ab_manager.stop_experiment(experiment_name)
        
        assert results.experiment_name == experiment_name
        assert experiment_name not in ab_manager.active_experiments
        
        # Check database status
        with sqlite3.connect(ab_manager.experiment_db) as conn:
            cursor = conn.execute(
                "SELECT status FROM ab_experiments WHERE experiment_name = ?",
                (experiment_name,)
            )
            row = cursor.fetchone()
            assert row[0] == ExperimentStatus.COMPLETED.value


class TestModelLoader:
    """Test model loading functionality."""
    
    @pytest.fixture
    def model_loader(self):
        """Create model loader for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            
            loader = ModelLoader(mlflow_tracking_uri="sqlite:///test_mlflow.db")
            yield loader
    
    def test_model_loader_initialization(self, model_loader):
        """Test model loader initialization."""
        assert model_loader.mlflow_tracking_uri == "sqlite:///test_mlflow.db"
        assert model_loader.model_cache == {}
    
    def test_model_validation(self, model_loader):
        """Test model validation."""
        # Create a mock model
        class MockModel:
            def predict(self, X):
                return np.array([1, 0])
            
            def predict_proba(self, X):
                return np.array([[0.3, 0.7], [0.8, 0.2]])
        
        mock_model = MockModel()
        
        # Validate model
        validation = model_loader.validate_model(mock_model)
        
        assert validation['is_valid']
        assert validation['has_predict']
        assert validation['has_predict_proba']
        assert len(validation['errors']) == 0
        assert 'sample_prediction' in validation
        assert 'sample_probabilities' in validation
    
    def test_cache_functionality(self, model_loader):
        """Test model caching."""
        # Create mock model
        mock_model = Mock()
        mock_model.predict.return_value = np.array([1])
        
        # Manually add to cache
        cache_key = "test_model:v1.0.0"
        model_loader.model_cache[cache_key] = mock_model
        
        # Check cache info
        cache_info = model_loader.get_cache_info()
        assert cache_info['cache_size'] == 1
        assert cache_key in cache_info['cached_models']
        
        # Clear cache
        model_loader.clear_cache()
        cache_info = model_loader.get_cache_info()
        assert cache_info['cache_size'] == 0


class TestProductionServer:
    """Test production server functionality."""
    
    @pytest.fixture
    def production_server(self):
        """Create production server for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            
            server = ProductionServer(
                model_name="test_model",
                mlflow_tracking_uri="sqlite:///test_mlflow.db",
                enable_monitoring=True,
                enable_ab_testing=True
            )
            
            # Mock a simple model
            class MockModel:
                def predict(self, X):
                    return np.array([1])
                
                def predict_proba(self, X):
                    return np.array([[0.3, 0.7]])
            
            # Add mock model
            server.models["v1.0.0"] = MockModel()
            server.current_model_version = "v1.0.0"
            
            yield server
    
    def test_production_server_initialization(self, production_server):
        """Test production server initialization."""
        assert production_server.model_name == "test_model"
        assert production_server.enable_monitoring
        assert production_server.enable_ab_testing
        assert production_server.monitor is not None
        assert production_server.ab_manager is not None
    
    def test_health_check_endpoint(self, production_server):
        """Test health check endpoint."""
        with production_server.app.test_client() as client:
            response = client.get('/health')
            
            assert response.status_code == 200
            data = json.loads(response.data)
            
            assert data['status'] == 'healthy'
            assert data['model_name'] == 'test_model'
            assert 'uptime_seconds' in data
            assert 'request_count' in data
    
    def test_prediction_endpoint(self, production_server):
        """Test prediction endpoint."""
        with production_server.app.test_client() as client:
            # Test prediction request
            response = client.post('/predict', 
                json={
                    'features': {
                        'M/F': 0,
                        'Age': 65,
                        'EDUC': 12,
                        'SES': 2,
                        'MMSE': 25,
                        'CDR': 0.5,
                        'eTIV': 1500,
                        'nWBV': 0.75,
                        'ASF': 1.2
                    },
                    'user_id': 'test_user'
                })
            
            assert response.status_code == 200
            data = json.loads(response.data)
            
            assert 'prediction' in data
            assert 'model_version' in data
            assert 'latency_ms' in data
            assert 'timestamp' in data
            assert data['model_version'] == 'v1.0.0'
    
    def test_batch_prediction_endpoint(self, production_server):
        """Test batch prediction endpoint."""
        with production_server.app.test_client() as client:
            # Test batch prediction request
            batch_features = [
                {
                    'M/F': 0, 'Age': 65, 'EDUC': 12, 'SES': 2, 'MMSE': 25,
                    'CDR': 0.5, 'eTIV': 1500, 'nWBV': 0.75, 'ASF': 1.2
                },
                {
                    'M/F': 1, 'Age': 70, 'EDUC': 16, 'SES': 3, 'MMSE': 22,
                    'CDR': 1.0, 'eTIV': 1400, 'nWBV': 0.70, 'ASF': 1.3
                }
            ]
            
            response = client.post('/predict/batch',
                json={
                    'batch_features': batch_features,
                    'user_ids': ['user1', 'user2']
                })
            
            assert response.status_code == 200
            data = json.loads(response.data)
            
            assert 'results' in data
            assert data['batch_size'] == 2
            assert len(data['results']) == 2
            assert 'total_latency_ms' in data
    
    def test_monitoring_endpoint(self, production_server):
        """Test monitoring status endpoint."""
        with production_server.app.test_client() as client:
            response = client.get('/monitoring/status')
            
            assert response.status_code == 200
            data = json.loads(response.data)
            
            assert 'total_predictions' in data
            assert 'status' in data
            assert 'period_hours' in data


class TestIntegration:
    """Integration tests for production MLOps features."""
    
    def test_end_to_end_monitoring_workflow(self):
        """Test complete monitoring workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            
            # Create sample data
            np.random.seed(42)
            baseline_data = pd.DataFrame({
                'feature1': np.random.normal(0, 1, 100),
                'feature2': np.random.normal(0, 1, 100)
            })
            baseline_labels = pd.Series(np.random.choice([0, 1], 100))
            
            # Create monitor
            monitor = ProductionMonitor("integration_test_model")
            monitor.set_baseline_metrics(baseline_data, baseline_labels)
            
            # Simulate production predictions
            current_data = baseline_data.iloc[:50] + np.random.normal(0, 0.1, (50, 2))  # Slight drift
            predictions = np.random.choice([0, 1], 50)
            true_labels = baseline_labels.iloc[:50].values
            
            # Log predictions
            result = monitor.log_prediction_batch(
                features=current_data,
                predictions=predictions,
                true_labels=true_labels,
                latency_ms=120.0
            )
            
            # Verify monitoring worked
            assert result['metrics']['prediction_count'] == 50
            
            # Get summary
            summary = monitor.get_monitoring_summary(hours=1)
            assert summary['total_predictions'] == 50
            assert summary['status'] in ['HEALTHY', 'DEGRADED', 'CRITICAL']
    
    def test_end_to_end_ab_testing_workflow(self):
        """Test complete A/B testing workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            
            # Create A/B manager
            ab_manager = ABTestingManager()
            
            # Create experiment
            config = ABTestConfig(
                experiment_name="integration_test",
                model_a_version="v1.0.0",
                model_b_version="v2.0.0",
                traffic_split=0.5,
                min_samples_per_variant=10
            )
            
            experiment_name = ab_manager.create_experiment(config)
            ab_manager.start_experiment(experiment_name)
            
            # Mock models
            class MockModel:
                def __init__(self, accuracy):
                    self.accuracy = accuracy
                
                def predict(self, X):
                    return np.array([1 if np.random.random() < self.accuracy else 0 for _ in range(len(X))])
            
            models = {
                "v1.0.0": MockModel(0.7),
                "v2.0.0": MockModel(0.8)
            }
            
            # Generate test data
            user_outcomes = []
            for i in range(50):
                result = ab_manager.make_prediction(
                    experiment_name=experiment_name,
                    user_id=f"user_{i}",
                    features={'age': 65 + i, 'mmse': 25},
                    models=models
                )
                
                # Simulate outcome
                true_label = 1 if np.random.random() < 0.75 else 0
                user_outcomes.append((f"user_{i}", true_label))
            
            # Update outcomes
            ab_manager.update_prediction_outcomes(experiment_name, user_outcomes)
            
            # Analyze results
            results = ab_manager.analyze_experiment(experiment_name)
            
            # Verify experiment ran
            assert results.samples_a > 0
            assert results.samples_b > 0
            assert results.experiment_name == experiment_name
            
            # Stop experiment
            final_results = ab_manager.stop_experiment(experiment_name)
            assert final_results.experiment_name == experiment_name


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])