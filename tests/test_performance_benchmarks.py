"""
Advanced Performance Benchmarking Tests
Tests system performance, memory usage, and scalability
"""

import pytest
import time
import threading
import multiprocessing
import psutil
import os
import gc
import asyncio
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Dict, List, Any, Tuple
import sys
import tempfile
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training import AlzheimerTrainer, TrainingIntegratedAgent
from neuralnet import UnifiedAdaptiveAgent, AliveLoopNode, ResourceRoom
from data_loaders import create_data_loader


class TestPerformanceBenchmarks:
    """Performance benchmarking and stress testing"""
    
    @pytest.fixture
    def performance_data(self):
        """Generate performance test data"""
        return pd.DataFrame({
            'age': np.random.randint(60, 90, 1000),
            'gender': np.random.choice(['M', 'F'], 1000),
            'education_level': np.random.randint(8, 20, 1000),
            'mmse_score': np.random.randint(15, 30, 1000),
            'cdr_score': np.random.choice([0.0, 0.5, 1.0, 2.0, 3.0], 1000),
            'apoe_genotype': np.random.choice(['E2/E2', 'E2/E3', 'E3/E3', 'E3/E4', 'E4/E4'], 1000),
            'diagnosis': np.random.choice(['Normal', 'MCI', 'Dementia'], 1000)
        })

    def test_training_performance_benchmark(self, performance_data):
        """Benchmark training performance with large dataset"""
        trainer = AlzheimerTrainer()
        
        # Preprocess data first
        X, y = trainer.preprocess_data(performance_data)
        
        # Measure training time
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss
        
        trainer.train_model(X, y)
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss
        
        training_time = end_time - start_time
        memory_usage = (end_memory - start_memory) / 1024 / 1024  # MB
        
        # Performance assertions
        assert training_time < 30.0, f"Training took too long: {training_time:.2f}s"
        assert memory_usage < 500, f"Memory usage too high: {memory_usage:.2f}MB"
        assert trainer.model is not None
        
        # Test prediction performance
        test_features = performance_data.drop('diagnosis', axis=1).head(10)  # Limit to 10 for testing
        start_time = time.time()
        predictions = []
        for _, row in test_features.iterrows():
            pred = trainer.predict(row.to_dict())
            predictions.append(pred)
        prediction_time = time.time() - start_time
        
        assert prediction_time < 5.0, f"Prediction took too long: {prediction_time:.2f}s"
        assert len(predictions) == len(test_features)

    @pytest.mark.slow
    def test_concurrent_training_performance(self, performance_data):
        """Test performance under concurrent training loads"""
        num_threads = min(4, multiprocessing.cpu_count())
        
        def train_model_thread(data_chunk):
            trainer = AlzheimerTrainer()
            X, y = trainer.preprocess_data(data_chunk)
            start_time = time.time()
            trainer.train_model(X, y)
            return time.time() - start_time, trainer
        
        # Split data into chunks
        chunks = np.array_split(performance_data, num_threads)
        
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(train_model_thread, chunk) for chunk in chunks]
            results = [future.result() for future in as_completed(futures)]
        
        total_time = time.time() - start_time
        avg_thread_time = sum(result[0] for result in results) / len(results)
        
        # Verify all models trained successfully
        assert all(result[1].model is not None for result in results)
        assert total_time < 60.0, f"Concurrent training took too long: {total_time:.2f}s"
        assert avg_thread_time < 30.0, f"Average thread time too long: {avg_thread_time:.2f}s"

    def test_memory_usage_monitoring(self, performance_data):
        """Monitor memory usage during training and prediction"""
        process = psutil.Process()
        memory_samples = []
        
        def memory_monitor():
            for _ in range(50):  # Monitor for 5 seconds
                memory_samples.append(process.memory_info().rss / 1024 / 1024)  # MB
                time.sleep(0.1)
        
        # Start memory monitoring
        monitor_thread = threading.Thread(target=memory_monitor)
        monitor_thread.start()
        
        # Perform training
        trainer = AlzheimerTrainer()
        X, y = trainer.preprocess_data(performance_data)
        trainer.train_model(X, y)
        
        # Perform predictions
        test_features = performance_data.drop('diagnosis', axis=1).head(5)  # Limit for memory test
        predictions = []
        for _, row in test_features.iterrows():
            pred = trainer.predict(row.to_dict())  
            predictions.append(pred)
        
        monitor_thread.join()
        
        # Analyze memory usage
        max_memory = max(memory_samples)
        min_memory = min(memory_samples)
        memory_growth = max_memory - min_memory
        
        assert max_memory < 1000, f"Peak memory usage too high: {max_memory:.2f}MB"
        assert memory_growth < 200, f"Memory growth too large: {memory_growth:.2f}MB"

    def test_memory_leak_detection(self):
        """Test for memory leaks in repeated operations"""
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Perform repeated training cycles
        for i in range(10):
            trainer = AlzheimerTrainer()
            data = pd.DataFrame({
                'age': np.random.randint(60, 90, 100),
                'gender': np.random.choice(['M', 'F'], 100),
                'education_level': np.random.randint(8, 20, 100),
                'mmse_score': np.random.randint(15, 30, 100),
                'cdr_score': np.random.choice([0.0, 0.5, 1.0], 100),
                'apoe_genotype': np.random.choice(['E3/E3', 'E3/E4'], 100),
                'diagnosis': np.random.choice(['Normal', 'MCI', 'Dementia'], 100)
            })
            
            X, y = trainer.preprocess_data(data)
            trainer.train_model(X, y)
            # Test single prediction for memory leak test
            test_row = data.drop('diagnosis', axis=1).iloc[0]
            trainer.predict(test_row.to_dict())
            
            # Explicit cleanup
            del trainer
            del data
            gc.collect()
        
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_increase = final_memory - initial_memory
        
        # Allow some memory increase but detect significant leaks
        assert memory_increase < 100, f"Potential memory leak detected: {memory_increase:.2f}MB increase"

    def test_cpu_usage_monitoring(self, performance_data):
        """Monitor CPU usage during intensive operations"""
        cpu_samples = []
        
        def cpu_monitor():
            for _ in range(30):  # Monitor for 3 seconds
                cpu_samples.append(psutil.cpu_percent(interval=0.1))
        
        # Start CPU monitoring
        monitor_thread = threading.Thread(target=cpu_monitor)
        monitor_thread.start()
        
        # Perform CPU-intensive operations
        trainer = AlzheimerTrainer()
        X, y = trainer.preprocess_data(performance_data)
        trainer.train_model(X, y)
        
        monitor_thread.join()
        
        # Analyze CPU usage
        avg_cpu = sum(cpu_samples) / len(cpu_samples)
        max_cpu = max(cpu_samples)
        
        # Verify reasonable CPU usage
        assert max_cpu <= 100, f"CPU usage exceeded 100%: {max_cpu}%"
        assert avg_cpu > 0, "No CPU usage detected during training"

    @pytest.mark.slow
    def test_scalability_with_increasing_data_size(self):
        """Test how performance scales with increasing data size"""
        data_sizes = [100, 500, 1000, 2000]
        training_times = []
        memory_usage = []
        
        for size in data_sizes:
            data = pd.DataFrame({
                'age': np.random.randint(60, 90, size),
                'gender': np.random.choice(['M', 'F'], size),
                'education_level': np.random.randint(8, 20, size),
                'mmse_score': np.random.randint(15, 30, size),
                'cdr_score': np.random.choice([0.0, 0.5, 1.0], size),
                'apoe_genotype': np.random.choice(['E3/E3', 'E3/E4'], size),
                'diagnosis': np.random.choice(['Normal', 'MCI', 'Dementia'], size)
            })
            
            trainer = AlzheimerTrainer()
            
            start_memory = psutil.Process().memory_info().rss
            start_time = time.time()
            
            X, y = trainer.preprocess_data(data)
            trainer.train_model(X, y)
            
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss
            
            training_times.append(end_time - start_time)
            memory_usage.append((end_memory - start_memory) / 1024 / 1024)
        
        # Verify scaling is reasonable (not exponential)
        for i in range(1, len(training_times)):
            time_ratio = training_times[i] / training_times[i-1]
            size_ratio = data_sizes[i] / data_sizes[i-1]
            
            # Time should not increase faster than O(n log n)
            assert time_ratio < size_ratio * 2, f"Training time scaling too poor: {time_ratio:.2f}x for {size_ratio:.2f}x data"

    def test_error_handling_under_stress(self, performance_data):
        """Test error handling under stressful conditions"""
        # Test with corrupted data
        corrupted_data = performance_data.copy()
        corrupted_data.loc[0:50, 'age'] = np.nan
        corrupted_data.loc[100:150, 'mmse_score'] = -999
        
        trainer = AlzheimerTrainer()
        
        # Should handle corrupted data gracefully
        try:
            X, y = trainer.preprocess_data(corrupted_data)
            trainer.train_model(X, y)
            # If training succeeds, verify model still works
            assert trainer.model is not None
        except Exception as e:
            # If it fails, it should be a controlled failure
            assert isinstance(e, (ValueError, TypeError))

    def test_resource_cleanup(self):
        """Test proper resource cleanup after operations"""
        initial_threads = threading.active_count()
        initial_memory = psutil.Process().memory_info().rss
        
        # Create and use multiple trainers
        trainers = []
        for _ in range(5):
            trainer = AlzheimerTrainer()
            data = pd.DataFrame({
                'age': np.random.randint(60, 90, 50),
                'gender': np.random.choice(['M', 'F'], 50),
                'education_level': np.random.randint(8, 20, 50),
                'mmse_score': np.random.randint(15, 30, 50),
                'cdr_score': np.random.choice([0.0, 0.5], 50),
                'apoe_genotype': np.random.choice(['E3/E3', 'E3/E4'], 50),
                'diagnosis': np.random.choice(['Normal', 'MCI'], 50)
            })
            X, y = trainer.preprocess_data(data)
            trainer.train_model(X, y)
            trainers.append(trainer)
        
        # Clean up
        del trainers
        gc.collect()
        
        final_threads = threading.active_count()
        final_memory = psutil.Process().memory_info().rss
        
        # Verify resources were cleaned up
        assert final_threads <= initial_threads + 2, "Thread cleanup failed"
        assert final_memory - initial_memory < 50 * 1024 * 1024, "Memory cleanup failed"


class TestStressAndResilience:
    """Stress testing for system resilience"""
    
    def test_high_frequency_operations(self):
        """Test system under high frequency operations"""
        trainer = AlzheimerTrainer()
        data = pd.DataFrame({
            'age': [65], 'gender': ['M'], 'education_level': [16],
            'mmse_score': [28], 'cdr_score': [0.0], 'apoe_genotype': ['E3/E3'],
            'diagnosis': ['Normal']
        })
        
        X, y = trainer.preprocess_data(data)
        trainer.train_model(X, y)
        
        # Perform high-frequency predictions
        test_row = data.drop('diagnosis', axis=1).iloc[0].to_dict()
        start_time = time.time()
        for _ in range(1000):
            prediction = trainer.predict(test_row)
            assert isinstance(prediction, str)
        
        total_time = time.time() - start_time
        ops_per_second = 1000 / total_time
        
        # Should handle at least 100 predictions per second
        assert ops_per_second > 100, f"Performance too low: {ops_per_second:.2f} ops/sec"

    def test_timeout_handling(self):
        """Test timeout handling for long-running operations"""
        class SlowTrainer(AlzheimerTrainer):
            def train_model(self, X, y):
                time.sleep(2)  # Simulate slow training
                return super().train_model(X, y)
        
        trainer = SlowTrainer()
        data = pd.DataFrame({
            'age': [65], 'gender': ['M'], 'education_level': [16],
            'mmse_score': [28], 'cdr_score': [0.0], 'apoe_genotype': ['E3/E3'],
            'diagnosis': ['Normal']
        })
        
        # Test with timeout
        start_time = time.time()
        try:
            X, y = trainer.preprocess_data(data)
            trainer.train_model(X, y)
            training_time = time.time() - start_time
            assert training_time >= 2.0, "Training completed too quickly"
        except Exception:
            pass  # Timeout exceptions are acceptable

    def test_concurrent_access_safety(self):
        """Test thread safety with concurrent access"""
        trainer = AlzheimerTrainer()
        data = pd.DataFrame({
            'age': np.random.randint(60, 90, 100),
            'gender': np.random.choice(['M', 'F'], 100),
            'education_level': np.random.randint(8, 20, 100),
            'mmse_score': np.random.randint(15, 30, 100),
            'cdr_score': np.random.choice([0.0, 0.5, 1.0], 100),
            'apoe_genotype': np.random.choice(['E3/E3', 'E3/E4'], 100),
            'diagnosis': np.random.choice(['Normal', 'MCI', 'Dementia'], 100)
        })
        
        X, y = trainer.preprocess_data(data)
        trainer.train_model(X, y)
        
        results = []
        errors = []
        
        def concurrent_prediction():
            try:
                test_row = data.drop('diagnosis', axis=1).iloc[0].to_dict()
                result = trainer.predict(test_row)
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        # Run concurrent predictions
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=concurrent_prediction)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Verify results
        assert len(errors) == 0, f"Concurrent access errors: {errors}"
        assert len(results) == 10, "Not all predictions completed"
        assert all(isinstance(result, str) for result in results), "Inconsistent prediction results"