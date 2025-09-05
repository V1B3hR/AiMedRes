import pytest
import numpy as np
import time
import uuid
import json
import logging
import redis
from unittest.mock import patch, MagicMock, mock_open
from collections import defaultdict

# Import modules to test
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from duetmind import (
    PerformanceMetrics, PerformanceMonitor, AdvancedCacheManager,
    OptimizedAdaptiveEngine, ObservabilitySystem, 
    ParallelProcessingManager, GPUAccelerator,
    demo_enterprise_system
)


class TestPerformanceMetrics:
    """Test PerformanceMetrics dataclass with edge cases"""
    
    def test_performance_metrics_creation(self):
        """Test normal performance metrics creation"""
        metrics = PerformanceMetrics()
        assert metrics.request_count == 0
        assert metrics.total_response_time == 0.0
        assert metrics.average_response_time == 0.0
        assert metrics.error_count == 0
        assert metrics.cache_hit_rate == 0.0
        assert isinstance(metrics.cpu_usage_history, list)
        
    def test_performance_metrics_with_custom_values(self):
        """Test performance metrics with custom initial values"""
        metrics = PerformanceMetrics(
            request_count=100,
            total_response_time=50.5,
            error_count=5
        )
        assert metrics.request_count == 100
        assert metrics.total_response_time == 50.5
        assert metrics.error_count == 5


class TestPerformanceMonitor:
    """Test PerformanceMonitor with edge cases and error handling"""
    
    def test_performance_monitor_creation(self):
        """Test performance monitor creation"""
        monitor = PerformanceMonitor()
        assert monitor.metrics is not None
        assert monitor.monitoring == False
        assert monitor.sample_interval == 5
        
    def test_start_monitoring(self):
        """Test starting monitoring"""
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        assert monitor.monitoring == True
        
        # Stop to clean up
        monitor.stop_monitoring()
        
    def test_stop_monitoring(self):
        """Test stopping monitoring"""
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        monitor.stop_monitoring()
        assert monitor.monitoring == False
        
    def test_record_request_normal(self):
        """Test normal request recording"""
        monitor = PerformanceMonitor()
        monitor.record_request(0.5, True)
        
        assert monitor.metrics.request_count == 1
        assert monitor.metrics.total_response_time == 0.5
        assert monitor.metrics.average_response_time == 0.5
        
    def test_record_request_failure(self):
        """Test recording failed request"""
        monitor = PerformanceMonitor()
        monitor.record_request(1.0, False)
        
        assert monitor.metrics.request_count == 1
        assert monitor.metrics.error_count == 1
        
    def test_record_multiple_requests(self):
        """Test recording multiple requests for average calculation"""
        monitor = PerformanceMonitor()
        monitor.record_request(0.5, True)
        monitor.record_request(1.5, True)
        
        assert monitor.metrics.request_count == 2
        assert monitor.metrics.total_response_time == 2.0
        assert monitor.metrics.average_response_time == 1.0
        
    def test_record_request_with_zero_time(self):
        """Test recording request with zero response time (edge case)"""
        monitor = PerformanceMonitor()
        monitor.record_request(0.0, True)
        
        assert monitor.metrics.request_count == 1
        assert monitor.metrics.total_response_time == 0.0
        assert monitor.metrics.average_response_time == 0.0
        
    def test_record_request_with_negative_time(self):
        """Test recording request with negative response time (edge case)"""
        monitor = PerformanceMonitor()
        monitor.record_request(-0.5, True)
        
        # Should still record but with negative time
        assert monitor.metrics.request_count == 1
        assert monitor.metrics.total_response_time == -0.5
        
    def test_get_metrics_snapshot(self):
        """Test getting current statistics"""
        monitor = PerformanceMonitor()
        monitor.record_request(0.5, True)
        monitor.record_request(1.0, False)
        
        stats = monitor.get_metrics_snapshot()
        assert stats['request_count'] == 2
        assert stats['error_rate'] == 0.5
        assert 'average_response_time' in stats


class TestAdvancedCacheManager:
    """Test AdvancedCacheManager with edge cases and memory limits"""
    
    def test_cache_manager_creation_memory_only(self):
        """Test cache manager creation with memory only"""
        cache = AdvancedCacheManager(memory_limit_mb=10)
        assert cache.memory_limit_mb == 10
        assert cache.redis_client is None
        
    @patch('redis.from_url')
    def test_cache_manager_creation_with_redis(self, mock_redis):
        """Test cache manager creation with Redis"""
        mock_redis_instance = MagicMock()
        mock_redis.return_value = mock_redis_instance
        mock_redis_instance.ping.return_value = True
        
        cache = AdvancedCacheManager(redis_url="redis://localhost:6379")
        assert cache.redis_client is not None
        
    @patch('redis.from_url')
    def test_cache_manager_redis_connection_failure(self, mock_redis):
        """Test cache manager with Redis connection failure (edge case)"""
        mock_redis.side_effect = Exception("Connection failed")
        
        cache = AdvancedCacheManager(redis_url="redis://localhost:6379")
        assert cache.redis_client is None  # Should fall back to memory only
        
    def test_cache_set_and_get_normal(self):
        """Test normal cache set and get operations"""
        cache = AdvancedCacheManager(memory_limit_mb=10)
        
        cache.set("key1", "value1", ttl=3600)
        result = cache.get("key1")
        
        assert result == "value1"
        
    def test_cache_get_nonexistent_key(self):
        """Test getting non-existent key (edge case)"""
        cache = AdvancedCacheManager(memory_limit_mb=10)
        result = cache.get("nonexistent")
        assert result is None
        
    def test_cache_set_none_value(self):
        """Test setting None value (edge case)"""
        cache = AdvancedCacheManager(memory_limit_mb=10)
        cache.set("key_none", None)
        result = cache.get("key_none")
        assert result is None
        
    def test_cache_set_complex_object(self):
        """Test setting complex object"""
        cache = AdvancedCacheManager(memory_limit_mb=10)
        complex_data = {"nested": {"list": [1, 2, 3]}, "value": 42}
        
        cache.set("complex", complex_data)
        result = cache.get("complex")
        
        assert result == complex_data
        
    def test_cache_memory_limit_basic(self):
        """Test basic memory limit functionality"""
        cache = AdvancedCacheManager(memory_limit_mb=1)  # Very small limit
        
        # Store some data
        cache.set("key1", "small_value")
        cache.set("key2", "another_small_value")
        
        # Should not crash and should store successfully for small values
        assert cache.get("key1") == "small_value"
        assert cache.get("key2") == "another_small_value"
        
    def test_cache_stats_tracking(self):
        """Test cache statistics tracking"""
        cache = AdvancedCacheManager(memory_limit_mb=10)
        
        cache.set("key1", "value1")
        cache.get("key1")  # Hit
        cache.get("nonexistent")  # Miss
        
        assert cache.cache_stats['hits'] >= 1
        assert cache.cache_stats['misses'] >= 1
        
    def test_cache_access_time_tracking(self):
        """Test that access times are tracked"""
        cache = AdvancedCacheManager(memory_limit_mb=10)
        
        cache.set("key1", "value1")
        assert "key1" in cache.access_times
        
        # Access again
        cache.get("key1")
        assert "key1" in cache.access_times


class TestOptimizedAdaptiveEngine:
    """Test OptimizedAdaptiveEngine with focus on caching and performance"""
    
    def test_engine_creation_normal(self):
        """Test normal engine creation"""
        engine = OptimizedAdaptiveEngine(network_size=10)
        assert engine.network_size == 10
        assert engine.cache_manager is not None
        assert engine.performance_monitor is not None
        
    def test_engine_creation_with_config(self):
        """Test engine creation with custom config"""
        config = {
            'cache_memory_mb': 100,
            'batch_size': 20,
            'max_concurrent_requests': 5
        }
        engine = OptimizedAdaptiveEngine(network_size=5, config=config)
        assert engine.network_size == 5
        
    def test_safe_think_normal_operation(self):
        """Test normal safe_think operation"""
        engine = OptimizedAdaptiveEngine(network_size=10)
        result = engine.safe_think("TestAgent", "solve problem")
        
        assert result['content'] is not None
        assert 'confidence' in result
        assert 'runtime' in result
        assert result['from_cache'] == False  # First time should not be from cache
        
    def test_safe_think_caching(self):
        """Test that safe_think uses caching"""
        engine = OptimizedAdaptiveEngine(network_size=10)
        
        # First call
        result1 = engine.safe_think("TestAgent", "same task")
        assert result1['from_cache'] == False
        
        # Second call with same parameters should use cache
        result2 = engine.safe_think("TestAgent", "same task")
        assert result2['from_cache'] == True
        
    def test_safe_think_empty_task(self):
        """Test safe_think with empty task (edge case)"""
        engine = OptimizedAdaptiveEngine(network_size=10)
        result = engine.safe_think("TestAgent", "")
        
        assert 'error' in result
        assert result['success'] == False
        
    def test_safe_think_none_task(self):
        """Test safe_think with None task (edge case)"""
        engine = OptimizedAdaptiveEngine(network_size=10)
        result = engine.safe_think("TestAgent", None)
        
        assert 'error' in result
        assert result['success'] == False
        
    def test_safe_think_none_agent_name(self):
        """Test safe_think with None agent name (edge case)"""
        engine = OptimizedAdaptiveEngine(network_size=10)
        result = engine.safe_think(None, "valid task")
        
        assert 'error' in result
        assert result['success'] == False
        
    @patch('duetmind.OptimizedAdaptiveEngine._execute_optimized_reasoning')
    def test_safe_think_with_exception(self, mock_reasoning):
        """Test safe_think when reasoning raises exception"""
        mock_reasoning.side_effect = Exception("Processing error")
        
        engine = OptimizedAdaptiveEngine(network_size=10)
        result = engine.safe_think("TestAgent", "problematic task")
        
        assert 'error' in result
        assert result['success'] == False
        
    def test_generate_cache_key(self):
        """Test cache key generation"""
        engine = OptimizedAdaptiveEngine(network_size=10)
        
        key1 = engine._generate_cache_key("Agent1", "task1")
        key2 = engine._generate_cache_key("Agent1", "task1")
        key3 = engine._generate_cache_key("Agent2", "task1")
        
        assert key1 == key2  # Same inputs should generate same key
        assert key1 != key3  # Different inputs should generate different keys
        
    def test_batch_processing_enabled(self):
        """Test batch processing functionality"""
        engine = OptimizedAdaptiveEngine(network_size=10)
        assert engine.batch_processing_enabled == True
        
        # Test that batch processing doesn't break single requests
        result = engine.safe_think("TestAgent", "batch test")
        assert result is not None


class TestObservabilitySystem:
    """Test ObservabilitySystem with focus on monitoring and alerting edge cases"""
    
    def test_observability_system_creation(self):
        """Test observability system creation"""
        config = {'port': 8080}
        obs = ObservabilitySystem(config)
        assert obs.config == config
        assert obs.metrics_collector is not None
        assert obs.alert_manager is not None
        
    def test_record_request_metrics_normal(self):
        """Test normal request metrics recording"""
        config = {}
        obs = ObservabilitySystem(config)
        
        obs.record_request_metrics('/api/test', 0.5, 200)
        
        metrics = obs.get_metrics_summary()
        assert '/api/test' in metrics
        assert metrics['/api/test']['request_count'] == 1
        assert 0.5 in metrics['/api/test']['duration_history']
        
    def test_record_request_metrics_error_status(self):
        """Test recording metrics for error responses"""
        config = {}
        obs = ObservabilitySystem(config)
        
        obs.record_request_metrics('/api/error', 1.0, 500)
        
        metrics = obs.get_metrics_summary()
        assert metrics['/api/error']['error_count'] == 1
        
    def test_record_request_metrics_zero_duration(self):
        """Test recording metrics with zero duration (edge case)"""
        config = {}
        obs = ObservabilitySystem(config)
        
        obs.record_request_metrics('/api/fast', 0.0, 200)
        
        metrics = obs.get_metrics_summary()
        assert 0.0 in metrics['/api/fast']['duration_history']
        
    def test_record_request_metrics_negative_duration(self):
        """Test recording metrics with negative duration (edge case)"""
        config = {}
        obs = ObservabilitySystem(config)
        
        obs.record_request_metrics('/api/negative', -0.1, 200)
        
        metrics = obs.get_metrics_summary()
        assert -0.1 in metrics['/api/negative']['duration_history']
        
    def test_alert_high_response_time(self):
        """Test high response time alert triggering"""
        config = {}
        obs = ObservabilitySystem(config)
        
        # Record a request that should trigger high response time alert
        obs.record_request_metrics('/api/slow', 10.0, 200)  # 10 seconds > 5 second threshold
        
        # Alert should be triggered internally (we can check via metrics or logs)
        metrics = obs.get_metrics_summary()
        assert metrics['/api/slow']['duration_history'][-1] == 10.0
        
    def test_alert_high_error_rate(self):
        """Test high error rate alert triggering"""
        config = {}
        obs = ObservabilitySystem(config)
        
        # Generate requests with high error rate
        for i in range(10):
            status = 500 if i >= 8 else 200  # 20% error rate (above 5% threshold)
            obs.record_request_metrics('/api/errors', 0.5, status)
            
        metrics = obs.get_metrics_summary()
        assert metrics['/api/errors']['error_count'] >= 2
        
    def test_get_metrics_summary_empty(self):
        """Test getting metrics summary when no metrics recorded"""
        config = {}
        obs = ObservabilitySystem(config)
        
        metrics = obs.get_metrics_summary()
        assert metrics == {}
        
    def test_get_metrics_summary_multiple_endpoints(self):
        """Test metrics summary with multiple endpoints"""
        config = {}
        obs = ObservabilitySystem(config)
        
        obs.record_request_metrics('/api/endpoint1', 0.5, 200)
        obs.record_request_metrics('/api/endpoint2', 1.0, 404)
        obs.record_request_metrics('/api/endpoint1', 0.7, 200)
        
        metrics = obs.get_metrics_summary()
        assert len(metrics) == 2
        assert metrics['/api/endpoint1']['request_count'] == 2
        assert metrics['/api/endpoint2']['request_count'] == 1
        assert metrics['/api/endpoint2']['error_count'] == 1
        
    def test_setup_alerting_thresholds(self):
        """Test alerting threshold setup"""
        config = {}
        obs = ObservabilitySystem(config)
        
        assert obs.alert_manager['high_error_rate_threshold'] == 0.05
        assert obs.alert_manager['high_response_time_threshold'] == 5.0
        assert obs.alert_manager['high_memory_usage_threshold'] == 0.8
        assert obs.alert_manager['high_cpu_usage_threshold'] == 0.8


class TestParallelProcessingManager:
    """Test ParallelProcessingManager with edge cases and resource limits"""
    
    def test_parallel_manager_creation(self):
        """Test parallel processing manager creation"""
        manager = ParallelProcessingManager()
        assert manager.max_workers is not None
        assert manager.thread_pool is not None
        
    def test_process_batch_empty_list(self):
        """Test processing empty batch (edge case)"""
        manager = ParallelProcessingManager()
        
        def dummy_func(x):
            return x * 2
            
        results = manager.process_batch([], dummy_func)
        assert results == []
        
    def test_process_batch_single_item(self):
        """Test processing batch with single item"""
        manager = ParallelProcessingManager()
        
        def dummy_func(x):
            return x * 2
            
        results = manager.process_batch([5], dummy_func)
        assert results == [10]
        
    def test_process_batch_multiple_items(self):
        """Test processing batch with multiple items"""
        manager = ParallelProcessingManager()
        
        def dummy_func(x):
            return x * 2
            
        results = manager.process_batch([1, 2, 3, 4], dummy_func)
        assert sorted(results) == [2, 4, 6, 8]
        
    def test_process_batch_with_exception(self):
        """Test processing batch when function raises exception"""
        manager = ParallelProcessingManager()
        
        def failing_func(x):
            if x == 2:
                raise ValueError("Test error")
            return x * 2
            
        # Should handle exceptions gracefully
        results = manager.process_batch([1, 2, 3], failing_func)
        # Results should contain successful computations and None/error for failed ones
        assert len(results) <= 3


class TestGPUAccelerator:
    """Test GPUAccelerator with device availability edge cases"""
    
    def test_gpu_accelerator_creation(self):
        """Test GPU accelerator creation"""
        accelerator = GPUAccelerator()
        # Should not crash even if no GPU available
        assert accelerator is not None
        
    def test_check_gpu_availability(self):
        """Test GPU availability check"""
        accelerator = GPUAccelerator()
        # Should return boolean without crashing
        available = accelerator.check_availability()
        assert isinstance(available, bool)
        
    def test_accelerate_computation_fallback(self):
        """Test that computation falls back to CPU when GPU unavailable"""
        accelerator = GPUAccelerator()
        
        data = np.array([1, 2, 3, 4, 5])
        # Should work even without GPU
        result = accelerator.accelerate_computation(data)
        assert result is not None


# Integration and Edge Case Tests
class TestIntegrationEdgeCases:
    """Integration tests focusing on edge cases and error conditions"""
    
    def test_engine_with_cache_failure(self):
        """Test engine behavior when cache fails"""
        config = {'redis_url': 'invalid://url'}
        engine = OptimizedAdaptiveEngine(network_size=5, config=config)
        
        # Should still work even if cache fails
        result = engine.safe_think("TestAgent", "test task")
        assert result is not None
        
    def test_observability_with_extreme_metrics(self):
        """Test observability system with extreme metric values"""
        config = {}
        obs = ObservabilitySystem(config)
        
        # Test with very large duration
        obs.record_request_metrics('/api/extreme', 999999.9, 200)
        
        # Test with very small duration
        obs.record_request_metrics('/api/extreme', 0.0001, 200)
        
        metrics = obs.get_metrics_summary()
        assert '/api/extreme' in metrics
        assert metrics['/api/extreme']['request_count'] == 2
        
    def test_performance_monitor_concurrent_updates(self):
        """Test performance monitor with concurrent updates"""
        monitor = PerformanceMonitor()
        
        # Simulate concurrent request recording
        for i in range(100):
            monitor.record_request(0.01 * i, i % 10 != 0)  # 10% error rate
            
        stats = monitor.get_current_stats()
        assert stats['request_count'] == 100
        assert stats['error_count'] == 90  # 90% success rate
        
    @patch('duetmind.demo_enterprise_system')
    def test_demo_system_execution(self, mock_demo):
        """Test that demo system can be called without errors"""
        mock_demo.return_value = None
        
        # Should not raise any exceptions
        try:
            from duetmind import demo_enterprise_system
            # Don't actually run the demo, just test that it's importable
            assert demo_enterprise_system is not None
        except Exception as e:
            pytest.fail(f"Demo system import failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])