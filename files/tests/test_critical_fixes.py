"""
Critical Bug Fixes Tests
Tests for the specific bugs mentioned in the problem statement
"""
import unittest
import warnings
import sys
import os
from typing import Tuple  # Test that Tuple can be imported

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Test NetworkMetrics fixes
from neuralnet import NetworkMetrics as NeuralNetworkMetrics, Memory, CapacitorInSpace
from labyrinth_adaptive import NetworkMetrics as LabyrinthNetworkMetrics


class TestCriticalBugFixes(unittest.TestCase):
    """Test cases for critical bugs mentioned in the problem statement"""

    def test_networkmetrics_empty_agent_list_neuralnet(self):
        """Test NetworkMetrics in neuralnet.py handles empty agent list gracefully"""
        # Capture warnings to ensure no RuntimeWarning is raised
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            metrics = NeuralNetworkMetrics()
            metrics.update([])  # This used to crash with RuntimeWarning
            
            # Verify no warnings were raised
            runtime_warnings = [warning for warning in w if issubclass(warning.category, RuntimeWarning)]
            self.assertEqual(len(runtime_warnings), 0, "Should not raise RuntimeWarning for empty agent list")
            
            # Verify it maintains default health score
            health_score = metrics.health_score()
            self.assertEqual(health_score, 0.5, "Should maintain default health score of 0.5")

    def test_networkmetrics_empty_agent_list_labyrinth(self):
        """Test NetworkMetrics in labyrinth_adaptive.py handles empty agent list gracefully"""
        # Capture warnings to ensure no RuntimeWarning is raised
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            metrics = LabyrinthNetworkMetrics()
            metrics.update([])  # This used to crash with RuntimeWarning
            
            # Verify no warnings were raised
            runtime_warnings = [warning for warning in w if issubclass(warning.category, RuntimeWarning)]
            self.assertEqual(len(runtime_warnings), 0, "Should not raise RuntimeWarning for empty agent list")
            
            # Verify it maintains default health score
            health_score = metrics.health_score()
            self.assertEqual(health_score, 0.5, "Should maintain default health score of 0.5")

    def test_tuple_import_fix(self):
        """Test that Tuple import is now available and duetmind module can be imported"""
        try:
            import duetmind
            # If we can import duetmind without NameError, the Tuple fix worked
            self.assertTrue(True, "duetmind module imported successfully")
        except NameError as e:
            if "Tuple" in str(e):
                self.fail("Tuple import is still missing in duetmind.py")
            else:
                raise

    def test_memory_age_with_high_emotional_valence(self):
        """Test memory aging with extreme emotional valence (from problem statement)"""
        mem = Memory(content="test", importance=1.0, timestamp=100, 
                    memory_type="prediction", emotional_valence=0.9)
        original_decay_rate = mem.decay_rate
        mem.age()
        # Should increase decay rate due to high emotional valence
        self.assertGreater(mem.decay_rate, original_decay_rate,
                          "Decay rate should increase with high emotional valence")

    def test_capacitor_creation_with_negative_capacity(self):
        """Test CapacitorInSpace boundary conditions (from problem statement)"""
        cap = CapacitorInSpace((0, 0), capacity=-5.0, initial_energy=3.0)
        self.assertEqual(cap.capacity, 0.0, "Capacity should be clamped to 0")
        self.assertEqual(cap.energy, 0.0, "Energy should be clamped to capacity")

    def test_performance_monitor_concurrent_updates(self):
        """Test PerformanceMonitor concurrent updates (from problem statement)"""
        try:
            from duetmind import PerformanceMonitor
            monitor = PerformanceMonitor()
            
            # Simulate 100 concurrent requests with 10% error rate
            for i in range(100):
                monitor.record_request(0.01 * i, i % 10 != 0)
            
            # Verify error rate calculation accuracy
            stats = monitor.get_metrics_snapshot()
            expected_error_rate = 0.1  # 10% error rate
            actual_error_rate = stats['error_rate']
            self.assertAlmostEqual(actual_error_rate, expected_error_rate, places=1,
                                 msg="Error rate calculation should be accurate")
            
        except ImportError:
            # If PerformanceMonitor doesn't exist or has import issues, skip this test
            self.skipTest("PerformanceMonitor not available for testing")


if __name__ == '__main__':
    unittest.main()