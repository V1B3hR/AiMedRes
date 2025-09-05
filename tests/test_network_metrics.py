"""
Comprehensive tests for NetworkMetrics monitoring system.
"""
import pytest
import numpy as np
from collections import deque

from labyrinth_adaptive import NetworkMetrics


class TestNetworkMetrics:
    """Test suite for NetworkMetrics functionality."""
    
    def test_network_metrics_initialization(self, network_metrics):
        """Test proper NetworkMetrics initialization."""
        assert isinstance(network_metrics.energy_history, deque)
        assert isinstance(network_metrics.confusion_history, deque)
        assert network_metrics.energy_history.maxlen == 1000
        assert network_metrics.confusion_history.maxlen == 1000
        assert len(network_metrics.energy_history) == 0
        assert len(network_metrics.confusion_history) == 0
        assert network_metrics.agent_statuses == []
    
    def test_update_with_agents(self, network_metrics, sample_agents):
        """Test updating metrics with agent data."""
        # Set some energy and confusion values
        sample_agents[0].alive_node.energy = 15.0
        sample_agents[1].alive_node.energy = 12.0
        sample_agents[2].alive_node.energy = 10.0
        
        sample_agents[0].confusion_level = 0.2
        sample_agents[1].confusion_level = 0.5
        sample_agents[2].confusion_level = 0.3
        
        network_metrics.update(sample_agents)
        
        # Check energy history
        assert len(network_metrics.energy_history) == 1
        expected_total_energy = 15.0 + 12.0 + 10.0
        assert network_metrics.energy_history[0] == expected_total_energy
        
        # Check confusion history
        assert len(network_metrics.confusion_history) == 1
        expected_avg_confusion = (0.2 + 0.5 + 0.3) / 3
        assert abs(network_metrics.confusion_history[0] - expected_avg_confusion) < 1e-10
        
        # Check agent statuses
        assert len(network_metrics.agent_statuses) == 3
    
    def test_health_score_empty_history(self, network_metrics):
        """Test health score calculation with empty history."""
        score = network_metrics.health_score()
        assert score == 0.5  # Default score when no history
    
    def test_health_score_calculation(self, network_metrics, sample_agents):
        """Test health score calculation with data."""
        # Set up test data
        sample_agents[0].alive_node.energy = 50.0  # High energy
        sample_agents[1].alive_node.energy = 30.0
        sample_agents[2].alive_node.energy = 20.0
        
        sample_agents[0].confusion_level = 0.1  # Low confusion
        sample_agents[1].confusion_level = 0.2
        sample_agents[2].confusion_level = 0.3
        
        network_metrics.update(sample_agents)
        score = network_metrics.health_score()
        
        # Score should be between 0 and 1
        assert 0 <= score <= 1
        
        # With high energy and low confusion, score should be high
        assert score > 0.6
    
    @pytest.mark.parametrize("energies,confusions,expected_range", [
        ([100, 100, 100], [0.0, 0.0, 0.0], (0.9, 1.0)),  # High energy, no confusion
        ([10, 10, 10], [1.0, 1.0, 1.0], (0.0, 0.3)),      # Low energy, high confusion
        ([50, 50, 50], [0.5, 0.5, 0.5], (0.4, 0.7)),      # Medium energy and confusion
        ([0, 0, 0], [0.0, 0.0, 0.0], (0.25, 0.75))        # No energy, no confusion
    ])
    def test_health_score_scenarios(self, network_metrics, sample_agents, 
                                  energies, confusions, expected_range):
        """Test health score under various scenarios."""
        # Set agent states
        for i, (energy, confusion) in enumerate(zip(energies, confusions)):
            sample_agents[i].alive_node.energy = energy
            sample_agents[i].confusion_level = confusion
        
        network_metrics.update(sample_agents)
        score = network_metrics.health_score()
        
        min_expected, max_expected = expected_range
        assert min_expected <= score <= max_expected, \
            f"Score {score} not in expected range {expected_range}"
    
    def test_multiple_updates(self, network_metrics, sample_agents):
        """Test multiple metric updates."""
        updates = 10
        
        for i in range(updates):
            # Gradually change energy and confusion
            for j, agent in enumerate(sample_agents):
                agent.alive_node.energy = 20.0 - i  # Decreasing energy
                agent.confusion_level = i * 0.1     # Increasing confusion
            
            network_metrics.update(sample_agents)
        
        assert len(network_metrics.energy_history) == updates
        assert len(network_metrics.confusion_history) == updates
        
        # Energy should be decreasing
        energies = list(network_metrics.energy_history)
        assert energies[0] > energies[-1]
        
        # Confusion should be increasing
        confusions = list(network_metrics.confusion_history)
        assert confusions[0] < confusions[-1]
    
    def test_history_maxlen_enforcement(self, network_metrics, sample_agents):
        """Test that history respects maxlen limit."""
        # Update more than maxlen times
        for i in range(1200):  # More than maxlen of 1000
            sample_agents[0].alive_node.energy = i
            sample_agents[0].confusion_level = 0.1
            network_metrics.update(sample_agents)
        
        # Should be limited to maxlen
        assert len(network_metrics.energy_history) == 1000
        assert len(network_metrics.confusion_history) == 1000
        
        # Should contain the most recent 1000 entries
        assert network_metrics.energy_history[-1] == 1199.0  # Last energy set
    
    def test_agent_statuses_update(self, network_metrics, sample_agents):
        """Test that agent statuses are properly captured."""
        # Set different statuses
        sample_agents[0].status = "active"
        sample_agents[1].status = "escaped"
        sample_agents[2].status = "in_resource_room"
        
        network_metrics.update(sample_agents)
        
        expected_statuses = ["active", "escaped", "in_resource_room"]
        assert network_metrics.agent_statuses == expected_statuses
    
    def test_zero_agents_update(self, network_metrics):
        """Test updating with zero agents."""
        network_metrics.update([])
        
        # Should handle empty agent list gracefully
        assert len(network_metrics.energy_history) == 1
        assert len(network_metrics.confusion_history) == 1
        assert network_metrics.energy_history[0] == 0
        assert network_metrics.confusion_history[0] == 0  # np.mean of empty array is 0
        assert network_metrics.agent_statuses == []
    
    def test_single_agent_update(self, network_metrics, sample_agent):
        """Test updating with a single agent."""
        sample_agent.alive_node.energy = 25.0
        sample_agent.confusion_level = 0.4
        
        network_metrics.update([sample_agent])
        
        assert network_metrics.energy_history[0] == 25.0
        assert network_metrics.confusion_history[0] == 0.4
        assert len(network_metrics.agent_statuses) == 1
    
    @pytest.mark.performance
    def test_large_scale_monitoring(self, network_metrics, sample_agents, performance_timer):
        """Test performance with large numbers of agents and updates."""
        # Create many agents
        many_agents = sample_agents * 100  # 300 agents
        
        performance_timer.start()
        
        # Perform many updates
        for i in range(100):
            for agent in many_agents:
                agent.alive_node.energy = 10.0 + i
                agent.confusion_level = 0.01 * i
            network_metrics.update(many_agents)
        
        elapsed = performance_timer.stop()
        
        # Should complete quickly even with many agents
        assert elapsed < 2.0, f"Monitoring took too long: {elapsed}s"
        assert len(network_metrics.energy_history) == 100
    
    def test_health_score_precision(self, network_metrics, sample_agents):
        """Test health score precision and rounding."""
        # Set specific values to test rounding
        sample_agents[0].alive_node.energy = 33.333
        sample_agents[1].alive_node.energy = 66.666
        sample_agents[2].alive_node.energy = 100.001
        
        sample_agents[0].confusion_level = 0.123456
        sample_agents[1].confusion_level = 0.654321
        sample_agents[2].confusion_level = 0.999999
        
        network_metrics.update(sample_agents)
        score = network_metrics.health_score()
        
        # Should be rounded to 3 decimal places
        assert isinstance(score, float)
        # Check that it's reasonably rounded (within expected precision)
        assert abs(score - round(score, 3)) < 1e-10
    
    def test_health_score_edge_cases(self, network_metrics, sample_agents):
        """Test health score calculation edge cases."""
        # Test with very high energy
        for agent in sample_agents:
            agent.alive_node.energy = 1000.0  # Very high
            agent.confusion_level = 0.0
        
        network_metrics.update(sample_agents)
        score = network_metrics.health_score()
        assert score <= 1.0  # Should be capped at 1.0
        
        # Test with very high confusion
        for agent in sample_agents:
            agent.alive_node.energy = 0.0
            agent.confusion_level = 10.0  # Very high
        
        network_metrics.update(sample_agents)
        score = network_metrics.health_score()
        assert score >= 0.0  # Should not go below 0.0
    
    @pytest.mark.integration
    def test_metrics_throughout_simulation(self, network_metrics, sample_agents):
        """Test metrics tracking throughout a full simulation."""
        simulation_steps = 50
        health_scores = []
        
        for step in range(simulation_steps):
            # Simulate changing conditions
            for i, agent in enumerate(sample_agents):
                # Energy decreases over time
                agent.alive_node.energy = max(0, 20.0 - step * 0.2 + i * 2)
                # Confusion increases then stabilizes
                agent.confusion_level = min(1.0, step * 0.02)
                
                # Simulate reasoning causing changes
                agent.reason(f"Simulation step {step}")
            
            network_metrics.update(sample_agents)
            health_scores.append(network_metrics.health_score())
        
        # Should have tracked all steps
        assert len(network_metrics.energy_history) == simulation_steps
        assert len(health_scores) == simulation_steps
        
        # Health should generally decline as energy decreases and confusion increases
        early_avg = np.mean(health_scores[:10])
        late_avg = np.mean(health_scores[-10:])
        assert late_avg <= early_avg + 0.1  # Allow some tolerance
    
    def test_concurrent_access_safety(self, network_metrics, sample_agents):
        """Test thread-safety considerations."""
        import threading
        import time
        
        def update_metrics():
            for i in range(10):
                network_metrics.update(sample_agents)
                time.sleep(0.001)  # Small delay
        
        # Run updates from multiple threads
        threads = [threading.Thread(target=update_metrics) for _ in range(3)]
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Should have completed without errors
        # Note: This is a basic test - real thread safety would need more sophisticated testing
        assert len(network_metrics.energy_history) == 30  # 3 threads * 10 updates each