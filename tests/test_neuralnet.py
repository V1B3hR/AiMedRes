import pytest
import numpy as np
import math
from collections import deque, Counter
from unittest.mock import patch, MagicMock

# Import modules to test
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuralnet import (
    Memory, SocialSignal, CapacitorInSpace, AliveLoopNode, 
    ResourceRoom, NetworkMetrics, MazeMaster, UnifiedAdaptiveAgent,
    run_labyrinth_simulation
)


class TestMemory:
    """Test the Memory dataclass with focus on edge cases and aging behavior"""
    
    def test_memory_creation_normal(self):
        """Test normal memory creation"""
        mem = Memory(content="test", importance=0.8, timestamp=100, memory_type="prediction")
        assert mem.content == "test"
        assert mem.importance == 0.8
        assert mem.timestamp == 100
        assert mem.memory_type == "prediction"
        assert mem.emotional_valence == 0.0  # default
        assert mem.decay_rate == 0.95  # default
    
    def test_memory_age_normal_decay(self):
        """Test normal memory aging"""
        mem = Memory(content="test", importance=1.0, timestamp=100, memory_type="prediction")
        original_importance = mem.importance
        mem.age()
        assert mem.importance == original_importance * 0.95
        
    def test_memory_age_with_high_emotional_valence(self):
        """Test memory aging with high emotional valence (edge case)"""
        mem = Memory(content="test", importance=1.0, timestamp=100, 
                    memory_type="prediction", emotional_valence=0.9)
        original_decay_rate = mem.decay_rate
        mem.age()
        # Should increase decay rate due to high emotional valence
        assert mem.decay_rate > original_decay_rate
        assert mem.decay_rate <= 0.997  # Should be capped
        
    def test_memory_age_with_negative_emotional_valence(self):
        """Test memory aging with negative emotional valence (edge case)"""
        mem = Memory(content="test", importance=1.0, timestamp=100, 
                    memory_type="prediction", emotional_valence=-0.8)
        original_decay_rate = mem.decay_rate
        mem.age()
        # Should increase decay rate due to absolute value of emotional valence
        assert mem.decay_rate > original_decay_rate
        
    def test_memory_age_boundary_emotional_valence(self):
        """Test memory aging at boundary emotional valence values"""
        # Exactly at threshold
        mem = Memory(content="test", importance=1.0, timestamp=100, 
                    memory_type="prediction", emotional_valence=0.7)
        original_decay_rate = mem.decay_rate
        mem.age()
        # Should NOT change decay rate at exactly 0.7
        assert mem.decay_rate == original_decay_rate
        
        # Just above threshold
        mem2 = Memory(content="test", importance=1.0, timestamp=100, 
                     memory_type="prediction", emotional_valence=0.71)
        mem2.age()
        assert mem2.decay_rate > original_decay_rate


class TestSocialSignal:
    """Test SocialSignal class with edge cases and error conditions"""
    
    def test_social_signal_creation_normal(self):
        """Test normal social signal creation"""
        signal = SocialSignal("hello", "greeting", 0.5, 123)
        assert signal.content == "hello"
        assert signal.signal_type == "greeting"
        assert signal.urgency == 0.5
        assert signal.source_id == 123
        assert signal.requires_response == False  # default
        assert signal.response is None
        assert signal.timestamp == 0
        assert signal.id is not None
        
    def test_social_signal_with_response_required(self):
        """Test social signal requiring response"""
        signal = SocialSignal("question?", "query", 0.9, 456, requires_response=True)
        assert signal.requires_response == True
        assert signal.urgency == 0.9
        
    def test_social_signal_extreme_urgency_values(self):
        """Test social signal with extreme urgency values (edge cases)"""
        # Very high urgency
        signal1 = SocialSignal("emergency", "alert", 999.9, 789)
        assert signal1.urgency == 999.9
        
        # Negative urgency
        signal2 = SocialSignal("low priority", "info", -0.5, 789)
        assert signal2.urgency == -0.5
        
        # Zero urgency
        signal3 = SocialSignal("no rush", "info", 0.0, 789)
        assert signal3.urgency == 0.0
        
    def test_social_signal_with_complex_content(self):
        """Test social signal with complex content types (edge case)"""
        complex_content = {"nested": {"data": [1, 2, 3]}, "list": ["a", "b"]}
        signal = SocialSignal(complex_content, "data", 0.7, 999)
        assert signal.content == complex_content
        
    def test_social_signal_empty_content(self):
        """Test social signal with empty content (edge case)"""
        signal = SocialSignal("", "empty", 0.1, 111)
        assert signal.content == ""
        
        signal2 = SocialSignal(None, "null", 0.1, 111)
        assert signal2.content is None


class TestCapacitorInSpace:
    """Test CapacitorInSpace with boundary conditions and energy management edge cases"""
    
    def test_capacitor_creation_normal(self):
        """Test normal capacitor creation"""
        cap = CapacitorInSpace((1, 2), capacity=10.0, initial_energy=5.0)
        assert np.array_equal(cap.position, np.array([1.0, 2.0]))
        assert cap.capacity == 10.0
        assert cap.energy == 5.0
        
    def test_capacitor_creation_with_negative_capacity(self):
        """Test capacitor creation with negative capacity (edge case)"""
        cap = CapacitorInSpace((0, 0), capacity=-5.0, initial_energy=3.0)
        assert cap.capacity == 0.0  # Should be clamped to 0
        assert cap.energy == 0.0  # Should be clamped to capacity
        
    def test_capacitor_creation_with_excessive_initial_energy(self):
        """Test capacitor creation with initial energy > capacity (edge case)"""
        cap = CapacitorInSpace((0, 0), capacity=5.0, initial_energy=10.0)
        assert cap.energy == 5.0  # Should be clamped to capacity
        
    def test_capacitor_creation_with_negative_initial_energy(self):
        """Test capacitor creation with negative initial energy (edge case)"""
        cap = CapacitorInSpace((0, 0), capacity=5.0, initial_energy=-3.0)
        assert cap.energy == 0.0  # Should be clamped to 0
        
    def test_capacitor_charge_normal(self):
        """Test normal charging operation"""
        cap = CapacitorInSpace((0, 0), capacity=10.0, initial_energy=3.0)
        cap.charge(2.0)
        assert cap.energy == 5.0
        
    def test_capacitor_charge_beyond_capacity(self):
        """Test charging beyond capacity (edge case)"""
        cap = CapacitorInSpace((0, 0), capacity=10.0, initial_energy=8.0)
        cap.charge(5.0)
        assert cap.energy == 10.0  # Should be clamped to capacity
        
    def test_capacitor_charge_negative_amount(self):
        """Test charging with negative amount (edge case)"""
        cap = CapacitorInSpace((0, 0), capacity=10.0, initial_energy=5.0)
        cap.charge(-2.0)
        assert cap.energy == 3.0  # Should effectively discharge
        
    def test_capacitor_discharge_normal(self):
        """Test normal discharge operation"""
        cap = CapacitorInSpace((0, 0), capacity=10.0, initial_energy=7.0)
        cap.discharge(3.0)
        assert cap.energy == 4.0
        
    def test_capacitor_discharge_beyond_available(self):
        """Test discharging more than available energy (edge case)"""
        cap = CapacitorInSpace((0, 0), capacity=10.0, initial_energy=2.0)
        cap.discharge(5.0)
        assert cap.energy == 0.0  # Should be clamped to 0
        
    def test_capacitor_discharge_negative_amount(self):
        """Test discharging negative amount (edge case)"""
        cap = CapacitorInSpace((0, 0), capacity=10.0, initial_energy=5.0)
        cap.discharge(-3.0)
        assert cap.energy == 8.0  # Should effectively charge
        
    def test_capacitor_status_string(self):
        """Test status string formatting"""
        cap = CapacitorInSpace((1.5, 2.7), capacity=10.0, initial_energy=3.456)
        status = cap.status()
        assert "Position [1.5 2.7]" in status
        assert "Energy 3.46/10.0" in status  # Should be rounded to 2 decimals


class TestAliveLoopNode:
    """Test AliveLoopNode with focus on safe_think method and edge cases"""
    
    def test_alive_loop_node_creation_normal(self):
        """Test normal node creation"""
        node = AliveLoopNode((1, 2), (0.1, 0.2), initial_energy=15.0, node_id=5)
        assert np.array_equal(node.position, np.array([1.0, 2.0]))
        assert np.array_equal(node.velocity, np.array([0.1, 0.2]))
        assert node.energy == 15.0
        assert node.node_id == 5
        assert node.phase == "active"
        assert node.confusion_level == 0.0
        
    def test_alive_loop_node_creation_with_negative_energy(self):
        """Test node creation with negative energy (edge case)"""
        node = AliveLoopNode((0, 0), (0, 0), initial_energy=-5.0)
        assert node.energy == 0.0  # Should be clamped to 0
        
    def test_alive_loop_node_radius_calculation(self):
        """Test radius calculation based on energy"""
        node1 = AliveLoopNode((0, 0), (0, 0), initial_energy=0.0)
        assert node1.radius == 0.1  # minimum radius
        
        node2 = AliveLoopNode((0, 0), (0, 0), initial_energy=10.0)
        expected_radius = 0.1 + 0.05 * 10.0
        assert node2.radius == expected_radius
        
    def test_safe_think_normal_operation(self):
        """Test normal safe_think operation"""
        node = AliveLoopNode((0, 0), (0, 0), initial_energy=10.0)
        result = node.safe_think("TestAgent", "solve problem")
        
        assert result["agent"] == "TestAgent"
        assert result["task"] == "solve problem"
        assert "insight" in result
        assert "confidence" in result
        assert 0.3 <= result["confidence"] <= 0.95
        assert result["energy"] == 10.0
        assert "confusion_level" in result
        
    def test_safe_think_empty_task(self):
        """Test safe_think with empty task (edge case)"""
        node = AliveLoopNode((0, 0), (0, 0), initial_energy=10.0)
        original_confusion = node.confusion_level
        
        result = node.safe_think("TestAgent", "")
        
        assert result["error"] == "Empty task"
        assert result["confidence"] == 0.0
        assert node.confusion_level > original_confusion  # Should increase confusion
        
    def test_safe_think_none_task(self):
        """Test safe_think with None task (edge case)"""
        node = AliveLoopNode((0, 0), (0, 0), initial_energy=10.0)
        original_confusion = node.confusion_level
        
        result = node.safe_think("TestAgent", None)
        
        assert result["error"] == "Empty task"
        assert result["confidence"] == 0.0
        assert node.confusion_level > original_confusion
        
    @patch('random.uniform')
    def test_safe_think_low_confidence_increases_confusion(self, mock_random):
        """Test that low confidence increases confusion level"""
        mock_random.return_value = 0.3  # Low confidence
        
        node = AliveLoopNode((0, 0), (0, 0), initial_energy=10.0)
        original_confusion = node.confusion_level
        
        result = node.safe_think("TestAgent", "difficult task")
        
        assert result["confidence"] == 0.3
        assert node.confusion_level > original_confusion
        
    @patch('random.uniform')
    def test_safe_think_high_confidence_decreases_confusion(self, mock_random):
        """Test that high confidence decreases confusion level"""
        mock_random.return_value = 0.9  # High confidence
        
        node = AliveLoopNode((0, 0), (0, 0), initial_energy=10.0)
        node.confusion_level = 0.5  # Start with some confusion
        original_confusion = node.confusion_level
        
        result = node.safe_think("TestAgent", "easy task")
        
        assert result["confidence"] == 0.9
        assert node.confusion_level < original_confusion
        
    def test_safe_think_memory_storage(self):
        """Test that safe_think stores memories"""
        node = AliveLoopNode((0, 0), (0, 0), initial_energy=10.0)
        assert len(node.memory) == 0
        
        node.safe_think("TestAgent", "task 1")
        assert len(node.memory) == 1
        
        memory = node.memory[0]
        assert memory.content == "task 1"
        assert memory.memory_type == "prediction"
        assert memory.timestamp == 1  # Should increment time
        
    def test_move_operation(self):
        """Test node movement"""
        node = AliveLoopNode((1, 2), (0.5, -0.3), initial_energy=10.0)
        original_position = node.position.copy()
        
        node.move()
        
        expected_position = original_position + node.velocity
        assert np.array_equal(node.position, expected_position)


class TestResourceRoom:
    """Test ResourceRoom data storage and retrieval with edge cases"""
    
    def test_resource_room_creation(self):
        """Test resource room creation"""
        room = ResourceRoom()
        assert room.resources == {}
        
    def test_deposit_and_retrieve_normal(self):
        """Test normal deposit and retrieve operations"""
        room = ResourceRoom()
        test_data = {"topic": "AI", "energy": 15.0}
        
        room.deposit("agent1", test_data)
        retrieved = room.retrieve("agent1")
        
        assert retrieved == test_data
        
    def test_retrieve_nonexistent_agent(self):
        """Test retrieving data for non-existent agent (edge case)"""
        room = ResourceRoom()
        retrieved = room.retrieve("nonexistent")
        assert retrieved == {}
        
    def test_deposit_overwrite(self):
        """Test depositing data overwrites previous data"""
        room = ResourceRoom()
        
        room.deposit("agent1", {"version": 1})
        room.deposit("agent1", {"version": 2})
        
        retrieved = room.retrieve("agent1")
        assert retrieved == {"version": 2}
        
    def test_deposit_empty_data(self):
        """Test depositing empty data (edge case)"""
        room = ResourceRoom()
        room.deposit("agent1", {})
        retrieved = room.retrieve("agent1")
        assert retrieved == {}
        
    def test_deposit_none_data(self):
        """Test depositing None data (edge case)"""
        room = ResourceRoom()
        room.deposit("agent1", None)
        retrieved = room.retrieve("agent1")
        assert retrieved is None
        
    def test_multiple_agents(self):
        """Test multiple agents storing different data"""
        room = ResourceRoom()
        
        room.deposit("agent1", {"data": "A"})
        room.deposit("agent2", {"data": "B"})
        room.deposit("agent3", {"data": "C"})
        
        assert room.retrieve("agent1") == {"data": "A"}
        assert room.retrieve("agent2") == {"data": "B"}
        assert room.retrieve("agent3") == {"data": "C"}


class TestNetworkMetrics:
    """Test NetworkMetrics with edge cases and boundary conditions"""
    
    def test_network_metrics_creation(self):
        """Test network metrics creation"""
        metrics = NetworkMetrics()
        assert len(metrics.energy_history) == 0
        assert len(metrics.confusion_history) == 0
        assert metrics.agent_statuses == []
        
    def test_health_score_empty_history(self):
        """Test health score with empty history (edge case)"""
        metrics = NetworkMetrics()
        score = metrics.health_score()
        assert score == 0.5  # Default score when no history
        
    def test_update_and_health_score_with_mock_agents(self):
        """Test update and health score calculation with mock agents"""
        metrics = NetworkMetrics()
        
        # Create mock agents
        mock_agents = []
        for i in range(3):
            mock_agent = MagicMock()
            mock_agent.alive_node.energy = 50.0 + i * 10  # 50, 60, 70
            mock_agent.confusion_level = 0.2 + i * 0.1    # 0.2, 0.3, 0.4
            mock_agent.status = f"agent_{i}_status"
            mock_agents.append(mock_agent)
            
        metrics.update(mock_agents)
        
        # Check that data was stored
        assert len(metrics.energy_history) == 1
        assert len(metrics.confusion_history) == 1
        assert metrics.energy_history[0] == 180.0  # 50 + 60 + 70
        assert abs(metrics.confusion_history[0] - 0.3) < 0.01  # (0.2 + 0.3 + 0.4) / 3
        
        score = metrics.health_score()
        assert 0.0 <= score <= 1.0
        
    def test_health_score_boundary_conditions(self):
        """Test health score with boundary energy and confusion values"""
        metrics = NetworkMetrics()
        
        # Test with very high energy and low confusion
        mock_agent = MagicMock()
        mock_agent.alive_node.energy = 200.0  # High energy
        mock_agent.confusion_level = 0.0      # No confusion
        mock_agent.status = "excellent"
        
        metrics.update([mock_agent])
        score = metrics.health_score()
        assert score > 0.75  # Should be high score
        
        # Test with very low energy and high confusion
        metrics2 = NetworkMetrics()
        mock_agent2 = MagicMock()
        mock_agent2.alive_node.energy = 1.0   # Very low energy
        mock_agent2.confusion_level = 1.0     # Maximum confusion
        mock_agent2.status = "struggling"
        
        metrics2.update([mock_agent2])
        score2 = metrics2.health_score()
        assert score2 < 0.25  # Should be low score
        
    def test_health_score_energy_capping(self):
        """Test that energy is capped at 100 in health score calculation"""
        metrics = NetworkMetrics()
        
        mock_agent = MagicMock()
        mock_agent.alive_node.energy = 500.0  # Way above 100
        mock_agent.confusion_level = 0.0
        mock_agent.status = "overpowered"
        
        metrics.update([mock_agent])
        score = metrics.health_score()
        
        # Energy contribution should be capped at 1.0
        expected_max_score = 0.5 * (1.0 + 1.0)  # Energy capped + confusion at 0
        assert abs(score - expected_max_score) < 0.01


# Integration tests and edge case scenarios
class TestIntegration:
    """Integration tests for component interactions and edge cases"""
    
    def test_agent_reasoning_with_empty_resource_room(self):
        """Test agent reasoning when resource room is empty"""
        resource_room = ResourceRoom()
        alive_node = AliveLoopNode((0, 0), (0, 0), 10.0)
        agent = UnifiedAdaptiveAgent("TestAgent", {"logic": 0.8}, alive_node, resource_room)
        
        result = agent.reason("test task")
        assert result["agent"] == "TestAgent"
        assert "confidence" in result
        
    def test_maze_master_intervention_edge_cases(self):
        """Test MazeMaster intervention with edge case thresholds"""
        maze_master = MazeMaster(confusion_escape_thresh=0.85, 
                               entropy_escape_thresh=1.5,
                               soft_advice_thresh=0.65)
        
        # Create agent at exact threshold
        resource_room = ResourceRoom()
        alive_node = AliveLoopNode((0, 0), (0, 0), 10.0)
        agent = UnifiedAdaptiveAgent("TestAgent", {"logic": 0.8}, alive_node, resource_room)
        
        # Set agent to exactly at escape threshold
        agent.confusion_level = 0.85
        agent.entropy = 1.5
        
        result = maze_master.intervene(agent)
        assert result["action"] == "escape"
        assert agent.status == "escaped"
        
    def test_simulation_robustness_with_no_agents(self):
        """Test network metrics with empty agent list (edge case)"""
        metrics = NetworkMetrics()
        
        # This should not crash and should not add invalid data
        metrics.update([])
        
        # History should remain empty when no agents are provided
        assert len(metrics.energy_history) == 0
        assert len(metrics.confusion_history) == 0
        
        score = metrics.health_score()
        assert score == 0.5  # Default score with no agents


if __name__ == "__main__":
    pytest.main([__file__, "-v"])