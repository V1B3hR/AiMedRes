import pytest
import numpy as np
from unittest.mock import MagicMock

# Import modules to test
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import labyrinth_adaptive


class TestLabyrinthAdaptiveBasic:
    """Basic tests for labyrinth_adaptive module to ensure edge case coverage"""
    
    def test_memory_aging_edge_case(self):
        """Test Memory aging with extreme emotional valence"""
        mem = labyrinth_adaptive.Memory(
            content="test", 
            importance=1.0, 
            timestamp=100, 
            memory_type="prediction",
            emotional_valence=0.95  # Very high emotional valence
        )
        original_decay_rate = mem.decay_rate
        mem.age()
        # Should increase decay rate due to high emotional valence
        assert mem.decay_rate > original_decay_rate
        
    def test_network_metrics_empty_agents_fix(self):
        """Test that NetworkMetrics handles empty agent lists properly (edge case fix)"""
        metrics = labyrinth_adaptive.NetworkMetrics()
        
        # This should not crash and should not add invalid data
        metrics.update([])
        
        # History should remain empty when no agents are provided
        assert len(metrics.energy_history) == 0
        assert len(metrics.confusion_history) == 0
        
        score = metrics.health_score()
        assert score == 0.5  # Default score with no agents
        
    def test_capacitor_edge_cases(self):
        """Test CapacitorInSpace with boundary conditions"""
        # Test with negative capacity (should be clamped)
        cap = labyrinth_adaptive.CapacitorInSpace((0, 0), capacity=-5.0, initial_energy=3.0)
        assert cap.capacity == 0.0
        assert cap.energy == 0.0
        
        # Test charging beyond capacity
        cap2 = labyrinth_adaptive.CapacitorInSpace((0, 0), capacity=10.0, initial_energy=8.0)
        cap2.charge(5.0)
        assert cap2.energy == 10.0  # Should be clamped to capacity
        
    def test_alive_loop_node_empty_task(self):
        """Test AliveLoopNode with empty task (edge case)"""
        node = labyrinth_adaptive.AliveLoopNode((0, 0), (0, 0), initial_energy=10.0)
        original_confusion = node.confusion_level
        
        result = node.safe_think("TestAgent", "")
        
        assert result["error"] == "Empty task"
        assert result["confidence"] == 0.0
        assert node.confusion_level > original_confusion  # Should increase confusion
        
    def test_resource_room_nonexistent_retrieval(self):
        """Test ResourceRoom retrieving non-existent data (edge case)"""
        room = labyrinth_adaptive.ResourceRoom()
        retrieved = room.retrieve("nonexistent_agent")
        assert retrieved == {}
        
    def test_maze_master_intervention_thresholds(self):
        """Test MazeMaster intervention at exact thresholds (edge case)"""
        maze_master = labyrinth_adaptive.MazeMaster(
            confusion_escape_thresh=0.85,
            entropy_escape_thresh=1.5
        )
        
        # Create mock agent at exact threshold
        mock_agent = MagicMock()
        mock_agent.confusion_level = 0.85  # Exactly at threshold
        mock_agent.entropy = 1.5  # Exactly at threshold
        mock_agent.status = "active"
        mock_agent.name = "TestAgent"
        
        result = maze_master.intervene(mock_agent)
        assert result["action"] == "escape"  # Should trigger escape at threshold
        
    def test_unified_adaptive_agent_style_influence(self):
        """Test UnifiedAdaptiveAgent style influence (boundary conditions)"""
        resource_room = labyrinth_adaptive.ResourceRoom()
        alive_node = labyrinth_adaptive.AliveLoopNode((0, 0), (0, 0), 10.0)
        
        # Agent with high style values (> 0.7 threshold)
        agent = labyrinth_adaptive.UnifiedAdaptiveAgent(
            "TestAgent", 
            {"logic": 0.8, "creativity": 0.9},  # Both > 0.7
            alive_node, 
            resource_room
        )
        
        result = agent.reason("test task")
        style_insights = result.get("style_insights", [])
        assert "Logic influence" in style_insights
        assert "Creativity influence" in style_insights


if __name__ == "__main__":
    pytest.main([__file__, "-v"])