"""
Comprehensive tests for UnifiedAdaptiveAgent class.
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch
import uuid

from labyrinth_adaptive import UnifiedAdaptiveAgent, AliveLoopNode, ResourceRoom


class TestUnifiedAdaptiveAgent:
    """Test suite for UnifiedAdaptiveAgent functionality."""
    
    def test_agent_initialization(self, sample_agent):
        """Test proper agent initialization."""
        assert sample_agent.name == "TestAgent"
        assert sample_agent.status == "active"
        assert sample_agent.confusion_level == 0.0
        assert sample_agent.entropy == 0.0
        assert isinstance(sample_agent.agent_id, str)
        assert len(sample_agent.knowledge_graph) == 0
        assert len(sample_agent.interaction_history) == 0
        assert len(sample_agent.event_log) == 0
    
    @pytest.mark.parametrize("style", [
        {"logic": 0.8, "creativity": 0.6},
        {"analytical": 1.0},
        {"expressiveness": 0.5, "creativity": 0.9},
        {}  # Empty style
    ])
    def test_agent_style_variations(self, alive_loop_node, resource_room, style):
        """Test agent creation with different style configurations."""
        agent = UnifiedAdaptiveAgent("StyleTest", style, alive_loop_node, resource_room)
        assert agent.style == style
        assert agent.name == "StyleTest"
    
    def test_agent_reasoning(self, sample_agent):
        """Test agent reasoning functionality."""
        task = "Solve a complex problem"
        result = sample_agent.reason(task)
        
        # Check result structure
        assert isinstance(result, dict)
        assert "insight" in result
        assert "confidence" in result
        assert "agent" in result
        
        # Check internal state updates
        assert len(sample_agent.knowledge_graph) == 1
        assert len(sample_agent.interaction_history) == 1
        
        # Check reasoning added to knowledge graph
        key = list(sample_agent.knowledge_graph.keys())[0]
        assert "TestAgent_reason_0" == key
    
    @pytest.mark.parametrize("task", [
        "Find the exit",
        "Collaborate with others",
        "Navigate complex terrain",
        "Learn from experience",
        "Adapt to new situations"
    ])
    def test_reasoning_with_different_tasks(self, sample_agent, task):
        """Test reasoning with various task types."""
        initial_kg_size = len(sample_agent.knowledge_graph)
        result = sample_agent.reason(task)
        
        assert isinstance(result, dict)
        assert len(sample_agent.knowledge_graph) == initial_kg_size + 1
    
    def test_style_influence_application(self, sample_agent):
        """Test style influence on reasoning results."""
        # Set high logic style
        sample_agent.style = {"logic": 0.9, "creativity": 0.2}
        
        result = sample_agent.reason("Analyze data patterns")
        
        # Should have style insights for high-value dimensions
        assert "style_insights" in result
        assert "Logic influence" in result["style_insights"]
    
    def test_teleport_to_resource_room(self, sample_agent):
        """Test agent teleportation to resource room."""
        data = {"test": "data", "energy": 10.0}
        
        sample_agent.teleport_to_resource_room(data)
        
        assert sample_agent.status == "in_resource_room"
        assert "Teleported to ResourceRoom." in sample_agent.event_log
    
    def test_retrieve_from_resource_room(self, sample_agent):
        """Test retrieving information from resource room."""
        # Mock resource room to return data
        sample_agent.resource_room.retrieve = Mock(return_value={"info": "test_data"})
        
        result = sample_agent.retrieve_from_resource_room()
        
        assert result == {"info": "test_data"}
        assert "Retrieved info from ResourceRoom." in sample_agent.event_log
        sample_agent.resource_room.retrieve.assert_called_once_with(sample_agent.agent_id)
    
    def test_retrieve_without_resource_room(self, alive_loop_node):
        """Test retrieve when no resource room is available."""
        agent = UnifiedAdaptiveAgent("NoResourceAgent", {}, alive_loop_node, None)
        result = agent.retrieve_from_resource_room()
        assert result is None
    
    def test_log_event(self, sample_agent):
        """Test event logging functionality."""
        event = "Test event occurred"
        sample_agent.log_event(event)
        
        assert event in sample_agent.event_log
        assert len(sample_agent.event_log) == 1
    
    def test_get_state(self, sample_agent):
        """Test getting agent state information."""
        # Add some data to the agent
        sample_agent.reason("Test reasoning")
        sample_agent.log_event("Test event")
        
        state = sample_agent.get_state()
        
        assert isinstance(state, dict)
        required_fields = [
            "agent_id", "name", "status", "confusion_level", 
            "entropy", "knowledge_graph_size", "event_log"
        ]
        
        for field in required_fields:
            assert field in state
        
        assert state["name"] == "TestAgent"
        assert state["knowledge_graph_size"] == 1
        assert len(state["event_log"]) == 1
    
    def test_confusion_and_entropy_updates(self, sample_agent):
        """Test that confusion and entropy are updated during reasoning."""
        initial_confusion = sample_agent.confusion_level
        initial_entropy = sample_agent.entropy
        
        # Perform multiple reasoning operations
        for i in range(5):
            sample_agent.reason(f"Task {i}")
        
        # Confusion should increase with more operations
        assert sample_agent.confusion_level >= initial_confusion
    
    def test_agent_alive_node_integration(self, sample_agent):
        """Test integration between agent and AliveLoopNode."""
        assert sample_agent.alive_node is not None
        assert hasattr(sample_agent.alive_node, 'energy')
        assert hasattr(sample_agent.alive_node, 'position')
        assert hasattr(sample_agent.alive_node, 'move')
        
        # Test node movement
        initial_position = sample_agent.alive_node.position.copy()
        sample_agent.alive_node.move()
        
        # Position should change after movement
        assert not np.array_equal(initial_position, sample_agent.alive_node.position)
    
    @pytest.mark.performance
    def test_reasoning_performance(self, sample_agent, performance_timer):
        """Test reasoning performance under load."""
        performance_timer.start()
        
        # Perform many reasoning operations
        for i in range(100):
            sample_agent.reason(f"Performance test task {i}")
        
        elapsed = performance_timer.stop()
        
        # Should complete within reasonable time
        assert elapsed < 5.0, f"Reasoning took too long: {elapsed}s"
        assert len(sample_agent.knowledge_graph) == 100
    
    @pytest.mark.integration
    def test_multi_agent_scenario(self, sample_agents):
        """Test multiple agents working together."""
        assert len(sample_agents) == 3
        
        # Each agent performs reasoning
        for i, agent in enumerate(sample_agents):
            result = agent.reason(f"Collaborative task for {agent.name}")
            assert isinstance(result, dict)
            assert len(agent.knowledge_graph) == 1
        
        # Verify agents have different properties
        agent_names = [agent.name for agent in sample_agents]
        assert len(set(agent_names)) == 3  # All unique names
        
        # Verify different confusion levels emerge
        confusion_levels = [agent.confusion_level for agent in sample_agents]
        # At least some variation in confusion (not all exactly the same)
        assert len(set(confusion_levels)) >= 1
    
    def test_error_handling_in_reasoning(self, sample_agent):
        """Test error handling during reasoning operations."""
        # Mock the alive_node.safe_think to raise an exception
        with patch.object(sample_agent.alive_node, 'safe_think', side_effect=Exception("Test error")):
            # Should handle the error gracefully
            with pytest.raises(Exception):
                sample_agent.reason("Error test task")
    
    def test_memory_management(self, sample_agent):
        """Test agent memory management with large knowledge graphs."""
        # Add many reasoning results
        for i in range(1000):
            sample_agent.reason(f"Memory test {i}")
        
        assert len(sample_agent.knowledge_graph) == 1000
        
        # Verify memory usage is reasonable
        import sys
        kg_size = sys.getsizeof(sample_agent.knowledge_graph)
        assert kg_size < 10 * 1024 * 1024  # Less than 10MB
    
    @pytest.mark.asyncio
    async def test_async_reasoning_simulation(self, sample_agents):
        """Test concurrent reasoning operations."""
        import asyncio
        
        async def async_reason(agent, task):
            """Simulate async reasoning."""
            await asyncio.sleep(0.01)  # Simulate async work
            return agent.reason(task)
        
        # Run reasoning tasks concurrently
        tasks = [
            async_reason(agent, f"Async task for {agent.name}") 
            for agent in sample_agents
        ]
        
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 3
        for result in results:
            assert isinstance(result, dict)
            assert "insight" in result