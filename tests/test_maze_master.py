"""
Comprehensive tests for MazeMaster governance system.
"""
import pytest
from unittest.mock import Mock, patch

from labyrinth_adaptive import MazeMaster, UnifiedAdaptiveAgent


class TestMazeMaster:
    """Test suite for MazeMaster governance functionality."""
    
    def test_maze_master_initialization(self, maze_master):
        """Test proper MazeMaster initialization."""
        assert maze_master.interventions == 0
        assert maze_master.confusion_escape_thresh == 0.85
        assert maze_master.entropy_escape_thresh == 1.5
        assert maze_master.soft_advice_thresh == 0.65
    
    @pytest.mark.parametrize("confusion_thresh,entropy_thresh,advice_thresh", [
        (0.9, 2.0, 0.7),
        (0.8, 1.0, 0.6),
        (0.95, 1.8, 0.75)
    ])
    def test_custom_thresholds(self, confusion_thresh, entropy_thresh, advice_thresh):
        """Test MazeMaster with custom threshold settings."""
        master = MazeMaster(
            confusion_escape_thresh=confusion_thresh,
            entropy_escape_thresh=entropy_thresh,
            soft_advice_thresh=advice_thresh
        )
        
        assert master.confusion_escape_thresh == confusion_thresh
        assert master.entropy_escape_thresh == entropy_thresh
        assert master.soft_advice_thresh == advice_thresh
    
    def test_quick_escape_intervention(self, maze_master, sample_agent):
        """Test quick escape intervention mechanism."""
        result = maze_master.quick_escape(sample_agent)
        
        assert isinstance(result, dict)
        assert result["action"] == "escape"
        assert "guided out by MazeMaster" in result["message"]
        assert sample_agent.status == "escaped"
        assert "MazeMaster: Quick escape triggered." in sample_agent.event_log
    
    def test_psychologist_intervention(self, maze_master, sample_agent):
        """Test psychologist intervention mechanism."""
        result = maze_master.psychologist(sample_agent)
        
        assert isinstance(result, dict)
        assert result["action"] == "advice"
        assert isinstance(result["message"], list)
        assert len(result["message"]) > 0
        # Should contain some advice
        advice_text = " ".join(result["message"])
        assert len(advice_text) > 0
    
    def test_no_intervention_needed(self, maze_master, sample_agent):
        """Test when no intervention is needed."""
        # Set agent to low confusion/entropy state
        sample_agent.confusion_level = 0.1
        sample_agent.entropy = 0.1
        sample_agent.status = "active"
        
        result = maze_master.intervene(sample_agent)
        
        assert result["action"] == "none"
        assert maze_master.interventions == 1  # Still counts as intervention check
    
    def test_confusion_escape_threshold_triggered(self, maze_master, sample_agent):
        """Test intervention when confusion exceeds escape threshold."""
        sample_agent.confusion_level = 0.9  # Above default 0.85
        sample_agent.entropy = 0.1
        sample_agent.status = "active"
        
        result = maze_master.intervene(sample_agent)
        
        assert result["action"] == "escape"
        assert sample_agent.status == "escaped"
        assert maze_master.interventions == 1
    
    def test_entropy_escape_threshold_triggered(self, maze_master, sample_agent):
        """Test intervention when entropy exceeds escape threshold."""
        sample_agent.confusion_level = 0.1
        sample_agent.entropy = 2.0  # Above default 1.5
        sample_agent.status = "active"
        
        result = maze_master.intervene(sample_agent)
        
        assert result["action"] == "escape"
        assert sample_agent.status == "escaped"
    
    @pytest.mark.parametrize("status", ["stuck", "looping"])
    def test_problematic_status_escape(self, maze_master, sample_agent, status):
        """Test escape intervention for problematic agent statuses."""
        sample_agent.confusion_level = 0.1
        sample_agent.entropy = 0.1
        sample_agent.status = status
        
        result = maze_master.intervene(sample_agent)
        
        assert result["action"] == "escape"
        assert sample_agent.status == "escaped"
    
    def test_soft_advice_threshold_triggered(self, maze_master, sample_agent):
        """Test soft advice when confusion/entropy is moderate."""
        sample_agent.confusion_level = 0.7  # Above 0.65 but below 0.85
        sample_agent.entropy = 0.1
        sample_agent.status = "active"
        
        result = maze_master.intervene(sample_agent)
        
        assert result["action"] == "advice"
        assert isinstance(result["message"], list)
        assert maze_master.interventions == 1
    
    def test_entropy_soft_advice_threshold(self, maze_master, sample_agent):
        """Test soft advice when entropy is moderate."""
        sample_agent.confusion_level = 0.1
        sample_agent.entropy = 1.0  # Above 0.65 but below 1.5
        sample_agent.status = "active"
        
        result = maze_master.intervene(sample_agent)
        
        assert result["action"] == "advice"
    
    def test_govern_agents_single(self, maze_master, sample_agent):
        """Test governing a single agent."""
        # Set up agent for intervention
        sample_agent.confusion_level = 0.7
        
        maze_master.govern_agents([sample_agent])
        
        assert maze_master.interventions == 1
        # Should have intervention message in event log
        assert any("MazeMaster intervention:" in event for event in sample_agent.event_log)
    
    def test_govern_agents_multiple(self, maze_master, sample_agents):
        """Test governing multiple agents."""
        # Set different states for agents
        sample_agents[0].confusion_level = 0.9  # Should escape
        sample_agents[1].confusion_level = 0.7  # Should get advice
        sample_agents[2].confusion_level = 0.1  # No intervention
        
        maze_master.govern_agents(sample_agents)
        
        assert maze_master.interventions == 3  # All agents checked
        
        # Check interventions were applied
        assert sample_agents[0].status == "escaped"
        assert any("MazeMaster intervention:" in event for event in sample_agents[1].event_log)
    
    def test_intervention_counting(self, maze_master, sample_agent):
        """Test that interventions are properly counted."""
        initial_count = maze_master.interventions
        
        # Perform multiple interventions
        for i in range(5):
            maze_master.intervene(sample_agent)
        
        assert maze_master.interventions == initial_count + 5
    
    @pytest.mark.parametrize("confusion,entropy,expected_action", [
        (0.1, 0.1, "none"),
        (0.7, 0.1, "advice"),
        (0.1, 1.0, "advice"),
        (0.9, 0.1, "escape"),
        (0.1, 2.0, "escape"),
        (0.9, 2.0, "escape")
    ])
    def test_intervention_decision_matrix(self, maze_master, sample_agent, 
                                        confusion, entropy, expected_action):
        """Test intervention decisions across various confusion/entropy combinations."""
        sample_agent.confusion_level = confusion
        sample_agent.entropy = entropy
        sample_agent.status = "active"
        
        result = maze_master.intervene(sample_agent)
        
        assert result["action"] == expected_action
    
    @pytest.mark.performance
    def test_governance_performance(self, maze_master, sample_agents, performance_timer):
        """Test governance performance with many agents."""
        # Create many agents
        many_agents = sample_agents * 100  # 300 agents
        
        performance_timer.start()
        maze_master.govern_agents(many_agents)
        elapsed = performance_timer.stop()
        
        # Should complete quickly even with many agents
        assert elapsed < 1.0, f"Governance took too long: {elapsed}s"
        assert maze_master.interventions == 300
    
    def test_psychologist_advice_variety(self, maze_master, sample_agent):
        """Test that psychologist gives varied advice."""
        advice_sets = []
        
        # Get multiple advice responses
        for _ in range(10):
            result = maze_master.psychologist(sample_agent)
            advice_text = " ".join(result["message"])
            advice_sets.append(advice_text)
        
        # Should have some variety in advice (not all identical)
        unique_advice = set(advice_sets)
        assert len(unique_advice) >= 3, "Advice should have some variety"
    
    @pytest.mark.integration
    def test_full_simulation_governance(self, maze_master, sample_agents):
        """Test governance throughout a simulated scenario."""
        steps = 20
        intervention_history = []
        
        for step in range(steps):
            # Simulate agent activity causing confusion to increase
            for agent in sample_agents:
                agent.reason(f"Step {step} task")
                # Manually increase confusion to simulate complex scenarios
                agent.confusion_level = min(1.0, agent.confusion_level + 0.05)
            
            # Apply governance
            initial_interventions = maze_master.interventions
            maze_master.govern_agents(sample_agents)
            step_interventions = maze_master.interventions - initial_interventions
            intervention_history.append(step_interventions)
        
        # Should have performed interventions as confusion increased
        total_interventions = sum(intervention_history)
        assert total_interventions > 0
        
        # Later steps should have more interventions as confusion builds up
        early_interventions = sum(intervention_history[:5])
        late_interventions = sum(intervention_history[-5:])
        # Later interventions should be >= early ones (confusion accumulates)
        assert late_interventions >= early_interventions
    
    def test_error_handling_in_intervention(self, maze_master, sample_agent):
        """Test error handling during intervention processes."""
        # Mock an error in the quick_escape method
        with patch.object(maze_master, 'quick_escape', side_effect=Exception("Test error")):
            sample_agent.confusion_level = 0.9  # Should trigger escape
            
            # Should handle error gracefully
            with pytest.raises(Exception):
                maze_master.intervene(sample_agent)
    
    def test_agent_state_preservation(self, maze_master, sample_agent):
        """Test that agent state is properly preserved during interventions."""
        # Set up initial state
        original_name = sample_agent.name
        original_id = sample_agent.agent_id
        sample_agent.confusion_level = 0.7  # Should get advice
        
        maze_master.intervene(sample_agent)
        
        # Core identity should be preserved
        assert sample_agent.name == original_name
        assert sample_agent.agent_id == original_id
        
        # But event log should be updated
        assert len(sample_agent.event_log) > 0