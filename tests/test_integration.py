"""
Integration tests for the complete duetmind_adaptive system.
"""
import pytest
import numpy as np
import time
from unittest.mock import patch

from labyrinth_adaptive import (
    run_labyrinth_simulation, UnifiedAdaptiveAgent, MazeMaster, 
    NetworkMetrics, ResourceRoom, AliveLoopNode, CapacitorInSpace
)


class TestSystemIntegration:
    """Integration tests for the complete system."""
    
    @pytest.mark.integration
    def test_full_simulation_run(self, capsys):
        """Test running the complete labyrinth simulation."""
        # Patch time.sleep to speed up the test
        with patch('time.sleep'):
            # Capture the simulation output
            run_labyrinth_simulation()
            
            captured = capsys.readouterr()
            
            # Verify simulation ran and produced expected output
            assert "Unified Adaptive Labyrinth Simulation" in captured.out
            assert "Simulation Complete" in captured.out
            assert "Step" in captured.out
            assert "Network Health Score" in captured.out
    
    @pytest.mark.integration
    def test_component_interactions(self, sample_agents, maze_master, network_metrics):
        """Test interactions between major components."""
        capacitor = CapacitorInSpace((1, 1), capacity=8.0, initial_energy=3.0)
        
        # Simulate several steps of the system
        for step in range(5):
            # Agents perform reasoning
            for i, agent in enumerate(sample_agents):
                task = f"Integration test step {step} for {agent.name}"
                result = agent.reason(task)
                assert isinstance(result, dict)
                
                # Move agents
                agent.alive_node.move()
            
            # Apply governance
            maze_master.govern_agents(sample_agents)
            
            # Update metrics
            network_metrics.update(sample_agents)
            health_score = network_metrics.health_score()
            
            # Verify health score is reasonable
            assert 0 <= health_score <= 1
        
        # Verify all components have updated state
        assert maze_master.interventions >= 0
        assert len(network_metrics.energy_history) == 5
        assert all(len(agent.knowledge_graph) == 5 for agent in sample_agents)
    
    @pytest.mark.integration
    def test_resource_room_integration(self, sample_agents):
        """Test resource room integration with agents."""
        # Agents teleport and retrieve from resource room
        for i, agent in enumerate(sample_agents):
            data = {"test_data": f"Data for {agent.name}", "step": i}
            
            # Teleport to resource room
            agent.teleport_to_resource_room(data)
            assert agent.status == "in_resource_room"
            
            # Retrieve information
            retrieved = agent.retrieve_from_resource_room()
            # ResourceRoom might return None or some data structure
            # The exact return value depends on the implementation
            
            # Verify event logging
            assert "Teleported to ResourceRoom." in agent.event_log
            assert "Retrieved info from ResourceRoom." in agent.event_log
    
    @pytest.mark.integration
    def test_multi_agent_reasoning_chain(self, sample_agents):
        """Test chained reasoning across multiple agents."""
        topics = ["Explore environment", "Share findings", "Collaborate"]
        
        # Create a reasoning chain
        for round_num in range(3):
            for i, agent in enumerate(sample_agents):
                topic = topics[round_num % len(topics)]
                
                # Each agent reasons about the topic
                result = agent.reason(f"{topic} - Round {round_num}")
                
                # Verify reasoning result
                assert isinstance(result, dict)
                assert "insight" in result
                
                # Knowledge graph should grow
                expected_size = (round_num + 1)
                assert len(agent.knowledge_graph) == expected_size
        
        # Verify all agents have developed knowledge
        for agent in sample_agents:
            assert len(agent.knowledge_graph) == 3
            assert len(agent.interaction_history) == 3
    
    @pytest.mark.integration
    def test_system_under_stress(self, maze_master, network_metrics):
        """Test system behavior under stress conditions."""
        # Create many agents with high confusion
        stress_agents = []
        for i in range(20):
            position = np.array([i, i])
            velocity = np.array([0.1, 0.1])
            node = AliveLoopNode(position, velocity, initial_energy=10.0, node_id=i)
            resource_room = ResourceRoom()
            agent = UnifiedAdaptiveAgent(f"StressAgent{i}", {"logic": 0.5}, node, resource_room)
            
            # Set high confusion to trigger interventions
            agent.confusion_level = 0.8
            stress_agents.append(agent)
        
        # Run stress test
        initial_interventions = maze_master.interventions
        maze_master.govern_agents(stress_agents)
        network_metrics.update(stress_agents)
        
        # Verify system handled stress
        assert maze_master.interventions > initial_interventions
        health_score = network_metrics.health_score()
        assert 0 <= health_score <= 1
    
    @pytest.mark.integration
    @pytest.mark.performance
    def test_system_performance(self, performance_timer):
        """Test overall system performance."""
        performance_timer.start()
        
        # Create a mini simulation
        resource_room = ResourceRoom()
        maze_master = MazeMaster()
        metrics = NetworkMetrics()
        
        # Create agents
        agents = []
        for i in range(10):
            position = np.array([i, 0])
            velocity = np.array([0.1, 0])
            node = AliveLoopNode(position, velocity, initial_energy=15.0, node_id=i)
            style = {"logic": 0.5 + i * 0.05, "creativity": 0.3 + i * 0.02}
            agent = UnifiedAdaptiveAgent(f"PerfAgent{i}", style, node, resource_room)
            agents.append(agent)
        
        # Run simulation steps
        for step in range(20):
            for agent in agents:
                agent.reason(f"Performance test step {step}")
                agent.alive_node.move()
            
            maze_master.govern_agents(agents)
            metrics.update(agents)
        
        elapsed = performance_timer.stop()
        
        # Should complete within reasonable time
        assert elapsed < 5.0, f"System performance test took too long: {elapsed}s"
        
        # Verify all components worked
        assert len(metrics.energy_history) == 20
        assert maze_master.interventions >= 0
    
    @pytest.mark.integration
    def test_capacitor_system_integration(self):
        """Test capacitor integration with the system."""
        capacitors = [
            CapacitorInSpace((1, 1), capacity=8.0, initial_energy=3.0),
            CapacitorInSpace((5, 5), capacity=10.0, initial_energy=7.0)
        ]
        
        # Test capacitor operations
        for capacitor in capacitors:
            initial_energy = capacitor.energy
            
            # Test charging
            capacitor.charge(2.0)
            assert capacitor.energy > initial_energy
            
            # Test discharging
            discharged = capacitor.discharge(1.0)
            assert discharged <= 1.0
            
            # Test status reporting
            status = capacitor.status()
            assert isinstance(status, str)
            assert "Position" in status
            assert "Energy" in status
    
    @pytest.mark.integration
    def test_memory_and_social_signals(self):
        """Test memory and social signal systems."""
        from labyrinth_adaptive import Memory, SocialSignal
        
        # Test memory creation and aging
        memory = Memory(
            content="Test memory",
            importance=0.8,
            timestamp=12345,
            memory_type="test"
        )
        
        initial_importance = memory.importance
        memory.age()
        assert memory.importance <= initial_importance
        
        # Test social signal
        signal = SocialSignal(
            content="Test signal",
            signal_type="communication",
            urgency=0.7,
            source_id=1,
            requires_response=True
        )
        
        assert signal.requires_response is True
        assert signal.urgency == 0.7
        assert isinstance(signal.id, str)
    
    @pytest.mark.integration
    def test_error_recovery(self, sample_agents, maze_master):
        """Test system error recovery capabilities."""
        # Simulate various error conditions
        
        # Test with corrupted agent data
        sample_agents[0].confusion_level = float('inf')  # Invalid value
        
        # System should handle this gracefully
        try:
            maze_master.govern_agents(sample_agents)
            # If no exception, that's good
        except Exception as e:
            # If exception occurs, it should be handled appropriately
            assert isinstance(e, (ValueError, TypeError))
        
        # Reset to valid state
        sample_agents[0].confusion_level = 0.5
        
        # Test with missing attributes (simulate corruption)
        original_status = sample_agents[1].status
        del sample_agents[1].status
        
        try:
            maze_master.govern_agents(sample_agents)
        except AttributeError:
            # Expected - system should detect missing attributes
            pass
        
        # Restore attribute
        sample_agents[1].status = original_status
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_long_running_simulation(self, capsys):
        """Test long-running simulation stability."""
        with patch('time.sleep'):  # Speed up the test
            # Mock the simulation to run more steps
            with patch('labyrinth_adaptive.range', return_value=range(1, 51)):  # 50 steps
                run_labyrinth_simulation()
                
                captured = capsys.readouterr()
                
                # Should complete without errors
                assert "Simulation Complete" in captured.out
                
                # Should have run many steps
                step_count = captured.out.count("--- Step")
                assert step_count >= 40  # Should have many steps
    
    @pytest.mark.integration
    def test_system_state_consistency(self, sample_agents, maze_master, network_metrics):
        """Test that system state remains consistent across operations."""
        # Take initial snapshots
        initial_agent_ids = [agent.agent_id for agent in sample_agents]
        initial_interventions = maze_master.interventions
        
        # Perform operations
        for i in range(10):
            # Agent operations
            for agent in sample_agents:
                agent.reason(f"Consistency test {i}")
                agent.alive_node.move()
            
            # System operations
            maze_master.govern_agents(sample_agents)
            network_metrics.update(sample_agents)
        
        # Verify consistency
        # Agent IDs should remain the same
        final_agent_ids = [agent.agent_id for agent in sample_agents]
        assert initial_agent_ids == final_agent_ids
        
        # Interventions should have increased
        assert maze_master.interventions >= initial_interventions
        
        # All agents should have knowledge
        for agent in sample_agents:
            assert len(agent.knowledge_graph) == 10
        
        # Metrics should have history
        assert len(network_metrics.energy_history) == 10
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_system_operations(self, sample_agents):
        """Test asynchronous system operations."""
        import asyncio
        
        async def async_agent_operation(agent, task_id):
            """Simulate async agent operation."""
            await asyncio.sleep(0.01)  # Simulate async work
            result = agent.reason(f"Async task {task_id}")
            return result
        
        # Run multiple agents concurrently
        tasks = [
            async_agent_operation(agent, i) 
            for i, agent in enumerate(sample_agents)
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Verify all operations completed
        assert len(results) == len(sample_agents)
        for result in results:
            assert isinstance(result, dict)
            assert "insight" in result