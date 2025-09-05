"""
Comprehensive tests for CapacitorInSpace and ResourceRoom functionality.
"""
import pytest
import numpy as np

from labyrinth_adaptive import CapacitorInSpace, ResourceRoom


class TestCapacitorInSpace:
    """Test suite for CapacitorInSpace functionality."""
    
    def test_capacitor_initialization(self, capacitor):
        """Test proper capacitor initialization."""
        assert isinstance(capacitor.position, np.ndarray)
        assert capacitor.capacity == 10.0
        assert capacitor.energy == 5.0
        assert 0 <= capacitor.energy <= capacitor.capacity
    
    @pytest.mark.parametrize("position,capacity,initial_energy", [
        ((0, 0), 5.0, 2.0),
        ((1.5, 2.5), 20.0, 15.0),
        ((-1, -2), 8.0, 0.0),
        ((10, 10), 100.0, 50.0)
    ])
    def test_capacitor_variations(self, position, capacity, initial_energy):
        """Test capacitor with various configurations."""
        cap = CapacitorInSpace(position, capacity, initial_energy)
        
        assert np.allclose(cap.position, np.array(position))
        assert cap.capacity == max(0.0, capacity)
        assert cap.energy == min(max(0.0, initial_energy), cap.capacity)
    
    def test_capacitor_charging(self, capacitor):
        """Test capacitor charging functionality."""
        initial_energy = capacitor.energy
        charge_amount = 2.0
        
        capacitor.charge(charge_amount)
        
        expected_energy = min(capacitor.capacity, initial_energy + charge_amount)
        assert capacitor.energy == expected_energy
    
    def test_capacitor_overcharge_protection(self, capacitor):
        """Test that capacitor doesn't overcharge beyond capacity."""
        capacitor.energy = 0.0  # Start empty
        overcharge_amount = capacitor.capacity + 5.0  # More than capacity
        
        capacitor.charge(overcharge_amount)
        
        assert capacitor.energy == capacitor.capacity  # Should be capped
    
    def test_capacitor_discharging(self, capacitor):
        """Test capacitor discharging functionality."""
        initial_energy = capacitor.energy
        discharge_amount = 2.0
        
        capacitor.discharge(discharge_amount)
        
        expected_energy = max(0.0, initial_energy - discharge_amount)
        assert capacitor.energy == expected_energy
    
    def test_capacitor_discharge_when_empty(self):
        """Test discharging when capacitor is empty."""
        cap = CapacitorInSpace((0, 0), capacity=10.0, initial_energy=0.0)
        
        cap.discharge(5.0)
        
        assert cap.energy == 0.0
    
    def test_capacitor_status_reporting(self, capacitor):
        """Test capacitor status string generation."""
        status = capacitor.status()
        
        assert isinstance(status, str)
        assert "Position" in status
        assert "Energy" in status
        assert str(capacitor.energy) in status
        assert str(capacitor.capacity) in status
    
    @pytest.mark.parametrize("charge_amount", [
        1.0, 2.5, 0.0, -1.0, 10.0
    ])
    def test_capacitor_charge_edge_cases(self, capacitor, charge_amount):
        """Test charging with various amounts including edge cases."""
        initial_energy = capacitor.energy
        capacitor.charge(charge_amount)
        
        # Energy should never exceed capacity or go below initial
        assert 0 <= capacitor.energy <= capacitor.capacity
        if charge_amount >= 0:
            assert capacitor.energy >= initial_energy
    
    @pytest.mark.parametrize("discharge_amount", [
        1.0, 2.5, 0.0, -1.0, 100.0
    ])
    def test_capacitor_discharge_edge_cases(self, capacitor, discharge_amount):
        """Test discharging with various amounts including edge cases."""
        initial_energy = capacitor.energy
        capacitor.discharge(discharge_amount)
        
        # Energy should never go negative
        assert capacitor.energy >= 0
        # Energy should decrease or stay the same (for positive discharge amounts)
        if discharge_amount > 0:
            assert capacitor.energy <= initial_energy


class TestResourceRoom:
    """Test suite for ResourceRoom functionality."""
    
    def test_resource_room_initialization(self, resource_room):
        """Test proper ResourceRoom initialization."""
        assert isinstance(resource_room.resources, dict)
        assert len(resource_room.resources) == 0
    
    def test_deposit_and_retrieve(self, resource_room):
        """Test basic deposit and retrieve operations."""
        agent_id = "test_agent_123"
        test_data = {"knowledge": "test_knowledge", "energy": 50.0}
        
        # Deposit data
        resource_room.deposit(agent_id, test_data)
        
        # Verify data was stored
        assert agent_id in resource_room.resources
        assert resource_room.resources[agent_id] == test_data
        
        # Retrieve data
        retrieved = resource_room.retrieve(agent_id)
        assert retrieved == test_data
    
    def test_retrieve_nonexistent_agent(self, resource_room):
        """Test retrieving data for non-existent agent."""
        result = resource_room.retrieve("nonexistent_agent")
        assert result == {}
    
    def test_multiple_agents(self, resource_room):
        """Test resource room with multiple agents."""
        agents_data = {
            "agent_1": {"task": "exploration", "progress": 0.5},
            "agent_2": {"task": "analysis", "progress": 0.8},
            "agent_3": {"task": "collaboration", "progress": 0.3}
        }
        
        # Deposit data for all agents
        for agent_id, data in agents_data.items():
            resource_room.deposit(agent_id, data)
        
        # Verify all data can be retrieved correctly
        for agent_id, expected_data in agents_data.items():
            retrieved = resource_room.retrieve(agent_id)
            assert retrieved == expected_data
        
        # Verify all agents are in the resource room
        assert len(resource_room.resources) == 3
    
    def test_overwrite_existing_data(self, resource_room):
        """Test overwriting existing agent data."""
        agent_id = "test_agent"
        original_data = {"task": "original", "value": 1}
        new_data = {"task": "updated", "value": 2}
        
        # Deposit original data
        resource_room.deposit(agent_id, original_data)
        assert resource_room.retrieve(agent_id) == original_data
        
        # Overwrite with new data
        resource_room.deposit(agent_id, new_data)
        assert resource_room.retrieve(agent_id) == new_data
    
    def test_empty_data_handling(self, resource_room):
        """Test handling of empty or None data."""
        agent_id = "empty_agent"
        
        # Test with empty dict
        resource_room.deposit(agent_id, {})
        assert resource_room.retrieve(agent_id) == {}
        
        # Test with None (if allowed by implementation)
        resource_room.deposit(agent_id, None)
        assert resource_room.retrieve(agent_id) is None
    
    @pytest.mark.parametrize("data_type", [
        {"string": "text"},
        {"number": 42},
        {"list": [1, 2, 3]},
        {"nested": {"inner": {"value": "deep"}}},
        {"mixed": {"str": "text", "num": 123, "list": [1, 2]}}
    ])
    def test_various_data_types(self, resource_room, data_type):
        """Test resource room with various data types."""
        agent_id = "type_test_agent"
        
        resource_room.deposit(agent_id, data_type)
        retrieved = resource_room.retrieve(agent_id)
        
        assert retrieved == data_type
    
    @pytest.mark.performance
    def test_large_scale_operations(self, resource_room, performance_timer):
        """Test resource room performance with many agents."""
        num_agents = 1000
        
        performance_timer.start()
        
        # Deposit data for many agents
        for i in range(num_agents):
            agent_id = f"agent_{i}"
            data = {"id": i, "task": f"task_{i}", "progress": i / num_agents}
            resource_room.deposit(agent_id, data)
        
        # Retrieve data for all agents
        for i in range(num_agents):
            agent_id = f"agent_{i}"
            retrieved = resource_room.retrieve(agent_id)
            assert retrieved["id"] == i
        
        elapsed = performance_timer.stop()
        
        # Should complete quickly even with many agents
        assert elapsed < 1.0, f"Large scale operations took too long: {elapsed}s"
        assert len(resource_room.resources) == num_agents
    
    @pytest.mark.integration
    def test_resource_room_with_capacitors(self, resource_room):
        """Test integration between resource room and capacitor systems."""
        # Create capacitors to represent energy sources
        capacitors = [
            CapacitorInSpace((i, 0), capacity=10.0, initial_energy=5.0)
            for i in range(3)
        ]
        
        # Agents deposit energy information
        for i, cap in enumerate(capacitors):
            agent_id = f"energy_agent_{i}"
            energy_data = {
                "position": cap.position.tolist(),
                "available_energy": cap.energy,
                "capacity": cap.capacity
            }
            resource_room.deposit(agent_id, energy_data)
        
        # Verify energy information can be retrieved
        for i in range(len(capacitors)):
            agent_id = f"energy_agent_{i}"
            retrieved = resource_room.retrieve(agent_id)
            
            assert "available_energy" in retrieved
            assert "capacity" in retrieved
            assert retrieved["available_energy"] <= retrieved["capacity"]
    
    def test_concurrent_access_simulation(self, resource_room):
        """Test simulated concurrent access to resource room."""
        import threading
        import time
        
        results = []
        
        def agent_operation(agent_id, data):
            resource_room.deposit(agent_id, data)
            time.sleep(0.001)  # Simulate processing time
            retrieved = resource_room.retrieve(agent_id)
            results.append((agent_id, retrieved))
        
        # Simulate multiple agents accessing concurrently
        threads = []
        for i in range(10):
            agent_id = f"concurrent_agent_{i}"
            data = {"id": i, "timestamp": time.time()}
            thread = threading.Thread(target=agent_operation, args=(agent_id, data))
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify all operations completed successfully
        assert len(results) == 10
        assert len(resource_room.resources) == 10