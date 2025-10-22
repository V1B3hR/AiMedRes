"""
Comprehensive Neural Network Testing
Advanced tests for neuralnet.py including adaptive learning, memory systems, and network topology
"""

import pytest
import numpy as np
import time
import threading
import uuid
import sys
import os
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor
import gc

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aimedres.core.cognitive_engine import (
    UnifiedAdaptiveAgent, AliveLoopNode, ResourceRoom, NetworkMetrics, 
    MazeMaster, Memory, SocialSignal, CapacitorInSpace
)


class TestMemorySystem:
    """Comprehensive testing of the memory system"""
    
    def test_memory_creation_and_properties(self):
        """Test memory object creation and property validation"""
        memory = Memory(
            content="Test memory content",
            importance=0.8,
            timestamp=1000,
            memory_type="episodic",
            emotional_valence=0.5
        )
        
        assert memory.content == "Test memory content"
        assert memory.importance == 0.8
        assert memory.timestamp == 1000
        assert memory.memory_type == "episodic"
        assert memory.emotional_valence == 0.5
        assert memory.decay_rate == 0.95
        assert memory.access_count == 0

    def test_memory_aging_mechanism(self):
        """Test memory aging and decay functionality"""
        memory = Memory(
            content="Aging test",
            importance=1.0,
            timestamp=0,
            memory_type="semantic"
        )
        
        initial_importance = memory.importance
        memory.age()
        
        # Importance should decay
        assert memory.importance < initial_importance
        assert memory.importance == initial_importance * 0.95

    def test_emotional_memory_preservation(self):
        """Test that highly emotional memories decay more slowly"""
        neutral_memory = Memory(
            content="Neutral",
            importance=1.0,
            timestamp=0,
            memory_type="episodic",
            emotional_valence=0.0
        )
        
        emotional_memory = Memory(
            content="Emotional",
            importance=1.0,
            timestamp=0,
            memory_type="episodic",
            emotional_valence=0.9
        )
        
        # Age both memories multiple times
        for _ in range(10):
            neutral_memory.age()
            emotional_memory.age()
        
        # Emotional memory should retain higher importance
        assert emotional_memory.importance > neutral_memory.importance

    def test_memory_access_tracking(self):
        """Test memory access count tracking"""
        memory = Memory(
            content="Access test",
            importance=0.7,
            timestamp=0,
            memory_type="procedural"
        )
        
        initial_access_count = memory.access_count
        
        # Simulate accessing the memory
        for _ in range(5):
            memory.access_count += 1
        
        assert memory.access_count == initial_access_count + 5


class TestSocialSignalSystem:
    """Test the social signaling system"""
    
    def test_social_signal_creation(self):
        """Test social signal object creation"""
        signal = SocialSignal(
            content="Alert message",
            signal_type="alert",
            urgency=0.8,
            source_id=123,
            requires_response=True
        )
        
        assert signal.content == "Alert message"
        assert signal.signal_type == "alert"
        assert signal.urgency == 0.8
        assert signal.source_id == 123
        assert signal.requires_response is True
        assert signal.response is None
        assert isinstance(signal.id, str)

    def test_signal_response_handling(self):
        """Test signal response mechanism"""
        signal = SocialSignal(
            content="Question",
            signal_type="query",
            urgency=0.5,
            source_id=456,
            requires_response=True
        )
        
        # Set response
        signal.response = "Answer to question"
        assert signal.response == "Answer to question"

    def test_signal_urgency_validation(self):
        """Test signal urgency levels"""
        # Test various urgency levels
        urgencies = [0.0, 0.3, 0.7, 1.0]
        
        for urgency in urgencies:
            signal = SocialSignal(
                content=f"Urgency {urgency}",
                signal_type="info",
                urgency=urgency,
                source_id=1
            )
            assert signal.urgency == urgency


class TestCapacitorInSpace:
    """Test the capacitor energy management system"""
    
    def test_capacitor_initialization(self):
        """Test capacitor creation and initialization"""
        position = [1.0, 2.0, 3.0]
        capacitor = CapacitorInSpace(
            position=position,
            capacity=10.0,
            initial_energy=5.0
        )
        
        assert np.array_equal(capacitor.position, np.array([1.0, 2.0, 3.0]))
        assert capacitor.capacity == 10.0
        assert capacitor.energy == 5.0

    def test_capacitor_charging(self):
        """Test capacitor charging mechanism"""
        capacitor = CapacitorInSpace(
            position=[0, 0, 0],
            capacity=10.0,
            initial_energy=3.0
        )
        
        # Charge within capacity
        capacitor.charge(4.0)
        assert capacitor.energy == 7.0
        
        # Attempt to overcharge
        capacitor.charge(5.0)
        assert capacitor.energy == 10.0  # Should not exceed capacity

    def test_capacitor_boundary_conditions(self):
        """Test capacitor boundary conditions"""
        # Test with zero capacity
        capacitor = CapacitorInSpace(
            position=[0, 0, 0],
            capacity=0.0,
            initial_energy=0.0
        )
        assert capacitor.capacity == 0.0
        assert capacitor.energy == 0.0
        
        # Test negative values are handled
        capacitor = CapacitorInSpace(
            position=[0, 0, 0],
            capacity=-5.0,  # Should be clamped to 0
            initial_energy=-2.0  # Should be clamped to 0
        )
        assert capacitor.capacity == 0.0
        assert capacitor.energy == 0.0


class TestAliveLoopNode:
    """Comprehensive testing of AliveLoopNode functionality"""
    
    @pytest.fixture
    def basic_node(self):
        """Create a basic node for testing"""
        return AliveLoopNode(node_id=1, position=[0, 0, 0])

    def test_node_initialization(self, basic_node):
        """Test node initialization"""
        assert basic_node.node_id == 1
        assert np.array_equal(basic_node.position, np.array([0, 0, 0]))
        assert isinstance(basic_node.memory_bank, list)
        assert basic_node.activation_level >= 0
        assert basic_node.learning_rate > 0

    def test_memory_storage_and_retrieval(self, basic_node):
        """Test node memory storage and retrieval"""
        memory = Memory(
            content="Test storage",
            importance=0.8,
            timestamp=1000,
            memory_type="test"
        )
        
        # Store memory
        basic_node.store_memory(memory)
        assert len(basic_node.memory_bank) > 0
        
        # Retrieve memory
        retrieved = basic_node.retrieve_memories(memory_type="test")
        assert len(retrieved) > 0
        assert retrieved[0].content == "Test storage"

    def test_node_adaptation(self, basic_node):
        """Test node adaptive behavior"""
        initial_learning_rate = basic_node.learning_rate
        
        # Simulate learning experience
        basic_node.adapt_learning_rate(success=True)
        
        # Learning rate should be adjusted based on success
        assert basic_node.learning_rate != initial_learning_rate

    def test_node_activation_dynamics(self, basic_node):
        """Test node activation level changes"""
        initial_activation = basic_node.activation_level
        
        # Stimulate node
        basic_node.update_activation(stimulus=0.5)
        
        # Activation should change
        assert basic_node.activation_level != initial_activation

    def test_node_energy_management(self, basic_node):
        """Test node energy consumption and management"""
        if hasattr(basic_node, 'energy_level'):
            initial_energy = basic_node.energy_level
            
            # Perform energy-consuming operation
            basic_node.process_information("Complex data")
            
            # Energy should be consumed
            assert basic_node.energy_level <= initial_energy

    def test_node_memory_consolidation(self, basic_node):
        """Test memory consolidation processes"""
        # Add multiple memories
        for i in range(10):
            memory = Memory(
                content=f"Memory {i}",
                importance=0.1 + (i * 0.1),
                timestamp=i * 100,
                memory_type="consolidation_test"
            )
            basic_node.store_memory(memory)
        
        initial_count = len(basic_node.memory_bank)
        
        # Trigger consolidation
        basic_node.consolidate_memories()
        
        # Less important memories might be removed
        assert len(basic_node.memory_bank) <= initial_count


class TestResourceRoom:
    """Test the resource management system"""
    
    @pytest.fixture
    def resource_room(self):
        """Create a resource room for testing"""
        return ResourceRoom(room_id=1, capacity=100)

    def test_resource_room_initialization(self, resource_room):
        """Test resource room creation"""
        assert resource_room.room_id == 1
        assert resource_room.capacity == 100
        assert isinstance(resource_room.resources, dict)

    def test_resource_allocation(self, resource_room):
        """Test resource allocation and deallocation"""
        # Allocate resources
        success = resource_room.allocate_resource("cpu", 50)
        assert success is True
        assert resource_room.resources.get("cpu", 0) == 50
        
        # Try to over-allocate
        success = resource_room.allocate_resource("cpu", 60)
        assert success is False  # Should fail due to capacity
        
        # Deallocate resources
        resource_room.deallocate_resource("cpu", 30)
        assert resource_room.resources.get("cpu", 0) == 20

    def test_resource_monitoring(self, resource_room):
        """Test resource usage monitoring"""
        resource_room.allocate_resource("memory", 40)
        resource_room.allocate_resource("storage", 30)
        
        total_usage = resource_room.get_total_usage()
        assert total_usage == 70
        
        utilization = resource_room.get_utilization()
        assert utilization == 0.7  # 70/100


class TestUnifiedAdaptiveAgent:
    """Comprehensive testing of the main adaptive agent"""
    
    @pytest.fixture
    def adaptive_agent(self):
        """Create an adaptive agent for testing"""
        return UnifiedAdaptiveAgent(agent_id="test_agent_001")

    def test_agent_initialization(self, adaptive_agent):
        """Test agent initialization"""
        assert adaptive_agent.agent_id == "test_agent_001"
        assert isinstance(adaptive_agent.nodes, list)
        assert len(adaptive_agent.nodes) > 0

    def test_agent_learning_process(self, adaptive_agent):
        """Test agent learning mechanisms"""
        training_data = [
            {"input": [1, 0, 1], "output": 1},
            {"input": [0, 1, 0], "output": 0},
            {"input": [1, 1, 1], "output": 1}
        ]
        
        # Train agent
        for data in training_data:
            adaptive_agent.learn(data["input"], data["output"])
        
        # Test prediction
        prediction = adaptive_agent.predict([1, 0, 1])
        assert prediction is not None

    def test_agent_memory_integration(self, adaptive_agent):
        """Test integration between agent and memory systems"""
        # Store important experience
        experience = {
            "situation": "medical_diagnosis",
            "action": "recommend_mri",
            "outcome": "positive"
        }
        
        adaptive_agent.store_experience(experience)
        
        # Retrieve similar experiences
        similar = adaptive_agent.retrieve_similar_experiences("medical_diagnosis")
        assert len(similar) > 0

    def test_agent_adaptation_mechanisms(self, adaptive_agent):
        """Test agent adaptation to new situations"""
        # Present novel situation
        novel_input = [0.5, 0.8, 0.2]
        
        # Agent should adapt its internal parameters
        initial_state = adaptive_agent.get_internal_state()
        adaptive_agent.adapt_to_situation(novel_input)
        final_state = adaptive_agent.get_internal_state()
        
        # Some internal parameters should change
        assert initial_state != final_state

    def test_agent_social_interaction(self, adaptive_agent):
        """Test agent social interaction capabilities"""
        # Create another agent for interaction
        other_agent = UnifiedAdaptiveAgent(agent_id="test_agent_002")
        
        # Send social signal
        signal = SocialSignal(
            content="collaboration_request",
            signal_type="request",
            urgency=0.6,
            source_id=adaptive_agent.agent_id
        )
        
        response = other_agent.process_social_signal(signal)
        assert response is not None

    @pytest.mark.slow
    def test_agent_long_term_adaptation(self, adaptive_agent):
        """Test long-term adaptation and learning"""
        # Simulate extended learning period
        for epoch in range(100):
            # Generate varied training data
            input_data = np.random.random(3)
            target = 1 if np.sum(input_data) > 1.5 else 0
            
            adaptive_agent.learn(input_data, target)
            
            # Periodically test adaptation
            if epoch % 20 == 0:
                test_input = [0.8, 0.7, 0.9]
                prediction = adaptive_agent.predict(test_input)
                assert prediction is not None

    def test_agent_error_recovery(self, adaptive_agent):
        """Test agent error handling and recovery"""
        # Introduce corrupted data
        corrupted_input = [float('inf'), -float('inf'), float('nan')]
        
        try:
            adaptive_agent.learn(corrupted_input, 1)
            # If no exception, agent handled gracefully
            assert True
        except Exception as e:
            # Should handle gracefully with specific exception types
            assert isinstance(e, (ValueError, OverflowError))

    def test_agent_concurrent_operations(self, adaptive_agent):
        """Test agent thread safety and concurrent operations"""
        results = []
        errors = []
        
        def concurrent_learning():
            try:
                for _ in range(10):
                    input_data = np.random.random(3)
                    target = np.random.randint(0, 2)
                    result = adaptive_agent.learn(input_data, target)
                    results.append(result)
            except Exception as e:
                errors.append(e)
        
        # Run concurrent learning
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=concurrent_learning)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Verify no errors occurred
        assert len(errors) == 0, f"Concurrent operation errors: {errors}"
        assert len(results) == 50  # 5 threads * 10 operations each


class TestNetworkTopology:
    """Test network topology and connectivity"""
    
    def test_network_creation(self):
        """Test creation of neural network topology"""
        # Create network with specific topology
        num_nodes = 10
        nodes = [AliveLoopNode(i, [i, 0, 0]) for i in range(num_nodes)]
        
        # Verify network structure
        assert len(nodes) == num_nodes
        assert all(node.node_id == i for i, node in enumerate(nodes))

    def test_network_connectivity(self):
        """Test network connectivity patterns"""
        nodes = [AliveLoopNode(i, [i, 0, 0]) for i in range(5)]
        
        # Create connections
        for i, node in enumerate(nodes):
            if i < len(nodes) - 1:
                node.connect_to(nodes[i + 1])
        
        # Verify connections
        for i in range(len(nodes) - 1):
            assert nodes[i].is_connected_to(nodes[i + 1])

    def test_network_signal_propagation(self):
        """Test signal propagation through network"""
        nodes = [AliveLoopNode(i, [0, 0, i]) for i in range(3)]
        
        # Connect nodes in sequence
        nodes[0].connect_to(nodes[1])
        nodes[1].connect_to(nodes[2])
        
        # Send signal from first node
        signal = SocialSignal(
            content="propagation_test",
            signal_type="test",
            urgency=0.7,
            source_id=0
        )
        
        nodes[0].send_signal(signal)
        
        # Verify signal propagation
        time.sleep(0.1)  # Allow propagation time
        assert nodes[2].has_received_signal(signal.id)

    def test_network_resilience(self):
        """Test network resilience to node failures"""
        nodes = [AliveLoopNode(i, [i, i, 0]) for i in range(6)]
        
        # Create mesh topology
        for i, node_a in enumerate(nodes):
            for j, node_b in enumerate(nodes):
                if i != j:
                    node_a.connect_to(node_b)
        
        # Simulate node failure
        failed_node = nodes[2]
        failed_node.set_active(False)
        
        # Test if network still functions
        signal = SocialSignal(
            content="resilience_test",
            signal_type="test",
            urgency=0.8,
            source_id=0
        )
        
        nodes[0].send_signal(signal)
        
        # Verify other nodes still receive signal
        time.sleep(0.1)
        active_nodes = [n for n in nodes if n.is_active() and n != nodes[0]]
        received_count = sum(1 for n in active_nodes if n.has_received_signal(signal.id))
        
        assert received_count > 0, "Network not resilient to node failure"


class TestNetworkMetrics:
    """Test network performance metrics"""
    
    def test_metrics_collection(self):
        """Test collection of network metrics"""
        metrics = NetworkMetrics()
        
        # Record some metrics
        metrics.record_latency(0.1)
        metrics.record_throughput(100)
        metrics.record_error_rate(0.05)
        
        assert len(metrics.latency_history) > 0
        assert len(metrics.throughput_history) > 0
        assert len(metrics.error_rate_history) > 0

    def test_metrics_analysis(self):
        """Test metrics analysis and reporting"""
        metrics = NetworkMetrics()
        
        # Record multiple data points
        latencies = [0.1, 0.15, 0.12, 0.18, 0.14]
        for latency in latencies:
            metrics.record_latency(latency)
        
        # Analyze metrics
        avg_latency = metrics.get_average_latency()
        max_latency = metrics.get_max_latency()
        
        assert avg_latency == sum(latencies) / len(latencies)
        assert max_latency == max(latencies)

    def test_performance_degradation_detection(self):
        """Test detection of performance degradation"""
        metrics = NetworkMetrics()
        
        # Record baseline performance
        for _ in range(10):
            metrics.record_latency(0.1)  # Good performance
        
        # Record degraded performance
        for _ in range(5):
            metrics.record_latency(0.5)  # Poor performance
        
        # Should detect degradation
        is_degraded = metrics.detect_performance_degradation()
        assert is_degraded is True