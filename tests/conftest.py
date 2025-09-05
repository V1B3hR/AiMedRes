"""
Test configuration and fixtures for duetmind_adaptive testing.
"""
import pytest
import numpy as np
from unittest.mock import Mock, MagicMock
import logging
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from labyrinth_adaptive import (
    UnifiedAdaptiveAgent, MazeMaster, NetworkMetrics, 
    ResourceRoom, AliveLoopNode, CapacitorInSpace, Memory, SocialSignal
)

# Configure logging for tests
logging.basicConfig(level=logging.WARNING)


@pytest.fixture
def sample_position():
    """Sample position coordinates."""
    return np.array([1.0, 2.0])


@pytest.fixture
def sample_velocity():
    """Sample velocity vector."""
    return np.array([0.1, 0.2])


@pytest.fixture
def alive_loop_node(sample_position, sample_velocity):
    """Create a basic AliveLoopNode for testing."""
    return AliveLoopNode(sample_position, sample_velocity, initial_energy=15.0, node_id=1)


@pytest.fixture
def resource_room():
    """Create a ResourceRoom instance for testing."""
    return ResourceRoom()


@pytest.fixture
def maze_master():
    """Create a MazeMaster instance for testing."""
    return MazeMaster()


@pytest.fixture
def network_metrics():
    """Create a NetworkMetrics instance for testing."""
    return NetworkMetrics()


@pytest.fixture
def capacitor(sample_position):
    """Create a CapacitorInSpace instance for testing."""
    return CapacitorInSpace(sample_position, capacity=10.0, initial_energy=5.0)


@pytest.fixture
def sample_agent(alive_loop_node, resource_room):
    """Create a sample UnifiedAdaptiveAgent for testing."""
    style = {"logic": 0.8, "creativity": 0.6}
    return UnifiedAdaptiveAgent("TestAgent", style, alive_loop_node, resource_room)


@pytest.fixture
def sample_agents(resource_room):
    """Create multiple agents for testing multi-agent scenarios."""
    agents = []
    styles = [
        {"logic": 0.8, "creativity": 0.5},
        {"creativity": 0.9, "analytical": 0.7},
        {"logic": 0.6, "expressiveness": 0.8}
    ]
    positions = [(0, 0), (2, 0), (0, 2)]
    velocities = [(0.5, 0), (0, 0.5), (0.3, -0.2)]
    
    for i, (style, pos, vel) in enumerate(zip(styles, positions, velocities)):
        node = AliveLoopNode(pos, vel, initial_energy=15.0 - i, node_id=i+1)
        agent = UnifiedAdaptiveAgent(f"Agent{chr(65+i)}", style, node, resource_room)
        agents.append(agent)
    
    return agents


@pytest.fixture
def sample_memory():
    """Create a sample Memory object for testing."""
    return Memory(
        content="Test memory content",
        importance=0.8,
        timestamp=12345,
        memory_type="test",
        emotional_valence=0.5
    )


@pytest.fixture
def sample_social_signal():
    """Create a sample SocialSignal for testing."""
    return SocialSignal(
        content="Test signal",
        signal_type="test",
        urgency=0.7,
        source_id=1,
        requires_response=True
    )


@pytest.fixture
def mock_logger():
    """Create a mock logger for testing."""
    return Mock(spec=logging.Logger)


# Test data generators
class TestDataGenerator:
    """Generate test data for various testing scenarios."""
    
    @staticmethod
    def generate_agent_styles(count=5):
        """Generate different agent style configurations."""
        styles = []
        attributes = ["logic", "creativity", "analytical", "expressiveness"]
        
        for i in range(count):
            style = {}
            for attr in attributes:
                if np.random.random() > 0.5:  # Random selection
                    style[attr] = np.random.uniform(0.1, 1.0)
            styles.append(style)
        
        return styles
    
    @staticmethod
    def generate_positions(count=5, range_limit=10):
        """Generate random positions for testing."""
        return [np.random.uniform(-range_limit, range_limit, 2) for _ in range(count)]
    
    @staticmethod
    def generate_tasks(count=10):
        """Generate various task strings for agent reasoning."""
        tasks = [
            "Find the exit",
            "Collaborate with other agents", 
            "Share knowledge",
            "Explore the environment",
            "Solve complex problems",
            "Navigate obstacles",
            "Communicate findings",
            "Learn from experience",
            "Adapt behavior",
            "Optimize performance"
        ]
        return tasks[:count]


@pytest.fixture
def test_data_generator():
    """Provide test data generator instance."""
    return TestDataGenerator()


# Performance testing utilities
@pytest.fixture
def performance_timer():
    """Timer utility for performance testing."""
    import time
    
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def start(self):
            self.start_time = time.time()
        
        def stop(self):
            self.end_time = time.time()
            return self.elapsed()
        
        def elapsed(self):
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return None
    
    return Timer()


# Mock factories
@pytest.fixture 
def mock_redis():
    """Mock Redis instance for testing."""
    mock = MagicMock()
    mock.get.return_value = None
    mock.set.return_value = True
    mock.delete.return_value = 1
    return mock


@pytest.fixture
def mock_flask_app():
    """Mock Flask app for API testing."""
    from flask import Flask
    app = Flask(__name__)
    app.config['TESTING'] = True
    return app