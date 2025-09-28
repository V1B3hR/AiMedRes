"""
Adaptive Neural Network Core Module

Secure, production-ready implementation of adaptive neural networks
with biological state simulation and safety monitoring.
"""

import numpy as np
import logging
import time
import threading
import json
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import uuid

from ..utils.safety import SafetyMonitor
from ..security.validation import InputValidator

logger = logging.getLogger(__name__)

class NodeState(Enum):
    """Neural node states"""
    ACTIVE = "active"
    SLEEPING = "sleeping"
    LEARNING = "learning"
    ERROR = "error"
    MAINTENANCE = "maintenance"

class MoodType(Enum):
    """Agent mood types"""
    NEUTRAL = "neutral"
    EXCITED = "excited"
    TIRED = "tired"
    FOCUSED = "focused"
    STRESSED = "stressed"

@dataclass
class BiologicalState:
    """Biological state simulation for neural nodes"""
    energy: float = 100.0
    sleep_need: float = 0.0
    stress_level: float = 0.0
    mood: MoodType = MoodType.NEUTRAL
    last_sleep: datetime = field(default_factory=datetime.now)
    sleep_debt: float = 0.0
    
    def __post_init__(self):
        """Validate initial state"""
        self.energy = max(0.0, min(100.0, self.energy))
        self.sleep_need = max(0.0, min(100.0, self.sleep_need))
        self.stress_level = max(0.0, min(100.0, self.stress_level))

@dataclass  
class NeuralNode:
    """Adaptive neural node with biological simulation"""
    node_id: str
    weights: np.ndarray
    bias: float = 0.0
    state: NodeState = NodeState.ACTIVE
    biological_state: BiologicalState = field(default_factory=BiologicalState)
    activation_history: List[float] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_active: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Initialize node with validation"""
        if len(self.activation_history) > 1000:
            self.activation_history = self.activation_history[-1000:]
    
    def activate(self, inputs: np.ndarray) -> float:
        """
        Activate node with biological state consideration
        
        Args:
            inputs: Input signals to process
            
        Returns:
            Activation output adjusted for biological state
        """
        if self.state != NodeState.ACTIVE:
            return 0.0
        
        # Calculate base activation
        raw_output = np.dot(inputs, self.weights) + self.bias
        
        # Apply biological modifications
        energy_factor = self.biological_state.energy / 100.0
        stress_factor = max(0.5, 1.0 - self.biological_state.stress_level / 200.0)
        
        # Mood affects processing
        mood_factor = self._get_mood_factor()
        
        # Final output with biological adjustments
        output = raw_output * energy_factor * stress_factor * mood_factor
        
        # Apply activation function (sigmoid)
        activated_output = 1.0 / (1.0 + np.exp(-np.clip(output, -500, 500)))
        
        # Update state
        self.activation_history.append(activated_output)
        if len(self.activation_history) > 1000:
            self.activation_history.pop(0)
        
        self.last_active = datetime.now()
        self._update_biological_state(activated_output)
        
        return activated_output
    
    def _get_mood_factor(self) -> float:
        """Get processing factor based on mood"""
        mood_factors = {
            MoodType.NEUTRAL: 1.0,
            MoodType.EXCITED: 1.2,
            MoodType.TIRED: 0.8,
            MoodType.FOCUSED: 1.1,
            MoodType.STRESSED: 0.9,
        }
        return mood_factors.get(self.biological_state.mood, 1.0)
    
    def _update_biological_state(self, activation: float):
        """Update biological state based on activity"""
        # Energy consumption based on activation
        energy_cost = abs(activation) * 0.1
        self.biological_state.energy = max(0.0, self.biological_state.energy - energy_cost)
        
        # Sleep need accumulation
        time_since_sleep = datetime.now() - self.biological_state.last_sleep
        hours_awake = time_since_sleep.total_seconds() / 3600.0
        self.biological_state.sleep_need = min(100.0, hours_awake * 5.0)
        
        # Stress from overactivation
        if activation > 0.8:
            self.biological_state.stress_level = min(100.0, self.biological_state.stress_level + 1.0)
        else:
            self.biological_state.stress_level = max(0.0, self.biological_state.stress_level - 0.5)
        
        # Update mood based on state
        self._update_mood()
    
    def _update_mood(self):
        """Update mood based on biological metrics"""
        if self.biological_state.energy < 20:
            self.biological_state.mood = MoodType.TIRED
        elif self.biological_state.stress_level > 80:
            self.biological_state.mood = MoodType.STRESSED
        elif self.biological_state.energy > 80 and self.biological_state.stress_level < 30:
            self.biological_state.mood = MoodType.EXCITED
        elif self.biological_state.sleep_need < 20:
            self.biological_state.mood = MoodType.FOCUSED
        else:
            self.biological_state.mood = MoodType.NEUTRAL
    
    def sleep(self, duration_hours: float = 8.0):
        """Put node to sleep for recovery"""
        self.state = NodeState.SLEEPING
        self.biological_state.last_sleep = datetime.now()
        
        # Recovery during sleep
        energy_recovery = min(100.0, self.biological_state.energy + duration_hours * 12.5)
        stress_reduction = max(0.0, self.biological_state.stress_level - duration_hours * 10.0)
        sleep_debt_payment = min(self.biological_state.sleep_debt, duration_hours * 10.0)
        
        self.biological_state.energy = energy_recovery
        self.biological_state.stress_level = stress_reduction
        self.biological_state.sleep_need = max(0.0, self.biological_state.sleep_need - duration_hours * 12.5)
        self.biological_state.sleep_debt = max(0.0, self.biological_state.sleep_debt - sleep_debt_payment)
        
        # Wake up refreshed
        self.state = NodeState.ACTIVE
        self.biological_state.mood = MoodType.FOCUSED
        
        logger.info(f"Node {self.node_id} completed {duration_hours}h sleep cycle")

class AdaptiveNeuralNetwork:
    """
    Production-ready adaptive neural network with security and safety features.
    
    Features:
    - Biological state simulation
    - Safety monitoring and circuit breakers
    - Secure input validation
    - Performance monitoring
    - Graceful degradation
    """
    
    def __init__(self, 
                 input_size: int = 32,
                 hidden_layers: List[int] = None,
                 output_size: int = 2,
                 learning_rate: float = 0.001,
                 enable_safety_monitoring: bool = True):
        """
        Initialize adaptive neural network
        
        Args:
            input_size: Number of input features
            hidden_layers: List of hidden layer sizes
            output_size: Number of output classes/values
            learning_rate: Learning rate for training
            enable_safety_monitoring: Enable safety and performance monitoring
        """
        self.input_size = input_size
        self.hidden_layers = hidden_layers or [64, 32, 16]
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Security and validation
        self.input_validator = InputValidator()
        
        # Safety monitoring
        if enable_safety_monitoring:
            self.safety_monitor = SafetyMonitor()
            self.safety_monitor.start()
        else:
            self.safety_monitor = None
        
        # Network architecture
        self.nodes: Dict[str, NeuralNode] = {}
        self.layers: List[List[str]] = []
        self.network_state = "initializing"
        
        # Performance metrics
        self.total_operations = 0
        self.error_count = 0
        self.last_prediction_time = None
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Initialize network
        self._initialize_network()
        
        logger.info(f"AdaptiveNeuralNetwork initialized: {input_size}→{hidden_layers}→{output_size}")
    
    def _initialize_network(self):
        """Initialize network topology with secure random weights"""
        layer_sizes = [self.input_size] + self.hidden_layers + [self.output_size]
        
        with self._lock:
            for layer_idx in range(len(layer_sizes) - 1):
                current_size = layer_sizes[layer_idx]
                next_size = layer_sizes[layer_idx + 1]
                
                layer_nodes = []
                
                for node_idx in range(next_size):
                    # Generate secure random weights
                    weights = np.random.normal(0, 0.1, current_size)
                    bias = np.random.normal(0, 0.1)
                    
                    node_id = f"layer_{layer_idx}_node_{node_idx}"
                    node = NeuralNode(
                        node_id=node_id,
                        weights=weights,
                        bias=bias
                    )
                    
                    self.nodes[node_id] = node
                    layer_nodes.append(node_id)
                
                self.layers.append(layer_nodes)
            
            self.network_state = "ready"
    
    def predict(self, inputs: np.ndarray, validate_input: bool = True) -> np.ndarray:
        """
        Make prediction with safety checks
        
        Args:
            inputs: Input data array
            validate_input: Whether to validate input data
            
        Returns:
            Prediction output array
            
        Raises:
            ValueError: If input validation fails
            RuntimeError: If network is in error state
        """
        start_time = time.time()
        
        try:
            # Input validation
            if validate_input:
                self._validate_prediction_input(inputs)
            
            # Safety check
            if self.safety_monitor and not self.safety_monitor.is_safe_to_operate():
                raise RuntimeError("Network in unsafe state - prediction blocked")
            
            # Network state check
            if self.network_state != "ready":
                raise RuntimeError(f"Network not ready: {self.network_state}")
            
            with self._lock:
                # Forward pass through network
                current_activations = inputs.copy()
                
                for layer_nodes in self.layers:
                    next_activations = []
                    
                    for node_id in layer_nodes:
                        node = self.nodes[node_id]
                        activation = node.activate(current_activations)
                        next_activations.append(activation)
                    
                    current_activations = np.array(next_activations)
                
                # Update metrics
                self.total_operations += 1
                self.last_prediction_time = datetime.now()
                
                # Safety monitoring
                if self.safety_monitor:
                    self.safety_monitor.record_operation(
                        operation="prediction",
                        duration=time.time() - start_time,
                        success=True
                    )
                
                return current_activations
                
        except Exception as e:
            self.error_count += 1
            logger.error(f"Prediction error: {e}")
            
            if self.safety_monitor:
                self.safety_monitor.record_operation(
                    operation="prediction",
                    duration=time.time() - start_time,
                    success=False,
                    error=str(e)
                )
            
            raise
    
    def _validate_prediction_input(self, inputs: np.ndarray):
        """Validate prediction input data"""
        if not isinstance(inputs, np.ndarray):
            raise ValueError("Input must be numpy array")
        
        if inputs.shape[0] != self.input_size:
            raise ValueError(f"Input size mismatch: expected {self.input_size}, got {inputs.shape[0]}")
        
        if not np.isfinite(inputs).all():
            raise ValueError("Input contains invalid values (inf/nan)")
        
        # Additional security checks
        if self.input_validator:
            self.input_validator.validate_array(inputs)
    
    def get_network_health(self) -> Dict[str, Any]:
        """Get comprehensive network health status"""
        with self._lock:
            total_nodes = len(self.nodes)
            active_nodes = sum(1 for node in self.nodes.values() if node.state == NodeState.ACTIVE)
            sleeping_nodes = sum(1 for node in self.nodes.values() if node.state == NodeState.SLEEPING)
            
            avg_energy = np.mean([node.biological_state.energy for node in self.nodes.values()])
            avg_stress = np.mean([node.biological_state.stress_level for node in self.nodes.values()])
            
            health = {
                "network_state": self.network_state,
                "total_nodes": total_nodes,
                "active_nodes": active_nodes,
                "sleeping_nodes": sleeping_nodes,
                "node_health_percentage": (active_nodes / total_nodes * 100) if total_nodes > 0 else 0,
                "average_energy": float(avg_energy),
                "average_stress": float(avg_stress),
                "total_operations": self.total_operations,
                "error_count": self.error_count,
                "error_rate": (self.error_count / max(1, self.total_operations)) * 100,
                "last_prediction": self.last_prediction_time.isoformat() if self.last_prediction_time else None,
                "uptime_seconds": (datetime.now() - self.nodes[list(self.nodes.keys())[0]].created_at).total_seconds() if self.nodes else 0
            }
            
            if self.safety_monitor:
                health["safety_status"] = self.safety_monitor.get_status()
            
            return health
    
    def sleep_cycle(self, duration_hours: float = 8.0, node_percentage: float = 0.3):
        """
        Perform network-wide sleep cycle for maintenance
        
        Args:
            duration_hours: Duration of sleep cycle
            node_percentage: Percentage of nodes to sleep (for redundancy)
        """
        with self._lock:
            nodes_to_sleep = list(self.nodes.values())
            np.random.shuffle(nodes_to_sleep)
            
            sleep_count = int(len(nodes_to_sleep) * node_percentage)
            
            for i in range(sleep_count):
                nodes_to_sleep[i].sleep(duration_hours)
            
            logger.info(f"Sleep cycle complete: {sleep_count}/{len(nodes_to_sleep)} nodes refreshed")
    
    def shutdown(self):
        """Graceful shutdown of neural network"""
        logger.info("Initiating neural network shutdown")
        
        with self._lock:
            self.network_state = "shutting_down"
            
            if self.safety_monitor:
                self.safety_monitor.stop()
            
            # Clear sensitive data
            for node in self.nodes.values():
                node.weights.fill(0)
            
            self.network_state = "shutdown"
        
        logger.info("Neural network shutdown complete")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup"""
        self.shutdown()