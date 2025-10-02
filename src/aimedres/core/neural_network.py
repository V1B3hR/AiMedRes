"""
Adaptive Neural Network Core Module

Secure, production-ready implementation of adaptive neural networks
with biological state simulation and safety monitoring.

Enhancements (2025-10):
- Expanded NodeState & MoodType semantics
- Configurable biological dynamics
- Advanced BiologicalState metrics & lifecycle management
- Mood inference engine with smoothing & weighted factors
- Stress & energy adaptive modeling
- Thread-safe mutation (optional)
"""

import numpy as np
import logging
import time
import threading
import json
from typing import Dict, List, Optional, Tuple, Any, Callable, Iterable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import uuid
import math

from ..utils.safety import SafetyMonitor
from ..security.validation import InputValidator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ENUMS WITH ENHANCED SEMANTICS
# ---------------------------------------------------------------------------

class NodeState(Enum):
    """
    Neural node operational states with semantic categories.
    
    Each state may affect:
    - Activation eligibility
    - Energy consumption scaling
    - Learning rate modulation
    - Reliability scoring
    
    New states:
    - DEGRADED: Node is operating below optimal thresholds
    - RECOVERING: Node restoring capacity after stress/sleep
    - QUARANTINED: Isolated due to detected anomaly
    - STANDBY: Available but idling for load balancing
    """
    ACTIVE = "active"
    SLEEPING = "sleeping"
    LEARNING = "learning"
    ERROR = "error"
    MAINTENANCE = "maintenance"
    DEGRADED = "degraded"
    RECOVERING = "recovering"
    QUARANTINED = "quarantined"
    STANDBY = "standby"

    @property
    def severity(self) -> int:
        """Higher value = more operational restriction."""
        mapping = {
            self.ACTIVE: 0,
            self.LEARNING: 1,
            self.STANDBY: 1,
            self.RECOVERING: 2,
            self.SLEEPING: 3,
            self.DEGRADED: 3,
            self.MAINTENANCE: 4,
            self.QUARANTINED: 5,
            self.ERROR: 6,
        }
        return mapping[self]

    @property
    def allows_activation(self) -> bool:
        """Whether forward activation should proceed."""
        return self in {
            self.ACTIVE,
            self.LEARNING,
            self.RECOVERING,
            self.STANDBY
        }

    @property
    def category(self) -> str:
        """Semantic grouping."""
        categories = {
            self.ACTIVE: "operational",
            self.LEARNING: "adaptive",
            self.RECOVERING: "restorative",
            self.STANDBY: "idle",
            self.SLEEPING: "inactive",
            self.MAINTENANCE: "service",
            self.DEGRADED: "impaired",
            self.QUARANTINED: "isolated",
            self.ERROR: "fault"
        }
        return categories[self]

    @staticmethod
    def validate_transition(src: "NodeState", dst: "NodeState") -> bool:
        """Constrain state transitions to reduce invalid operational drift."""
        allowed: Dict[NodeState, Iterable[NodeState]] = {
            NodeState.ACTIVE: {
                NodeState.LEARNING, NodeState.SLEEPING, NodeState.MAINTENANCE,
                NodeState.DEGRADED, NodeState.ERROR, NodeState.STANDBY
            },
            NodeState.LEARNING: {
                NodeState.ACTIVE, NodeState.DEGRADED, NodeState.RECOVERING,
                NodeState.ERROR, NodeState.STANDBY
            },
            NodeState.SLEEPING: {NodeState.RECOVERING, NodeState.MAINTENANCE},
            NodeState.RECOVERING: {NodeState.ACTIVE, NodeState.LEARNING, NodeState.STANDBY},
            NodeState.MAINTENANCE: {NodeState.RECOVERING, NodeState.STANDBY},
            NodeState.DEGRADED: {NodeState.RECOVERING, NodeState.MAINTENANCE, NodeState.ERROR},
            NodeState.QUARANTINED: {NodeState.MAINTENANCE, NodeState.ERROR},
            NodeState.STANDBY: {NodeState.ACTIVE, NodeState.LEARNING, NodeState.SLEEPING},
            NodeState.ERROR: {NodeState.MAINTENANCE, NodeState.QUARANTINED}
        }
        return dst in allowed.get(src, [])


class MoodType(Enum):
    """
    Agent / node mood spectrum affecting processing modulation.

    Added nuanced moods:
    - CALM: Stable & low noise
    - ANXIOUS: High stress forecasting / overcautious
    - OVERLOADED: Capacity saturation
    - DEPRESSED: Energy & motivation collapse (long stress + low energy)
    - FLOW: Optimal performance state (high focus, low stress)
    """
    NEUTRAL = "neutral"
    EXCITED = "excited"
    TIRED = "tired"
    FOCUSED = "focused"
    STRESSED = "stressed"
    CALM = "calm"
    ANXIOUS = "anxious"
    OVERLOADED = "overloaded"
    DEPRESSED = "depressed"
    FLOW = "flow"

    @property
    def processing_modifier(self) -> float:
        """Multiplier applied to base activation pre-nonlinearity."""
        modifiers = {
            self.NEUTRAL: 1.00,
            self.EXCITED: 1.15,
            self.TIRED: 0.82,
            self.FOCUSED: 1.10,
            self.STRESSED: 0.92,
            self.CALM: 1.05,
            self.ANXIOUS: 0.90,
            self.OVERLOADED: 0.75,
            self.DEPRESSED: 0.70,
            self.FLOW: 1.20
        }
        return modifiers[self]

    @property
    def stability_bias(self) -> float:
        """How stable / noisy outputs might become (higher = more stable)."""
        biases = {
            self.NEUTRAL: 1.0,
            self.CALM: 1.1,
            self.FLOW: 1.15,
            self.FOCUSED: 1.1,
            self.EXCITED: 0.95,
            self.TIRED: 0.9,
            self.STRESSED: 0.85,
            self.ANXIOUS: 0.8,
            self.OVERLOADED: 0.75,
            self.DEPRESSED: 0.7
        }
        return biases[self]


# ---------------------------------------------------------------------------
# CONFIGURATION STRUCTURES
# ---------------------------------------------------------------------------

@dataclass
class BiologicalConfig:
    """Centralized tunable constants for biological simulation."""
    max_energy: float = 100.0
    max_stress: float = 100.0
    max_sleep_need: float = 100.0
    max_sleep_debt: float = 200.0

    energy_activation_cost: float = 0.1
    stress_overactivation_increment: float = 1.0
    stress_relief_passive: float = 0.4
    stress_relief_sleep_hour: float = 10.0

    sleep_need_rate_per_hour: float = 5.0
    sleep_need_recovery_per_hour: float = 12.5
    sleep_debt_payment_rate: float = 10.0

    circadian_period_hours: float = 24.0
    circadian_amplitude: float = 8.0  # affects baseline energy modulation

    fatigue_energy_threshold: float = 25.0
    overload_activation_threshold: float = 0.9
    depressed_energy_threshold: float = 15.0
    chronic_stress_threshold: float = 70.0
    flow_energy_min: float = 70.0
    flow_stress_max: float = 25.0

    mood_smoothing_factor: float = 0.6  # EMA smoothing on mood inference
    risk_stress_weight: float = 0.6
    risk_debt_weight: float = 0.3
    risk_energy_inverse_weight: float = 0.4

    def clamp(self, value: float, max_val: float) -> float:
        return max(0.0, min(max_val, value))


# ---------------------------------------------------------------------------
# BIOLOGICAL STATE
# ---------------------------------------------------------------------------

@dataclass
class BiologicalState:
    """
    Biological state simulation with advanced homeostasis modeling.

    Added:
    - Config-driven dynamics
    - Derived metrics (fatigue_index, homeostasis_score, risk_score)
    - Mood inference with smoothing
    - Circadian modulation
    - Stress adaptation (acute vs chronic)
    - Serialization
    - Thread-safe optional lock
    """
    energy: float = 100.0
    sleep_need: float = 0.0
    stress_level: float = 0.0
    mood: MoodType = MoodType.NEUTRAL
    last_sleep: datetime = field(default_factory=datetime.now)
    sleep_debt: float = 0.0

    # New fields
    config: BiologicalConfig = field(default_factory=BiologicalConfig)
    mood_history: List[MoodType] = field(default_factory=list)
    last_update: datetime = field(default_factory=datetime.now)
    chronic_stress_exposure_hours: float = 0.0
    lock: Optional[threading.RLock] = field(default=None, repr=False, compare=False)

    # Cached derived metrics
    fatigue_index: float = 0.0
    performance_modifier: float = 1.0
    homeostasis_score: float = 1.0
    risk_score: float = 0.0

    def __post_init__(self):
        self._clamp_all()
        if self.lock is None:
            self.lock = threading.RLock()

    # ------------------------- PUBLIC INTERFACE ---------------------------- #

    def update_after_activation(self, activation: float):
        """Update biological state after node activation."""
        with self.lock:
            cfg = self.config
            # Energy cost
            self.energy -= abs(activation) * cfg.energy_activation_cost
            # Stress modulation
            if activation > cfg.overload_activation_threshold:
                self.stress_level += cfg.stress_overactivation_increment
            else:
                self.stress_level -= cfg.stress_relief_passive
            self._apply_time_drift()
            self._clamp_all()
            self._infer_mood()
            self._update_derived_metrics()

    def apply_sleep(self, duration_hours: float):
        """Recover via sleep with diminishing returns if excessive."""
        with self.lock:
            cfg = self.config
            base_recovery = duration_hours * cfg.sleep_need_recovery_per_hour
            efficiency = self._sleep_efficiency_factor()
            self.energy += base_recovery * efficiency
            self.stress_level -= duration_hours * cfg.stress_relief_sleep_hour * (0.8 + 0.2 * efficiency)
            self.sleep_need -= base_recovery
            debt_payment = min(self.sleep_debt, duration_hours * cfg.sleep_debt_payment_rate * efficiency)
            self.sleep_debt -= debt_payment
            self.last_sleep = datetime.now()
            self._clamp_all()
            self._infer_mood(force=True)
            self._update_derived_metrics()

    def decay(self):
        """Periodic maintenance update (e.g., call every X seconds)."""
        with self.lock:
            self._apply_time_drift()
            self._infer_mood()
            self._update_derived_metrics()

    def to_dict(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "energy": self.energy,
                "sleep_need": self.sleep_need,
                "stress_level": self.stress_level,
                "mood": self.mood.value,
                "last_sleep": self.last_sleep.isoformat(),
                "sleep_debt": self.sleep_debt,
                "fatigue_index": self.fatigue_index,
                "performance_modifier": self.performance_modifier,
                "homeostasis_score": self.homeostasis_score,
                "risk_score": self.risk_score,
                "chronic_stress_exposure_hours": self.chronic_stress_exposure_hours
            }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], config: Optional[BiologicalConfig] = None) -> "BiologicalState":
        obj = cls(
            energy=data.get("energy", 100.0),
            sleep_need=data.get("sleep_need", 0.0),
            stress_level=data.get("stress_level", 0.0),
            mood=MoodType(data.get("mood", "neutral")),
            last_sleep=datetime.fromisoformat(data.get("last_sleep")) if data.get("last_sleep") else datetime.now(),
            sleep_debt=data.get("sleep_debt", 0.0),
            config=config or BiologicalConfig()
        )
        obj.decay()  # initialize derived metrics
        return obj

    # ------------------------- INTERNAL LOGIC ----------------------------- #

    def _apply_time_drift(self):
        """Accumulate sleep need, stress chronic exposure, and circadian modulation."""
        now = datetime.now()
        elapsed = (now - self.last_update).total_seconds()
        if elapsed <= 0:
            return
        hours = elapsed / 3600.0
        cfg = self.config

        # Sleep need accumulation
        self.sleep_need += hours * cfg.sleep_need_rate_per_hour
        # Sleep debt grows if sleep_need stays high
        if self.sleep_need > 70:
            self.sleep_debt += hours * (self.sleep_need - 70) * 0.5

        # Chronic stress tracking
        if self.stress_level > cfg.chronic_stress_threshold:
            self.chronic_stress_exposure_hours += hours
        else:
            # Small recovery of chronic load
            self.chronic_stress_exposure_hours = max(
                0.0, self.chronic_stress_exposure_hours - hours * 0.5
            )

        # Circadian energy modulation (sinusoidal baseline)
        circadian_phase = (now.timestamp() / 3600.0) % cfg.circadian_period_hours
        phase_angle = (2 * math.pi * circadian_phase) / cfg.circadian_period_hours
        circadian_adjustment = math.sin(phase_angle) * (cfg.circadian_amplitude / 24.0)
        self.energy += circadian_adjustment

        self.last_update = now

    def _sleep_efficiency_factor(self) -> float:
        """Dynamic recovery efficiency based on debt and stress."""
        # Higher debt => stronger recovery, but cap to prevent runaway
        debt_factor = 1.0 + min(0.8, self.sleep_debt / (self.config.max_sleep_debt * 1.5))
        stress_penalty = 1.0 - min(0.4, self.stress_level / (self.config.max_stress * 2.5))
        return max(0.3, debt_factor * stress_penalty)

    def _infer_mood(self, force: bool = False):
        """Weighted mood inference using current physiological metrics."""
        cfg = self.config

        # Build candidate scores
        energy = self.energy
        stress = self.stress_level
        need = self.sleep_need
        debt = self.sleep_debt
        chronic = self.chronic_stress_exposure_hours

        scores: Dict[MoodType, float] = {}

        # Neutral baseline
        scores[MoodType.NEUTRAL] = 0.5

        # Flow: high energy, low stress, manageable sleep need
        scores[MoodType.FLOW] = max(0.0,
            (energy - cfg.flow_energy_min) / (cfg.max_energy - cfg.flow_energy_min + 1e-6)
            * (1.0 - stress / (cfg.flow_stress_max + 1e-6))
            * (1.0 - (need / cfg.max_sleep_need) * 0.5)
        )

        # Tired: high need OR low energy
        scores[MoodType.TIRED] = max(
            need / cfg.max_sleep_need,
            (cfg.fatigue_energy_threshold - energy) / cfg.fatigue_energy_threshold
        )

        # Stressed
        scores[MoodType.STRESSED] = stress / cfg.max_stress

        # Excited: high energy + mid stress (arousal zone)
        arousal = energy / cfg.max_energy * (0.4 + 0.6 * (stress / cfg.max_stress))
        scores[MoodType.EXCITED] = max(0.0, arousal - 0.3)

        # Focused: moderate-high energy, low-mid stress, low sleep_need
        scores[MoodType.FOCUSED] = (
            (energy / cfg.max_energy) * (1.0 - stress / (cfg.max_stress * 1.2)) *
            (1.0 - (need / cfg.max_sleep_need) * 0.7)
        )

        # Calm: low stress + stable energy
        scores[MoodType.CALM] = (1.0 - stress / cfg.max_stress) * (0.5 + 0.5 * energy / cfg.max_energy)

        # Anxious: rising stress + moderate-high sleep debt
        scores[MoodType.ANXIOUS] = (stress / cfg.max_stress) * (0.3 + 0.7 * (debt / (cfg.max_sleep_debt + 1e-6)))

        # Overloaded: stress + high activation proxies (approx by stress + need)
        scores[MoodType.OVERLOADED] = min(1.0, (stress / cfg.max_stress) * 0.7 + (need / cfg.max_sleep_need) * 0.5)

        # Depressed: chronic stress + low energy + high debt
        scores[MoodType.DEPRESSED] = (
            (max(0.0, cfg.depressed_energy_threshold - energy) / cfg.depressed_energy_threshold) *
            (chronic / 24.0) * 0.6 +
            (debt / cfg.max_sleep_debt) * 0.4
        )

        # Normalize and pick
        # Smooth with exponential moving average on prior mood distribution
        raw_items = list(scores.items())
        values = np.array([v for _, v in raw_items])
        if values.sum() > 0:
            values = values / (values.sum() + 1e-9)
        mood_candidates = {mt: float(val) for (mt, _), val in zip(raw_items, values)}

        selected = max(mood_candidates.items(), key=lambda kv: kv[1])[0]

        if not force and self.mood_history:
            # Smooth via categorical inertia
            if selected != self.mood:
                # Only switch if signal strong
                if mood_candidates[selected] < 0.35 and mood_candidates.get(self.mood, 0) > 0.25:
                    selected = self.mood

        # Apply smoothing over historical moods
        self.mood_history.append(selected)
        if len(self.mood_history) > 25:
            self.mood_history.pop(0)

        self.mood = selected

    def _update_derived_metrics(self):
        """Compute composite indicators."""
        cfg = self.config
        energy_norm = self.energy / cfg.max_energy
        stress_norm = self.stress_level / cfg.max_stress
        need_norm = self.sleep_need / cfg.max_sleep_need
        debt_norm = self.sleep_debt / cfg.max_sleep_debt

        # Fatigue index: combination of low energy + high sleep need/debt
        self.fatigue_index = float(
            (1 - energy_norm) * 0.5 + need_norm * 0.3 + debt_norm * 0.2
        )

        # Homeostasis: inverse disparity of key metrics
        dispersion = np.std([energy_norm, 1 - stress_norm, 1 - need_norm])
        self.homeostasis_score = float(max(0.0, 1.0 - dispersion))

        # Performance modifier from mood + fatigue penalty
        self.performance_modifier = float(
            self.mood.processing_modifier *
            (1.0 - 0.4 * self.fatigue_index) *
            max(0.6, self.homeostasis_score)
        )

        # Risk score (higher = risk of degradation)
        self.risk_score = float(
            stress_norm * cfg.risk_stress_weight +
            debt_norm * cfg.risk_debt_weight +
            (1 - energy_norm) * cfg.risk_energy_inverse_weight
        )

    def _clamp_all(self):
        cfg = self.config
        self.energy = cfg.clamp(self.energy, cfg.max_energy)
        self.sleep_need = cfg.clamp(self.sleep_need, cfg.max_sleep_need)
        self.stress_level = cfg.clamp(self.stress_level, cfg.max_stress)
        self.sleep_debt = cfg.clamp(self.sleep_debt, cfg.max_sleep_debt)
        self.chronic_stress_exposure_hours = max(0.0, self.chronic_stress_exposure_hours)


# ---------------------------------------------------------------------------
# NEURAL NODE WITH INTEGRATION OF ENHANCED BIOLOGICAL STATE
# ---------------------------------------------------------------------------

@dataclass
class NeuralNode:
    """Adaptive neural node with biological simulation."""
    node_id: str
    weights: np.ndarray
    bias: float = 0.0
    state: NodeState = NodeState.ACTIVE
    biological_state: BiologicalState = field(default_factory=BiologicalState)
    activation_history: List[float] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_active: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        if len(self.activation_history) > 2000:
            self.activation_history = self.activation_history[-2000:]

    def set_state(self, new_state: NodeState):
        if not NodeState.validate_transition(self.state, new_state):
            logger.warning(f"Illegal state transition {self.state} -> {new_state} for {self.node_id}")
            return
        self.state = new_state

    def activate(self, inputs: np.ndarray) -> float:
        if not self.state.allows_activation:
            return 0.0

        raw_output = float(np.dot(inputs, self.weights) + self.bias)

        bio = self.biological_state
        mood_factor = bio.mood.processing_modifier
        performance_factor = bio.performance_modifier
        energy_factor = (bio.energy / bio.config.max_energy)
        stress_factor = max(0.4, 1.0 - (bio.stress_level / (bio.config.max_stress * 1.8)))

        composite = raw_output * mood_factor * performance_factor * energy_factor * stress_factor
        activated_output = 1.0 / (1.0 + np.exp(-np.clip(composite, -500, 500)))

        self.activation_history.append(activated_output)
        if len(self.activation_history) > 2000:
            self.activation_history.pop(0)

        self.last_active = datetime.now()
        bio.update_after_activation(activated_output)

        # Auto degrade / recover based on risk
        if bio.risk_score > 1.2 and self.state == NodeState.ACTIVE:
            self.set_state(NodeState.DEGRADED)
        elif bio.risk_score < 0.6 and self.state in {NodeState.DEGRADED, NodeState.RECOVERING}:
            self.set_state(NodeState.ACTIVE)

        return activated_output

    def sleep(self, duration_hours: float = 8.0):
        self.set_state(NodeState.SLEEPING)
        self.biological_state.apply_sleep(duration_hours)
        self.set_state(NodeState.RECOVERING)
        # Short stabilization window
        self.set_state(NodeState.ACTIVE)
        self.biological_state.mood = MoodType.FOCUSED
        logger.info(f"Node {self.node_id} completed {duration_hours}h sleep cycle")


# ---------------------------------------------------------------------------
# ADAPTIVE NETWORK (TRIMMED TO FOCUS ON ENHANCEMENTS)
# ---------------------------------------------------------------------------

class AdaptiveNeuralNetwork:
    """
    Production-ready adaptive neural network with security and safety features.
    """

    def __init__(self,
                 input_size: int = 32,
                 hidden_layers: List[int] = None,
                 output_size: int = 2,
                 learning_rate: float = 0.001,
                 enable_safety_monitoring: bool = True,
                 biological_config: Optional[BiologicalConfig] = None):
        self.input_size = input_size
        self.hidden_layers = hidden_layers or [64, 32, 16]
        self.output_size = output_size
        self.learning_rate = learning_rate

        self.input_validator = InputValidator()

        if enable_safety_monitoring:
            self.safety_monitor = SafetyMonitor()
            self.safety_monitor.start()
        else:
            self.safety_monitor = None

        self.nodes: Dict[str, NeuralNode] = {}
        self.layers: List[List[str]] = []
        self.network_state = "initializing"

        self.total_operations = 0
        self.error_count = 0
        self.last_prediction_time = None

        self._lock = threading.RLock()
        self.biological_config = biological_config or BiologicalConfig()

        self._initialize_network()

        logger.info(f"AdaptiveNeuralNetwork initialized: {input_size}→{self.hidden_layers}→{output_size}")

    def _initialize_network(self):
        layer_sizes = [self.input_size] + self.hidden_layers + [self.output_size]
        with self._lock:
            for layer_idx in range(len(layer_sizes) - 1):
                current_size = layer_sizes[layer_idx]
                next_size = layer_sizes[layer_idx + 1]
                layer_nodes = []
                for node_idx in range(next_size):
                    weights = np.random.normal(0, 0.1, current_size)
                    bias = np.random.normal(0, 0.1)
                    node_id = f"layer_{layer_idx}_node_{node_idx}"
                    node = NeuralNode(
                        node_id=node_id,
                        weights=weights,
                        bias=bias,
                        biological_state=BiologicalState(config=self.biological_config)
                    )
                    self.nodes[node_id] = node
                    layer_nodes.append(node_id)
                self.layers.append(layer_nodes)
            self.network_state = "ready"

    def predict(self, inputs: np.ndarray, validate_input: bool = True) -> np.ndarray:
        start_time = time.time()
        try:
            if validate_input:
                self._validate_prediction_input(inputs)
            if self.safety_monitor and not self.safety_monitor.is_safe_to_operate():
                raise RuntimeError("Network in unsafe state - prediction blocked")
            if self.network_state != "ready":
                raise RuntimeError(f"Network not ready: {self.network_state}")

            with self._lock:
                current_activations = inputs.copy()
                for layer_nodes in self.layers:
                    next_activations = []
                    for node_id in layer_nodes:
                        node = self.nodes[node_id]
                        next_activations.append(node.activate(current_activations))
                    current_activations = np.array(next_activations)

                self.total_operations += 1
                self.last_prediction_time = datetime.now()

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
        if not isinstance(inputs, np.ndarray):
            raise ValueError("Input must be numpy array")
        if inputs.shape[0] != self.input_size:
            raise ValueError(f"Input size mismatch: expected {self.input_size}, got {inputs.shape[0]}")
        if not np.isfinite(inputs).all():
            raise ValueError("Input contains invalid values (inf/nan)")
        if self.input_validator:
            self.input_validator.validate_array(inputs)

    def get_network_health(self) -> Dict[str, Any]:
        with self._lock:
            total_nodes = len(self.nodes)
            active_nodes = sum(1 for n in self.nodes.values() if n.state == NodeState.ACTIVE)
            sleeping_nodes = sum(1 for n in self.nodes.values() if n.state == NodeState.SLEEPING)
            degraded_nodes = sum(1 for n in self.nodes.values() if n.state == NodeState.DEGRADED)

            avg_energy = np.mean([n.biological_state.energy for n in self.nodes.values()])
            avg_stress = np.mean([n.biological_state.stress_level for n in self.nodes.values()])
            avg_risk = np.mean([n.biological_state.risk_score for n in self.nodes.values()])
            avg_perf = np.mean([n.biological_state.performance_modifier for n in self.nodes.values()])

            health = {
                "network_state": self.network_state,
                "total_nodes": total_nodes,
                "active_nodes": active_nodes,
                "sleeping_nodes": sleeping_nodes,
                "degraded_nodes": degraded_nodes,
                "node_health_percentage": (active_nodes / total_nodes * 100) if total_nodes else 0,
                "average_energy": float(avg_energy),
                "average_stress": float(avg_stress),
                "average_risk": float(avg_risk),
                "average_performance_modifier": float(avg_perf),
                "total_operations": self.total_operations,
                "error_count": self.error_count,
                "error_rate": (self.error_count / max(1, self.total_operations)) * 100,
                "last_prediction": self.last_prediction_time.isoformat() if self.last_prediction_time else None,
                "uptime_seconds": (datetime.now() - self.nodes[list(self.nodes.keys())[0]].created_at).total_seconds() if self.nodes else 0
            }
            if self.safety_monitor:
                health["safety_status"] = self.safety_monitor.get_status()
            return health

    def sleep_cycle(self, duration_hours: float = 6.0, node_percentage: float = 0.25):
        with self._lock:
            nodes = list(self.nodes.values())
            np.random.shuffle(nodes)
            count = int(len(nodes) * node_percentage)
            for node in nodes[:count]:
                node.sleep(duration_hours)
            logger.info(f"Sleep cycle complete: {count}/{len(nodes)} nodes refreshed")

    def shutdown(self):
        logger.info("Initiating neural network shutdown")
        with self._lock:
            self.network_state = "shutting_down"
            if self.safety_monitor:
                self.safety_monitor.stop()
            for node in self.nodes.values():
                node.weights.fill(0)
            self.network_state = "shutdown"
        logger.info("Neural network shutdown complete")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
