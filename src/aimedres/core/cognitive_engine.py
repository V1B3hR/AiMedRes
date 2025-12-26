"""
Enhanced cognitive-agent simulation components.

Key improvements:
- Configurable dataclasses for clarity and testability.
- Advanced memory handling (query, prune, promotion, adaptive decay).
- Plugin-based intervention architecture for MazeMaster.
- Rolling metrics and health analytics.
- Structured logging and optional tracing.
- Stronger type hints and docstrings.
"""

from __future__ import annotations

import logging
import math
import random
import threading
import uuid
from collections import Counter, deque
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Protocol,
    Tuple,
)

import numpy as np

# ======================================================================================
# Logging
# ======================================================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("UnifiedLabyrinth")

# ======================================================================================
# Configuration Dataclasses
# ======================================================================================


@dataclass
class MemoryConfig:
    max_short_term: int = 1000
    working_memory_limit: int = 7
    decay_rate_base: float = 0.95
    emotional_decay_bonus_threshold: float = 0.7
    emotional_decay_rate_max: float = 0.997
    emotional_decay_bonus_scale: float = 0.1
    promotion_importance_threshold: float = 0.78
    max_long_term_per_key: int = 10
    adaptive_decay_by_access: bool = True
    access_decay_penalty: float = 0.002  # Additional decay per access (if adaptive)


@dataclass
class AgentConfig:
    confusion_increase_step: float = 0.1
    confusion_decrease_step: float = 0.1
    low_confidence_threshold: float = 0.4
    high_confidence_threshold: float = 0.8
    style_influence_threshold: float = 0.7
    entropy_smoothing: float = 0.15  # smoothing factor for incremental entropy updates
    enable_thread_safety: bool = False
    enable_tracing: bool = False


@dataclass
class InterventionPolicy:
    confusion_escape_thresh: float = 0.85
    entropy_escape_thresh: float = 1.5
    soft_advice_thresh: float = 0.65


# ======================================================================================
# Memory Model
# ======================================================================================


@dataclass
class Memory:
    content: Any
    importance: float
    timestamp: int
    memory_type: str
    emotional_valence: float = 0.0
    decay_rate: float = 0.95
    access_count: int = 0
    source_node: Optional[int] = None
    validation_count: int = 0

    def age(self, config: MemoryConfig):
        """
        Apply time-based decay with emotional modulation and optional access-based penalty.
        """
        original_decay = self.decay_rate

        if abs(self.emotional_valence) > config.emotional_decay_bonus_threshold:
            # Increase persistence when strong valence
            bonus = (
                abs(self.emotional_valence) - config.emotional_decay_bonus_threshold
            ) * config.emotional_decay_bonus_scale
            self.decay_rate = min(config.emotional_decay_rate_max, self.decay_rate + bonus)

        if config.adaptive_decay_by_access and self.access_count > 5:
            # Mild penalty for very frequently accessed memories to encourage diversity
            self.decay_rate = max(
                0.01, self.decay_rate - (self.access_count * config.access_decay_penalty)
            )

        self.importance *= self.decay_rate
        logger.debug(
            f"[Memory] Aged memory '{self.memory_type}' | importance: {self.importance:.4f}, "
            f"decay changed {original_decay:.4f}->{self.decay_rate:.4f}"
        )

    def validate(self, increment: int = 1):
        self.validation_count += increment

    def mark_accessed(self):
        self.access_count += 1


# ======================================================================================
# Social Signal
# ======================================================================================


class SocialSignal:
    def __init__(
        self,
        content: Any,
        signal_type: str,
        urgency: float,
        source_id: int,
        requires_response: bool = False,
    ):
        self.id = str(uuid.uuid4())
        self.content = content
        self.signal_type = signal_type
        self.urgency = urgency
        self.source_id = source_id
        self.timestamp = 0
        self.requires_response = requires_response
        self.response: Optional[Any] = None


# ======================================================================================
# Capacitor
# ======================================================================================


class CapacitorInSpace:
    """
    A spatial energy storage element.
    """

    def __init__(
        self, position: Iterable[float], capacity: float = 5.0, initial_energy: float = 0.0
    ):
        self.position = np.array(list(position), dtype=float)
        self.capacity = max(0.0, capacity)
        self.energy = min(max(0.0, initial_energy), self.capacity)

    def charge(self, amount: float):
        if amount < 0:
            raise ValueError("Charge amount must be non-negative.")
        before = self.energy
        self.energy = min(self.capacity, self.energy + amount)
        logger.debug(f"[Capacitor] Charged {amount:.2f} | {before:.2f}->{self.energy:.2f}")

    def discharge(self, amount: float):
        if amount < 0:
            raise ValueError("Discharge amount must be non-negative.")
        before = self.energy
        self.energy = max(0.0, self.energy - amount)
        logger.debug(f"[Capacitor] Discharged {amount:.2f} | {before:.2f}->{self.energy:.2f}")

    def status(self) -> str:
        return f"Capacitor: Position {self.position.tolist()}, Energy {round(self.energy, 2)}/{self.capacity}"


# ======================================================================================
# Alive Loop Node
# ======================================================================================


class AliveLoopNode:
    """
    Represents a 'living' node with a simple internal reasoning tick.
    """

    sleep_stages = ["light", "REM", "deep"]

    def __init__(
        self,
        position: Iterable[float],
        velocity: Iterable[float],
        initial_energy: float = 10.0,
        field_strength: float = 1.0,
        node_id: int = 0,
        memory_config: Optional[MemoryConfig] = None,
    ):
        self.position = np.array(list(position), dtype=float)
        self.velocity = np.array(list(velocity), dtype=float)
        self.energy = max(0.0, initial_energy)
        self.field_strength = field_strength
        self.radius = max(0.1, 0.1 + 0.05 * initial_energy)
        self.phase = "active"
        self.node_id = node_id
        self.anxiety = 0.0
        self.phase_history: deque[str] = deque(maxlen=24)

        # Memory systems
        self.memory: deque[Memory] = deque(
            maxlen=(memory_config.max_short_term if memory_config else 1000)
        )
        self.working_memory: deque[Memory] = deque(
            maxlen=(memory_config.working_memory_limit if memory_config else 7)
        )
        self.long_term_memory: Dict[str, List[Memory]] = {}

        self.communication_queue: deque[SocialSignal] = deque(maxlen=20)
        self.trust_network: Dict[str, float] = {}
        self.influence_network: Dict[str, float] = {}
        self.emotional_state: Dict[str, float] = {"valence": 0.0}
        self._time = 0
        self.communication_style = {"directness": 0.7, "formality": 0.3, "expressiveness": 0.6}
        self.confusion_level = 0.0
        self.loop_counter = 0
        self._memory_config = memory_config or MemoryConfig()

    def safe_think(
        self,
        agent_name: str,
        task: str,
        temperature: float = 0.6,
        max_tokens: int = 128,
        trace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Simulates a reasoning step with randomized confidence.
        """
        self._time += 1
        if not task:
            self.confusion_level += 0.1
            return {"error": "Empty task", "confidence": 0.0}

        # The 'temperature' and 'max_tokens' are placeholders for future LLM integration.
        random.seed(self._time + hash(task) + int(temperature * 100))
        confidence = random.uniform(0.3, 0.95)
        if confidence < 0.4:
            self.confusion_level += 0.05
        elif confidence > 0.8:
            self.confusion_level = max(0, self.confusion_level - 0.05)

        result = {
            "agent": agent_name,
            "task": task,
            "insight": f"{agent_name} reasoned (temp={temperature}): {task}",
            "confidence": confidence,
            "energy": self.energy,
            "confusion_level": self.confusion_level,
            "trace_id": trace_id,
        }

        mem = Memory(
            content=task,
            importance=confidence,
            timestamp=self._time,
            memory_type="prediction",
            emotional_valence=random.uniform(-1, 1),
            decay_rate=self._memory_config.decay_rate_base,
        )
        self.memory.append(mem)
        self._promote_if_important(mem)
        return result

    def move(self):
        before = self.position.copy()
        self.position += self.velocity
        logger.debug(f"[AliveLoopNode] Moved {before.tolist()} -> {self.position.tolist()}")

    def age_memories(self):
        for mem in list(self.memory):
            mem.age(self._memory_config)

    def query_memory(
        self,
        predicate: Optional[Callable[[Memory], bool]] = None,
        top_k: int = 5,
        sort_key: Callable[[Memory], float] = lambda m: m.importance,
        reverse: bool = True,
    ) -> List[Memory]:
        """
        Retrieve memories filtered by predicate and sorted by provided key.
        """
        if predicate is None:
            predicate = lambda m: True
        filtered = [m for m in self.memory if predicate(m)]
        filtered.sort(key=sort_key, reverse=reverse)
        return filtered[:top_k]

    def prune_memory(self, min_importance: float = 0.05):
        """
        Remove very low-importance memories to free capacity.
        """
        original_len = len(self.memory)
        kept = [m for m in self.memory if m.importance >= min_importance]
        self.memory.clear()
        self.memory.extend(kept)
        logger.debug(f"[AliveLoopNode] Pruned memory {original_len}->{len(kept)}")

    def _promote_if_important(self, mem: Memory):
        if mem.importance >= self._memory_config.promotion_importance_threshold:
            key = mem.memory_type
            bucket = self.long_term_memory.setdefault(key, [])
            bucket.append(mem)
            # Keep size bounded
            if len(bucket) > self._memory_config.max_long_term_per_key:
                bucket.sort(key=lambda m: m.importance, reverse=True)
                del bucket[self._memory_config.max_long_term_per_key :]


# ======================================================================================
# Shared Resource Room
# ======================================================================================


class ResourceRoom:
    def __init__(self):
        self.resources: Dict[str, Dict[str, Any]] = {}

    def deposit(self, agent_id: str, info: Dict[str, Any]):
        self.resources[agent_id] = info

    def retrieve(self, agent_id: str) -> Dict[str, Any]:
        return self.resources.get(agent_id, {})


# ======================================================================================
# Network Metrics & Analytics
# ======================================================================================


class NetworkMetrics:
    """
    Tracks global aggregates and derives a health score.
    """

    def __init__(self, rolling_window: int = 100):
        self.energy_history: deque[float] = deque(maxlen=1000)
        self.confusion_history: deque[float] = deque(maxlen=1000)
        self.agent_statuses: List[str] = []
        self._rolling_window = rolling_window

    def update(self, agents: List["UnifiedAdaptiveAgent"]):
        if not agents:
            self.energy_history.append(0.0)
            self.confusion_history.append(0.0)
            self.agent_statuses = []
            return

        total_energy = sum(a.alive_node.energy for a in agents)
        avg_confusion = float(np.mean([a.confusion_level for a in agents])) if agents else 0.0
        self.energy_history.append(total_energy)
        self.confusion_history.append(avg_confusion)
        self.agent_statuses = [a.status for a in agents]

    def _moving_average(self, data: deque[float]) -> float:
        if not data:
            return 0.0
        window = list(data)[-self._rolling_window :]
        return sum(window) / len(window)

    def health_score(self) -> float:
        if not self.energy_history:
            return 0.5
        avg_energy = self._moving_average(self.energy_history)
        avg_confusion = self._moving_average(self.confusion_history)
        score = 0.5 * (min(avg_energy / 100, 1.0) + max(0, 1.0 - avg_confusion))
        return round(score, 3)

    def confusion_derivative(self) -> float:
        if len(self.confusion_history) < 2:
            return 0.0
        return self.confusion_history[-1] - self.confusion_history[-2]


# ======================================================================================
# MazeMaster Intervention Plugins
# ======================================================================================


class InterventionPlugin(Protocol):
    """
    Protocol for dynamic intervention strategies.
    """

    name: str

    def evaluate(self, agent: "UnifiedAdaptiveAgent") -> Optional[Dict[str, Any]]: ...


@dataclass
class DefaultAdvicePlugin:
    name: str = "default_advice"

    def evaluate(self, agent: "UnifiedAdaptiveAgent") -> Optional[Dict[str, Any]]:
        tips = []
        if agent.confusion_level > 0.7:
            tips.append("High confusion detected. Break problem into smaller steps.")
        if agent.entropy > 1.0:
            tips.append("Entropy high—consider consolidating hypotheses.")
        if not tips:
            return None
        return {"action": "advice", "message": tips}


# ======================================================================================
# MazeMaster
# ======================================================================================


class MazeMaster:
    """
    Governs agents and applies interventions based on thresholds or plugins.
    """

    def __init__(self, policy: Optional[InterventionPolicy] = None):
        self.interventions = 0
        self.policy = policy or InterventionPolicy()
        self.plugins: Dict[str, InterventionPlugin] = {}
        # Register default plugin
        self.register_plugin(DefaultAdvicePlugin())

    def register_plugin(self, plugin: InterventionPlugin):
        self.plugins[plugin.name] = plugin

    def quick_escape(self, agent: "UnifiedAdaptiveAgent"):
        agent.log_event("MazeMaster: Quick escape triggered.", category="intervention")
        agent.status = "escaped"
        return {"action": "escape", "message": f"Agent {agent.name} guided out by MazeMaster."}

    def _fallback_advice(self, agent: "UnifiedAdaptiveAgent"):
        feedback = []
        if agent.confusion_level > 0.7:
            feedback.append("You seem overwhelmed; focus on one subproblem.")
        if agent.entropy > 0.8:
            feedback.append("Too many branches—narrow the hypothesis set.")
        if agent.status == "stuck":
            feedback.append("Reframe the objective; ask what success looks like.")
        if not feedback:
            feedback.append("You're on track.")
        agent.log_event("MazeMaster advice: " + " | ".join(feedback), category="intervention")
        return {"action": "advice", "message": feedback}

    def intervene(self, agent: "UnifiedAdaptiveAgent"):
        self.interventions += 1
        if (
            agent.confusion_level >= self.policy.confusion_escape_thresh
            or agent.entropy >= self.policy.entropy_escape_thresh
            or agent.status in {"stuck", "looping"}
        ):
            return self.quick_escape(agent)

        if (
            agent.confusion_level >= self.policy.soft_advice_thresh
            or agent.entropy >= self.policy.soft_advice_thresh
        ):
            # Try plugins first
            for plugin in self.plugins.values():
                out = plugin.evaluate(agent)
                if out:
                    agent.log_event(
                        f"MazeMaster plugin {plugin.name}: {out['message']}",
                        category="intervention",
                    )
                    return out
            return self._fallback_advice(agent)
        return {"action": "none"}

    def govern_agents(self, agents: List["UnifiedAdaptiveAgent"]):
        for agent in agents:
            action = self.intervene(agent)
            if action["action"] != "none":
                agent.log_event(
                    f"MazeMaster intervention: {action['message']}", category="intervention"
                )


# ======================================================================================
# Unified Adaptive Agent
# ======================================================================================


class UnifiedAdaptiveAgent:
    """
    High-level agent adapting via internal node reasoning and style influences.
    """

    def __init__(
        self,
        name: str,
        style: Dict[str, float],
        alive_node: AliveLoopNode,
        resource_room: ResourceRoom,
        agent_config: Optional[AgentConfig] = None,
    ):
        self.agent_id = str(uuid.uuid4())
        self.name = name
        self.style = style
        self.alive_node = alive_node
        self.resource_room = resource_room
        self.status = "active"
        self.confusion_level = 0.0
        self.entropy = 0.0
        self.knowledge_graph: Dict[str, Any] = {}
        self.interaction_history: List[Dict[str, Any]] = []
        self.event_log: List[str] = []
        self.style_cache: List[str] = []

        self._cfg = agent_config or AgentConfig()
        self._lock = threading.Lock() if self._cfg.enable_thread_safety else None

    # ---------------- Logging ----------------

    def log_event(self, event: str, category: str = "general"):
        msg = f"[{self.name}][{category}] {event}"
        self.event_log.append(msg)
        logger.info(msg)

    # ---------------- Reasoning ----------------

    def reason(
        self,
        task: str,
        temperature: float = 0.6,
        max_tokens: int = 128,
        trace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        if self._cfg.enable_thread_safety and self._lock:
            with self._lock:
                return self._reason_impl(task, temperature, max_tokens, trace_id)
        return self._reason_impl(task, temperature, max_tokens, trace_id)

    def _reason_impl(
        self, task: str, temperature: float, max_tokens: int, trace_id: Optional[str]
    ) -> Dict[str, Any]:
        if self._cfg.enable_tracing and trace_id is None:
            trace_id = str(uuid.uuid4())
        result = self.alive_node.safe_think(
            self.name, task, temperature=temperature, max_tokens=max_tokens, trace_id=trace_id
        )
        styled_result = self._apply_style_influence(result)
        key = f"{self.name}_reason_{len(self.knowledge_graph)}"
        self.knowledge_graph[key] = styled_result
        self.interaction_history.append(styled_result)
        self._update_confusion_and_entropy(styled_result)
        return styled_result

    def _apply_style_influence(self, base_result: Dict[str, Any]) -> Dict[str, Any]:
        styled = dict(base_result)
        insights = []
        for dim, val in self.style.items():
            if val > self._cfg.style_influence_threshold:
                insights.append(f"{dim.capitalize()} influence")
        styled["style_insights"] = insights
        styled["style_signature"] = hash(tuple(sorted(insights))) if insights else None
        return styled

    def _update_confusion_and_entropy(self, result: Dict[str, Any]):
        conf = result.get("confidence", 0.5)
        if conf < self._cfg.low_confidence_threshold:
            self.confusion_level = min(
                1.0, self.confusion_level + self._cfg.confusion_increase_step
            )
        elif conf > self._cfg.high_confidence_threshold:
            self.confusion_level = max(
                0.0, self.confusion_level - self._cfg.confusion_decrease_step
            )

        # Update entropy based on style usage distribution (incremental)
        self.style_cache.extend(result.get("style_insights", []))
        counts = Counter(self.style_cache)
        total = sum(counts.values())
        if total > 0:
            probs = [c / total for c in counts.values()]
            new_entropy = -sum(p * math.log2(p) for p in probs if p > 0)
            # Simple smoothing
            self.entropy = (
                1 - self._cfg.entropy_smoothing
            ) * self.entropy + self._cfg.entropy_smoothing * new_entropy

    # ---------------- Resource Room ----------------

    def teleport_to_resource_room(self, info: Dict[str, Any]):
        if self.resource_room:
            self.resource_room.deposit(self.agent_id, info)
            prev_status = self.status
            self.status = "in_resource_room"
            self.log_event(f"Teleported to ResourceRoom (prev={prev_status}).", category="resource")

    def retrieve_from_resource_room(self) -> Dict[str, Any]:
        if self.resource_room:
            info = self.resource_room.retrieve(self.agent_id)
            self.log_event("Retrieved info from ResourceRoom.", category="resource")
            return info
        return {}

    # ---------------- Introspection ----------------

    def get_state(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "status": self.status,
            "confusion_level": self.confusion_level,
            "entropy": self.entropy,
            "knowledge_graph_size": len(self.knowledge_graph),
            "event_log_length": len(self.event_log),
        }

    # ---------------- Advanced Memory Access ----------------

    def search_memories(
        self,
        query: str,
        top_k: int = 5,
        long_term: bool = True,
    ) -> List[Tuple[float, Memory]]:
        """
        Naive semantic-like search: score by substring + importance weighting.
        (Hook point for vector store / embedding similarity.)
        """
        matches: List[Tuple[float, Memory]] = []
        q_lower = query.lower()

        # Short-term
        for mem in self.alive_node.memory:
            base = 0.0
            if isinstance(mem.content, str) and q_lower in mem.content.lower():
                base += 0.5
            score = base + mem.importance * 0.5
            if score > 0.1:
                matches.append((score, mem))

        if long_term:
            for bucket in self.alive_node.long_term_memory.values():
                for mem in bucket:
                    base = 0.0
                    if isinstance(mem.content, str) and q_lower in mem.content.lower():
                        base += 0.7
                    score = base + mem.importance * 0.7
                    if score > 0.2:
                        matches.append((score, mem))

        matches.sort(key=lambda x: x[0], reverse=True)
        return matches[:top_k]


# ======================================================================================
# Utilities
# ======================================================================================


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    logger.info(f"[Seed] Global seed set to {seed}")


# ======================================================================================
# Module Exports
# ======================================================================================

__all__ = [
    "Memory",
    "MemoryConfig",
    "AgentConfig",
    "InterventionPolicy",
    "SocialSignal",
    "CapacitorInSpace",
    "AliveLoopNode",
    "ResourceRoom",
    "NetworkMetrics",
    "MazeMaster",
    "UnifiedAdaptiveAgent",
    "InterventionPlugin",
    "DefaultAdvicePlugin",
    "set_global_seed",
]


# NOTE: run_labyrinth_simulation function still lives in labyrinth_simulation.py
if __name__ == "__main__":
    from labyrinth_simulation import run_labyrinth_simulation  # noqa: E402

    run_labyrinth_simulation()
