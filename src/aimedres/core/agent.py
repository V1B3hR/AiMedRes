"""
AiMedRes Agent - Secure Cognitive Agent Implementation (Enhanced Version)

Advanced, production-oriented cognitive agent with:
- Adaptive neural network "brain"
- Biological state & mood simulation
- Safety monitoring (prediction + communication)
- Structured, extensible memory architecture (working / long-term / topical index)
- Secure input validation & sanitation
- Conversation management
- Telemetry & rolling performance metrics
- Event hook system (extensible lifecycle callbacks)
- Robust state transition management
- Graceful degradation & defensive programming
"""

from __future__ import annotations

import logging
import math
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
)

import numpy as np

from ..security.validation import InputValidator
from ..utils.safety import SafetyMonitor
from .neural_network import AdaptiveNeuralNetwork, BiologicalState, MoodType

# -----------------------------------------------------------------------------
# Logging Setup (non-intrusive; respects existing configuration)
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)
if not logger.handlers:
    # Avoid duplicate handlers in multi-import scenarios
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s :: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


# -----------------------------------------------------------------------------
# Agent Enumerations & Data Models
# -----------------------------------------------------------------------------
class AgentState(Enum):
    """Agent operational states"""

    ACTIVE = "active"
    Mindful = "mindful"  # Preserved for backward compatibility
    MINDFUL = "mindful"  # Canonical alias
    SLEEPING = "sleeping"
    LEARNING = "learning"
    COMMUNICATING = "communicating"
    ERROR = "error"
    MAINTENANCE = "maintenance"
    STARTING = "starting"
    SHUTTING_DOWN = "shutting_down"


@dataclass
class ConversationMessage:
    """Structured conversation message"""

    message_id: str
    sender_id: str
    sender_name: str
    content: str
    timestamp: datetime
    recipient_id: Optional[str] = None
    biological_mood: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Conversation:
    """Secure conversation record"""

    conversation_id: str
    participants: List[str]
    messages: List[ConversationMessage] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_activity: datetime = field(default_factory=datetime.utcnow)
    active: bool = True

    def add_message(self, msg: ConversationMessage):
        self.messages.append(msg)
        self.last_activity = msg.timestamp


@dataclass
class MemoryItem:
    """Unified internal memory structure"""

    memory_id: str
    type: str
    payload: Dict[str, Any]
    confidence: float
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    topics: List[str] = field(default_factory=list)
    importance: float = 0.5

    def touch(self):
        self.last_accessed = datetime.utcnow()


@dataclass
class RollingMetric:
    """Exponential moving average for telemetry metrics"""

    name: str
    alpha: float = 0.15
    value: Optional[float] = None
    count: int = 0

    def update(self, new_value: float):
        if not math.isfinite(new_value):
            return
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.alpha * new_value + (1 - self.alpha) * self.value
        self.count += 1

    def snapshot(self) -> Dict[str, Any]:
        return {"name": self.name, "ema": self.value, "updates": self.count}


# -----------------------------------------------------------------------------
# Configuration DataClass
# -----------------------------------------------------------------------------
@dataclass
class AgentConfig:
    """Configuration parameters for DuetMindAgent"""

    input_validation: bool = True
    safety_monitoring: bool = True
    memory_limit: int = 2000
    working_memory_limit: int = 32
    long_term_limit: int = 250
    topic_index_limit: int = 1000
    confidence_threshold_working: float = 0.70
    confidence_threshold_long_term: float = 0.82
    conversation_inactivity_timeout: timedelta = timedelta(hours=12)
    auto_cleanup_interval: timedelta = timedelta(minutes=10)
    enable_topic_indexing: bool = True
    communication_max_length: int = 12_000
    low_energy_threshold: float = 10.0
    critical_energy_threshold: float = 5.0
    enable_structured_metrics: bool = True
    state_transition_logging: bool = True
    allow_state_force: bool = False
    prediction_timeout_seconds: float = 10.0
    max_error_rate: float = 0.35
    auto_nap_hours: float = 1.0
    memory_token_pattern: re.Pattern = re.compile(r"[A-Za-z][A-Za-z0-9_]{2,32}")


# -----------------------------------------------------------------------------
# Event Hook Types
# -----------------------------------------------------------------------------
OnThinkHook = Callable[[Dict[str, Any]], None]
OnMessageHook = Callable[[ConversationMessage, Conversation], None]
OnStateChangeHook = Callable[[AgentState, AgentState], None]


# -----------------------------------------------------------------------------
# Main Agent Implementation
# -----------------------------------------------------------------------------
class DuetMindAgent:
    """
    Enhanced secure cognitive agent with adaptive neural network backend.
    """

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------
    def __init__(
        self,
        agent_id: Optional[str] = None,
        name: Optional[str] = None,
        neural_config: Optional[Dict[str, Any]] = None,
        enable_safety_monitoring: bool = True,
        enable_input_validation: bool = True,
        config: Optional[AgentConfig] = None,
    ):
        self.agent_id: str = agent_id or str(uuid.uuid4())
        self.name: str = name or f"Agent-{self.agent_id[:8]}"
        self.created_at: datetime = datetime.utcnow()
        self.last_activity: datetime = self.created_at

        # Config
        self.config: AgentConfig = config or AgentConfig(
            input_validation=enable_input_validation,
            safety_monitoring=enable_safety_monitoring,
        )

        # Validators & Safety
        self.input_validator: Optional[InputValidator] = (
            InputValidator() if self.config.input_validation else None
        )
        self.safety_monitor: Optional[SafetyMonitor] = (
            SafetyMonitor() if self.config.safety_monitoring else None
        )

        # Brain
        neural_config = neural_config or {}
        self.brain = AdaptiveNeuralNetwork(
            input_size=neural_config.get("input_size", 64),
            hidden_layers=neural_config.get(
                "hidden_layers",
                [4096, 2048, 1024, 512, 256, 128, 512, 128],
            ),
            output_size=neural_config.get("output_size", 4),
            learning_rate=neural_config.get("learning_rate", 0.0007),
            enable_safety_monitoring=self.config.safety_monitoring,
        )

        # State
        self.state: AgentState = AgentState.STARTING
        self.biological_state: BiologicalState = BiologicalState()

        # Communication
        self.conversations: Dict[str, Conversation] = {}
        self.message_queue: List[Dict[str, Any]] = []
        self.communication_style: Dict[str, float] = {
            "directness": 0.7,
            "formality": 0.4,
            "expressiveness": 0.6,
            "assertiveness": 0.7,
        }

        # Memory Structures
        self.memory: List[MemoryItem] = []
        self.working_memory: List[MemoryItem] = []
        self.long_term_memory: Dict[str, MemoryItem] = {}
        self.topic_index: Dict[str, List[str]] = {}  # topic -> list of memory_ids

        # Metrics
        self.total_interactions: int = 0
        self.successful_interactions: int = 0
        self.error_count: int = 0
        self.metrics_latency = RollingMetric("think_latency")
        self.metrics_confidence = RollingMetric("confidence")
        self.metrics_energy = RollingMetric("energy")
        self.metrics_entropy = RollingMetric("prediction_entropy")

        # Event hooks
        self._on_think_hooks: List[OnThinkHook] = []
        self._on_message_hooks: List[OnMessageHook] = []
        self._on_state_change_hooks: List[OnStateChangeHook] = []

        # Threading & Control
        self._lock = threading.RLock()
        self._last_cleanup: datetime = datetime.utcnow()

        # Transition to ACTIVE
        self._set_state(AgentState.ACTIVE)

        logger.info(f"AiMedRes agent initialized (enhanced): {self.name} ({self.agent_id})")

    # -------------------------------------------------------------------------
    # Public API: Event Hook Registration
    # -------------------------------------------------------------------------
    def add_on_think_hook(self, hook: OnThinkHook):
        self._on_think_hooks.append(hook)

    def add_on_message_hook(self, hook: OnMessageHook):
        self._on_message_hooks.append(hook)

    def add_on_state_change_hook(self, hook: OnStateChangeHook):
        self._on_state_change_hooks.append(hook)

    # -------------------------------------------------------------------------
    # Internal: State Management
    # -------------------------------------------------------------------------
    def _set_state(self, new_state: AgentState, force: bool = False):
        if not isinstance(new_state, AgentState):
            raise ValueError("Invalid agent state")

        with self._lock:
            old_state = self.state
            if old_state == AgentState.ERROR and not (force or self.config.allow_state_force):
                logger.warning("Attempt to change state from ERROR ignored (force disabled)")
                return
            if old_state == new_state:
                return
            self.state = new_state
            if self.config.state_transition_logging:
                logger.debug(f"State transition: {old_state.value} -> {new_state.value}")
            for hook in self._on_state_change_hooks:
                safe_call_hook(hook, old_state, new_state)

    from contextlib import contextmanager

    @contextmanager
    def _state_context(self, temp_state: AgentState):
        previous = self.state
        self._set_state(temp_state)
        try:
            yield
        finally:
            # Don't overwrite ERROR or SHUTTING_DOWN
            if self.state not in (AgentState.ERROR, AgentState.SHUTTING_DOWN):
                self._set_state(previous)

    # -------------------------------------------------------------------------
    # Core Cognitive Processing
    # -------------------------------------------------------------------------
    def think(
        self,
        inputs: np.ndarray,
        context: Optional[Dict[str, Any]] = None,
        validate_input: bool = True,
        timeout_seconds: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Perform secure cognitive processing.

        Args:
            inputs: Input vector (numpy ndarray)
            context: Optional metadata/context
            validate_input: Whether to validate input
            timeout_seconds: Override internal prediction timeout if needed

        Returns:
            Dict containing prediction, confidence, biological state, and metadata
        """
        context = context or {}
        start_time = time.time()
        local_timeout = timeout_seconds or self.config.prediction_timeout_seconds

        try:
            if validate_input and self.input_validator:
                self._validate_think_input(inputs, context)

            if self.state == AgentState.ERROR:
                raise RuntimeError("Agent in error state - thinking disabled")

            # Biological capacity check
            if self.biological_state.energy < self.config.low_energy_threshold:
                self._handle_low_energy()
                if self.biological_state.energy < self.config.low_energy_threshold:
                    return self._error_result(
                        "Insufficient energy for thinking", start_time, err_type="low_energy"
                    )

            with self._lock, self._state_context(AgentState.LEARNING):
                prediction = self._safe_predict(inputs, local_timeout)

                confidence, entropy = self._calculate_confidence(prediction)
                elapsed = time.time() - start_time

                result = {
                    "agent_id": self.agent_id,
                    "agent_name": self.name,
                    "prediction": prediction.tolist(),
                    "confidence": confidence,
                    "timestamp": datetime.utcnow().isoformat(),
                    "processing_time": elapsed,
                    "biological_state": self._biological_snapshot(),
                    "context": context,
                    "prediction_entropy": entropy,
                }

                # Memory & metrics
                self._store_memory_internal(
                    MemoryItem(
                        memory_id=str(uuid.uuid4()),
                        type="thought",
                        payload={"result": result, "raw_context": context},
                        confidence=confidence,
                        importance=self._derive_importance(confidence, entropy),
                        topics=self._extract_topics(context),
                    )
                )

                self._post_think_update(confidence, entropy, elapsed)
                for hook in self._on_think_hooks:
                    safe_call_hook(hook, result)

                return result

        except Exception as e:
            return self._handle_exception(e, start_time, phase="think")

    # -------------------------------------------------------------------------
    # Communication
    # -------------------------------------------------------------------------
    def communicate(
        self,
        message: str,
        recipient_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        validate_input: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Secure communication with other agents or broadcast.

        Returns dict containing success flag, conversation id, and message id.
        """
        metadata = metadata or {}
        try:
            if validate_input and self.input_validator:
                self._validate_communication_input(message, recipient_id)

            if len(message) > self.config.communication_max_length:
                raise ValueError("Message exceeds configured maximum length")

            with self._lock, self._state_context(AgentState.COMMUNICATING):
                convo = self._get_or_create_conversation(conversation_id, recipient_id)

                msg_obj = ConversationMessage(
                    message_id=str(uuid.uuid4()),
                    sender_id=self.agent_id,
                    sender_name=self.name,
                    content=message,
                    recipient_id=recipient_id,
                    timestamp=datetime.utcnow(),
                    biological_mood=self.biological_state.mood.value,
                    metadata=metadata,
                )
                convo.add_message(msg_obj)

                self._store_memory_internal(
                    MemoryItem(
                        memory_id=str(uuid.uuid4()),
                        type="communication",
                        payload={
                            "message_id": msg_obj.message_id,
                            "conversation_id": convo.conversation_id,
                            "content": message,
                            "recipient_id": recipient_id,
                            "metadata": metadata,
                        },
                        confidence=0.65,
                        importance=0.4,
                        topics=self._extract_topics({"content": message}),
                    )
                )

                if self.safety_monitor:
                    try:
                        self.safety_monitor.log_event("communication", {"length": len(message)})
                    except Exception:
                        pass

                for hook in self._on_message_hooks:
                    safe_call_hook(hook, msg_obj, convo)

                logger.info(
                    f"Agent {self.name} sent message {msg_obj.message_id} "
                    f"to {recipient_id or 'broadcast'} in conversation {convo.conversation_id}"
                )

                self.last_activity = datetime.utcnow()
                return {
                    "success": True,
                    "conversation_id": convo.conversation_id,
                    "message_id": msg_obj.message_id,
                    "timestamp": msg_obj.timestamp.isoformat(),
                }

        except Exception as e:
            logger.error(f"Communication error: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
            }

    # -------------------------------------------------------------------------
    # Biological Recovery
    # -------------------------------------------------------------------------
    def sleep(self, duration_hours: float = 8.0) -> Dict[str, Any]:
        """
        Put agent to sleep for recovery & memory consolidation.
        """
        with self._lock, self._state_context(AgentState.SLEEPING):
            start = datetime.utcnow()
            self.brain.sleep_cycle(duration_hours, node_percentage=0.3)

            self.biological_state.energy = min(
                100.0, self.biological_state.energy + duration_hours * 12.5
            )
            self.biological_state.stress_level = max(
                0.0, self.biological_state.stress_level - duration_hours * 10.0
            )
            self.biological_state.sleep_need = max(
                0.0, self.biological_state.sleep_need - duration_hours * 12.5
            )
            self.biological_state.last_sleep = start
            self.biological_state.mood = MoodType.FOCUSED

            self._consolidate_memory()

            end = datetime.utcnow()
            result = {
                "agent_id": self.agent_id,
                "agent_name": self.name,
                "sleep_duration_hours": duration_hours,
                "start_time": start.isoformat(),
                "end_time": end.isoformat(),
                "energy_after": self.biological_state.energy,
                "stress_after": self.biological_state.stress_level,
                "mood_after": self.biological_state.mood.value,
            }
            logger.info(f"Agent {self.name} completed sleep cycle ({duration_hours}h)")
            return result

    # -------------------------------------------------------------------------
    # Status & Metrics
    # -------------------------------------------------------------------------
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive agent status snapshot."""
        with self._lock:
            success_rate = (
                self.successful_interactions / self.total_interactions
                if self.total_interactions
                else 0.0
            )

            brain_health = {}
            try:
                brain_health = self.brain.get_network_health()
            except Exception as e:
                brain_health = {"error": str(e)}

            # Memory stats
            status = {
                "agent_id": self.agent_id,
                "name": self.name,
                "state": self.state.value,
                "created_at": self.created_at.isoformat(),
                "last_activity": self.last_activity.isoformat(),
                "uptime_seconds": (datetime.utcnow() - self.created_at).total_seconds(),
                "biological_state": self._biological_snapshot(),
                "performance": {
                    "total_interactions": self.total_interactions,
                    "successful_interactions": self.successful_interactions,
                    "error_count": self.error_count,
                    "success_rate": success_rate,
                },
                "metrics": {
                    "latency": self.metrics_latency.snapshot(),
                    "confidence": self.metrics_confidence.snapshot(),
                    "energy": self.metrics_energy.snapshot(),
                    "prediction_entropy": self.metrics_entropy.snapshot(),
                },
                "brain_health": brain_health,
                "memory": {
                    "total": len(self.memory),
                    "working_memory_items": len(self.working_memory),
                    "long_term_items": len(self.long_term_memory),
                    "topic_index_entries": len(self.topic_index),
                },
                "communication": {
                    "active_conversations": len(
                        [c for c in self.conversations.values() if c.active]
                    ),
                    "total_conversations": len(self.conversations),
                    "messages_in_queue": len(self.message_queue),
                },
                "config": {
                    "memory_limit": self.config.memory_limit,
                    "working_memory_limit": self.config.working_memory_limit,
                    "long_term_limit": self.config.long_term_limit,
                },
            }
            return status

    # -------------------------------------------------------------------------
    # Memory Query API
    # -------------------------------------------------------------------------
    def query_memory(
        self,
        topic: Optional[str] = None,
        memory_type: Optional[str] = None,
        limit: int = 10,
        min_confidence: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Query agent memory by topic and/or type.
        """
        with self._lock:
            candidates: Iterable[MemoryItem]
            if topic and self.config.enable_topic_indexing:
                memory_ids = self.topic_index.get(topic.lower(), [])
                mapping = {m.memory_id: m for m in self.memory}
                candidates = (mapping[mid] for mid in memory_ids if mid in mapping)
            else:
                candidates = list(self.memory)

            filtered = [
                m
                for m in candidates
                if (memory_type is None or m.type == memory_type) and m.confidence >= min_confidence
            ]

            # Sort by importance then recency
            filtered.sort(key=lambda m: (m.importance, m.created_at), reverse=True)
            return [
                {
                    "memory_id": m.memory_id,
                    "type": m.type,
                    "confidence": m.confidence,
                    "created_at": m.created_at.isoformat(),
                    "importance": m.importance,
                    "topics": m.topics,
                }
                for m in filtered[:limit]
            ]

    # -------------------------------------------------------------------------
    # Shutdown & Lifecycle
    # -------------------------------------------------------------------------
    def shutdown(self):
        """Graceful shutdown of agent & resources."""
        with self._lock:
            self._set_state(AgentState.SHUTTING_DOWN, force=True)
            logger.info(f"Shutting down agent {self.name}")

            try:
                self.brain.shutdown()
            except Exception as e:
                logger.warning(f"Brain shutdown encountered an error: {e}")

            self.memory.clear()
            self.working_memory.clear()
            self.long_term_memory.clear()
            self.message_queue.clear()
            self.conversations.clear()
            self.topic_index.clear()

            self._set_state(AgentState.MAINTENANCE, force=True)
            logger.info(f"Agent {self.name} shutdown complete")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()

    # -------------------------------------------------------------------------
    # Internal Helpers: Validation
    # -------------------------------------------------------------------------
    def _validate_think_input(self, inputs: np.ndarray, context: Dict[str, Any]):
        if not isinstance(inputs, np.ndarray):
            raise ValueError("Inputs must be a numpy ndarray")
        if inputs.ndim != 1:
            raise ValueError("Inputs must be a 1D vector")
        if not np.isfinite(inputs).all():
            raise ValueError("Inputs contain non-finite values")
        if not isinstance(context, dict):
            raise ValueError("Context must be a dictionary")

    def _validate_communication_input(self, message: str, recipient_id: Optional[str]):
        if not isinstance(message, str):
            raise ValueError("Message must be string")
        if not message.strip():
            raise ValueError("Message cannot be empty")
        if recipient_id is not None and not isinstance(recipient_id, str):
            raise ValueError("Recipient ID must be string")

    # -------------------------------------------------------------------------
    # Internal Helpers: Prediction & Confidence
    # -------------------------------------------------------------------------
    def _safe_predict(self, inputs: np.ndarray, timeout_seconds: float) -> np.ndarray:
        start = time.time()
        prediction = self.brain.predict(inputs, validate_input=False)
        if (time.time() - start) > timeout_seconds:
            raise TimeoutError("Prediction exceeded timeout")
        return prediction

    def _calculate_confidence(self, prediction: np.ndarray) -> Tuple[float, float]:
        """
        Return (confidence, entropy)
        Confidence factoring energy & stress.
        """
        if prediction.size == 0:
            return 0.0, 0.0

        if prediction.size > 1:
            # Softmax
            stabilized = prediction - np.max(prediction)
            exp_vals = np.exp(stabilized)
            probs = exp_vals / np.sum(exp_vals)
            entropy = -np.sum(probs * np.log(probs + 1e-12))
            max_entropy = math.log(len(probs))
            base_conf = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 0.0
        else:
            # Regression-like single output
            entropy = 0.0
            base_conf = 1.0 / (1.0 + math.exp(-abs(float(prediction[0]))))
            base_conf = (base_conf - 0.5) * 2.0  # scale to 0..1

        energy_factor = self.biological_state.energy / 100.0
        stress_penalty = self.biological_state.stress_level / 150.0
        final_conf = base_conf * energy_factor * (1.0 - stress_penalty)
        final_conf = max(0.0, min(1.0, final_conf))
        return final_conf, entropy

    def _derive_importance(self, confidence: float, entropy: float) -> float:
        # Higher confidence & lower entropy => more important
        return max(0.0, min(1.0, (confidence * 0.7) + (0.3 * (1.0 - (entropy / 5.0)))))

    # -------------------------------------------------------------------------
    # Internal Helpers: Memory
    # -------------------------------------------------------------------------
    def _store_memory_internal(self, item: MemoryItem):
        self.memory.append(item)
        # Trim global memory
        if len(self.memory) > self.config.memory_limit:
            self.memory = self.memory[-self.config.memory_limit :]

        # Working memory
        if item.confidence >= self.config.confidence_threshold_working:
            self.working_memory.append(item)
            if len(self.working_memory) > self.config.working_memory_limit:
                self.working_memory.pop(0)

        # Topic indexing
        if self.config.enable_topic_indexing:
            for topic in item.topics:
                lst = self.topic_index.setdefault(topic.lower(), [])
                lst.append(item.memory_id)
                if len(lst) > 200:  # per-topic cap
                    del lst[0]
            # Trim global index if needed
            if len(self.topic_index) > self.config.topic_index_limit:
                # Remove oldest topic references arbitrarily
                for k in list(self.topic_index.keys())[:50]:
                    del self.topic_index[k]

    def _consolidate_memory(self):
        # Promote high-confidence working memory -> long-term
        promotable = [
            m
            for m in self.working_memory
            if m.confidence >= self.config.confidence_threshold_long_term
        ]
        for m in promotable:
            if len(self.long_term_memory) >= self.config.long_term_limit:
                # Remove oldest by created_at
                oldest_key = min(
                    self.long_term_memory,
                    key=lambda x: self.long_term_memory[x].created_at,
                )
                del self.long_term_memory[oldest_key]
            self.long_term_memory[m.memory_id] = m

        self.working_memory.clear()
        logger.info(f"[Memory] Consolidated: {len(self.long_term_memory)} long-term memories")

    def _extract_topics(self, context: Dict[str, Any]) -> List[str]:
        topics: List[str] = []

        def add_token(tok: str):
            if (
                2 < len(tok) < 28
                and self.config.memory_token_pattern.match(tok)
                and tok.lower() not in {"the", "and", "with", "from", "this", "that"}
            ):
                topics.append(tok.lower())

        for v in context.values():
            if isinstance(v, str):
                for token in re.findall(r"[A-Za-z0-9_]+", v):
                    add_token(token)
            elif isinstance(v, (list, tuple)):
                for el in v:
                    if isinstance(el, str):
                        add_token(el)
        return list(dict.fromkeys(topics))[:10]

    # -------------------------------------------------------------------------
    # Internal Helpers: Biological & Metrics updates
    # -------------------------------------------------------------------------
    def _post_think_update(self, confidence: float, entropy: float, latency: float):
        self.total_interactions += 1
        self.successful_interactions += 1
        self.last_activity = datetime.utcnow()

        # Biological adjustments
        energy_cost = (1.0 - confidence) * 2.0
        self.biological_state.energy = max(0.0, self.biological_state.energy - energy_cost)
        if confidence < 0.3:
            self.biological_state.stress_level = min(
                100.0, self.biological_state.stress_level + 2.0
            )

        time_since_sleep = datetime.utcnow() - self.biological_state.last_sleep
        hours_awake = time_since_sleep.total_seconds() / 3600.0
        self.biological_state.sleep_need = min(100.0, hours_awake * 4.0)

        # Metrics
        self.metrics_latency.update(latency)
        self.metrics_confidence.update(confidence)
        self.metrics_energy.update(self.biological_state.energy)
        self.metrics_entropy.update(entropy)

        # Cleanup cycle
        if (datetime.utcnow() - self._last_cleanup) > self.config.auto_cleanup_interval:
            self._maintenance_cleanup()

    def _maintenance_cleanup(self):
        # Deactivate stale conversations
        cutoff = datetime.utcnow() - self.config.conversation_inactivity_timeout
        deactivated = 0
        for convo in self.conversations.values():
            if convo.active and convo.last_activity < cutoff:
                convo.active = False
                deactivated += 1
        self._last_cleanup = datetime.utcnow()
        if deactivated:
            logger.debug(f"Deactivated {deactivated} stale conversations")

    def _handle_low_energy(self):
        logger.warning(f"Agent {self.name} low energy: {self.biological_state.energy:.2f}")
        if self.biological_state.energy < self.config.critical_energy_threshold:
            logger.info("Critical energy - triggering automatic nap")
            self.sleep(self.config.auto_nap_hours)

    def _biological_snapshot(self) -> Dict[str, Any]:
        return {
            "energy": self.biological_state.energy,
            "sleep_need": self.biological_state.sleep_need,
            "stress_level": self.biological_state.stress_level,
            "mood": self.biological_state.mood.value,
            "last_sleep": self.biological_state.last_sleep.isoformat(),
        }

    # -------------------------------------------------------------------------
    # Internal Helpers: Conversation Management
    # -------------------------------------------------------------------------
    def _get_or_create_conversation(
        self, conversation_id: Optional[str], recipient_id: Optional[str]
    ) -> Conversation:
        if conversation_id and conversation_id in self.conversations:
            return self.conversations[conversation_id]

        conv_id = conversation_id or str(uuid.uuid4())
        participants = [self.agent_id]
        if recipient_id:
            participants.append(recipient_id)

        convo = Conversation(conversation_id=conv_id, participants=participants)
        self.conversations[conv_id] = convo
        return convo

    # -------------------------------------------------------------------------
    # Internal Helpers: Error Handling
    # -------------------------------------------------------------------------
    def _error_result(self, message: str, start_time: float, err_type: str = "error"):
        return {
            "agent_id": self.agent_id,
            "agent_name": self.name,
            "error": message,
            "error_type": err_type,
            "confidence": 0.0,
            "timestamp": datetime.utcnow().isoformat(),
            "processing_time": time.time() - start_time,
        }

    def _handle_exception(self, e: Exception, start_time: float, phase: str) -> Dict[str, Any]:
        self.error_count += 1
        self.total_interactions += 1
        logger.error(f"Agent {self.name} {phase} error: {e}")
        if self._should_enter_error_state():
            self._set_state(AgentState.ERROR, force=True)
        return self._error_result(str(e), start_time, err_type=phase)

    def _should_enter_error_state(self) -> bool:
        if self.total_interactions < 10:
            return False
        current_error_rate = self.error_count / self.total_interactions
        return current_error_rate > self.config.max_error_rate


# -----------------------------------------------------------------------------
# Utility: Safe hook caller
# -----------------------------------------------------------------------------
def safe_call_hook(hook: Callable, *args, **kwargs):
    try:
        hook(*args, **kwargs)
    except Exception as e:
        logger.debug(f"Hook call failed (ignored): {e}")
