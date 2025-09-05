import numpy as np
import random
import math
import time
import uuid
import logging
from collections import deque, defaultdict, Counter
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

# ===================== Logging & Monitoring Setup =====================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("UnifiedLabyrinth")

# ===================== Memory & SocialSignal Core =====================
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

    def age(self):
        self.importance *= self.decay_rate
        if abs(self.emotional_valence) > 0.7:
            self.decay_rate = min(0.997, 0.97 + (abs(self.emotional_valence) - 0.7) * 0.1)

class SocialSignal:
    def __init__(self, content: Any, signal_type: str, urgency: float, source_id: int, requires_response: bool = False):
        self.id = str(uuid.uuid4())
        self.content = content
        self.signal_type = signal_type
        self.urgency = urgency
        self.source_id = source_id
        self.timestamp = 0
        self.requires_response = requires_response
        self.response = None

# ===================== Capacitor Resource Core =====================
class CapacitorInSpace:
    def __init__(self, position, capacity=5.0, initial_energy=0.0):
        self.position = np.array(position, dtype=float)
        self.capacity = max(0.0, capacity)
        self.energy = min(max(0.0, initial_energy), self.capacity)

    def charge(self, amount):
        self.energy = min(self.capacity, self.energy + amount)

    def discharge(self, amount):
        self.energy = max(0.0, self.energy - amount)

    def status(self):
        return f"Capacitor: Position {self.position}, Energy {round(self.energy,2)}/{self.capacity}"

# ===================== AliveLoopNode (Agent Brain) =====================
class AliveLoopNode:
    sleep_stages = ["light", "REM", "deep"]

    def __init__(self, position, velocity, initial_energy=10.0, field_strength=1.0, node_id=0):
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.energy = max(0.0, initial_energy)
        self.field_strength = field_strength
        self.radius = max(0.1, 0.1 + 0.05 * initial_energy)
        self.phase = "active"
        self.node_id = node_id
        self.anxiety = 0.0
        self.phase_history = deque(maxlen=24)
        self.memory = deque(maxlen=1000)
        self.working_memory = deque(maxlen=7)
        self.long_term_memory = {}
        self.communication_queue = deque(maxlen=20)
        self.trust_network = {}
        self.influence_network = {}
        self.emotional_state = {"valence": 0.0}
        self._time = 0
        self.communication_style = {"directness": 0.7, "formality": 0.3, "expressiveness": 0.6}
        self.confusion_level = 0.0
        self.loop_counter = 0

    def safe_think(self, agent_name: str, task: str) -> Dict[str, Any]:
        self._time += 1
        if not task:
            self.confusion_level += 0.1
            return {"error": "Empty task", "confidence": 0.0}
        confidence = random.uniform(0.3, 0.95)
        if confidence < 0.4:
            self.confusion_level += 0.05
        elif confidence > 0.8:
            self.confusion_level = max(0, self.confusion_level - 0.05)
        result = {
            "agent": agent_name,
            "task": task,
            "insight": f"{agent_name} reasoned: {task}",
            "confidence": confidence,
            "energy": self.energy,
            "confusion_level": self.confusion_level
        }
        mem = Memory(content=task, importance=confidence, timestamp=self._time, memory_type="prediction", emotional_valence=random.uniform(-1,1))
        self.memory.append(mem)
        return result

    def move(self):
        self.position += self.velocity

# ===================== ResourceRoom & NetworkMetrics =====================
class ResourceRoom:
    def __init__(self):
        self.resources: Dict[str, Dict[str, Any]] = {}

    def deposit(self, agent_id: str, info: Dict[str, Any]):
        self.resources[agent_id] = info

    def retrieve(self, agent_id: str) -> Dict[str, Any]:
        return self.resources.get(agent_id, {})

class NetworkMetrics:
    def __init__(self):
        self.energy_history = deque(maxlen=1000)
        self.confusion_history = deque(maxlen=1000)
        self.agent_statuses = []

    def update(self, agents: List["UnifiedAdaptiveAgent"]):
        if not agents:  # Handle empty agent list edge case
            return
        total_energy = sum(a.alive_node.energy for a in agents)
        avg_confusion = np.mean([a.confusion_level for a in agents])
        self.energy_history.append(total_energy)
        self.confusion_history.append(avg_confusion)
        self.agent_statuses = [a.status for a in agents]

    def health_score(self):
        if not self.energy_history: return 0.5
        e = np.mean(self.energy_history)
        c = np.mean(self.confusion_history)
        score = 0.5 * (min(e/100,1.0) + max(0,1.0-c))
        return round(score, 3)

# ===================== MazeMaster (Governance Layer) =====================
class MazeMaster:
    def __init__(self, confusion_escape_thresh=0.85, entropy_escape_thresh=1.5, soft_advice_thresh=0.65):
        self.interventions = 0
        self.confusion_escape_thresh = confusion_escape_thresh
        self.entropy_escape_thresh = entropy_escape_thresh
        self.soft_advice_thresh = soft_advice_thresh

    def quick_escape(self, agent: "UnifiedAdaptiveAgent"):
        agent.log_event("MazeMaster: Quick escape triggered.")
        agent.status = "escaped"
        return {"action": "escape", "message": f"Agent {agent.name} guided out by MazeMaster."}

    def psychologist(self, agent: "UnifiedAdaptiveAgent"):
        feedback = []
        if agent.confusion_level > 0.7:
            feedback.append("You seem overwhelmed; focus on one subproblem at a time.")
        if agent.entropy > 0.8:
            feedback.append("Too many branches—narrow the hypothesis set.")
        if agent.status == "stuck":
            feedback.append("It's okay to seek help; try reframing the objective.")
        if not feedback:
            feedback.append("You're doing well—continue.")
        agent.log_event("MazeMaster advice: " + " | ".join(feedback))
        return {"action": "advice", "message": feedback}

    def intervene(self, agent: "UnifiedAdaptiveAgent"):
        self.interventions += 1
        if (agent.confusion_level >= self.confusion_escape_thresh or
            agent.entropy >= self.entropy_escape_thresh or
            agent.status in {"stuck", "looping"}):
            return self.quick_escape(agent)
        if (agent.confusion_level >= self.soft_advice_thresh or
            agent.entropy >= self.soft_advice_thresh):
            return self.psychologist(agent)
        return {"action": "none"}

    def govern_agents(self, agents: List["UnifiedAdaptiveAgent"]):
        for agent in agents:
            action = self.intervene(agent)
            if action["action"] != "none":
                agent.log_event(f"MazeMaster intervention: {action['message']}")

# ===================== Unified Adaptive Agent =====================
class UnifiedAdaptiveAgent:
    def __init__(self, name: str, style: Dict[str, float], alive_node: AliveLoopNode, resource_room: ResourceRoom):
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

    def log_event(self, event: str):
        self.event_log.append(event)
        logger.info(f"[{self.name}] {event}")

    def reason(self, task: str) -> Dict[str, Any]:
        result = self.alive_node.safe_think(self.name, task)
        styled_result = self._apply_style_influence(result)
        key = f"{self.name}_reason_{len(self.knowledge_graph)}"
        self.knowledge_graph[key] = styled_result
        self.interaction_history.append(styled_result)
        self._update_confusion_and_entropy(styled_result)
        return styled_result

    def _apply_style_influence(self, base_result: Dict[str, Any]) -> Dict[str, Any]:
        styled = base_result.copy()
        insights = []
        for dim, val in self.style.items():
            if val > 0.7:
                insights.append(f"{dim.capitalize()} influence")
        styled["style_insights"] = insights
        return styled

    def _update_confusion_and_entropy(self, result: Dict[str, Any]):
        conf = result.get("confidence", 0.5)
        if conf < 0.4:
            self.confusion_level = min(1.0, self.confusion_level + 0.1)
        elif conf > 0.8:
            self.confusion_level = max(0.0, self.confusion_level - 0.1)
        self.style_cache.extend(result.get("style_insights", []))
        if self.style_cache:
            counts = Counter(self.style_cache)
            total = sum(counts.values())
            probs = [c/total for c in counts.values() if total]
            self.entropy = -sum(p * math.log2(p) for p in probs if p > 0)

    def teleport_to_resource_room(self, info: Dict[str, Any]):
        if self.resource_room:
            self.resource_room.deposit(self.agent_id, info)
            self.status = "in_resource_room"
            self.log_event("Teleported to ResourceRoom.")

    def retrieve_from_resource_room(self):
        if self.resource_room:
            info = self.resource_room.retrieve(self.agent_id)
            self.log_event("Retrieved info from ResourceRoom.")
            return info

    def get_state(self):
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "status": self.status,
            "confusion_level": self.confusion_level,
            "entropy": self.entropy,
            "knowledge_graph_size": len(self.knowledge_graph),
            "event_log": self.event_log,
        }

# ===================== Simulation Runner =====================
def run_labyrinth_simulation():
    logger.info("=== Unified Adaptive Labyrinth Simulation ===")
    resource_room = ResourceRoom()
    maze_master = MazeMaster()
    metrics = NetworkMetrics()

    # Create agents
    agents = [
        UnifiedAdaptiveAgent("AgentA", {"logic": 0.8, "creativity": 0.5}, AliveLoopNode((0,0), (0.5,0), 15.0, node_id=1), resource_room),
        UnifiedAdaptiveAgent("AgentB", {"creativity": 0.9, "analytical": 0.7}, AliveLoopNode((2,0), (0,0.5), 12.0, node_id=2), resource_room),
        UnifiedAdaptiveAgent("AgentC", {"logic": 0.6, "expressiveness": 0.8}, AliveLoopNode((0,2), (0.3,-0.2), 10.0, node_id=3), resource_room),
    ]
    capacitors = [CapacitorInSpace((1,1), capacity=8.0, initial_energy=3.0)]

    topics = ["Find exit", "Share wisdom", "Collaborate"]
    for step in range(1, 21):
        logger.info(f"\n--- Step {step} ---")
        for i, agent in enumerate(agents):
            topic = topics[step % len(topics)]
            agent.reason(f"{topic} at step {step}")
            agent.alive_node.move()
            if step % 5 == 0:
                agent.teleport_to_resource_room({"topic": topic, "step": step, "energy": agent.alive_node.energy})
                retrieved = agent.retrieve_from_resource_room()
        maze_master.govern_agents(agents)
        metrics.update(agents)
        logger.info(f"Network Health Score: {metrics.health_score()}")
        for capacitor in capacitors:
            logger.info(capacitor.status())
        for agent in agents:
            logger.info(f"{agent.name} state: {agent.get_state()}")
        time.sleep(0.2)

    logger.info("\n=== Simulation Complete ===")
    logger.info(f"Total MazeMaster interventions: {maze_master.interventions}")

if __name__ == "__main__":
    run_labyrinth_simulation()
