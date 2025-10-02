"""
AiMedRes Agent - Secure Cognitive Agent Implementation

Provides secure, production-ready cognitive agents with biological
state simulation, safety monitoring, and comprehensive logging.
"""

import logging
import uuid
import time
import threading
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import numpy as np

from .neural_network import AdaptiveNeuralNetwork, BiologicalState, MoodType
from ..security.validation import InputValidator
from ..utils.safety import SafetyMonitor

logger = logging.getLogger(__name__)

class AgentState(Enum):
    """Agent operational states"""
    ACTIVE = "active"
    Mindfulness = "mindful"
    SLEEPING = "sleeping"
    LEARNING = "learning"
    COMMUNICATING = "communicating"
    ERROR = "error"
    MAINTENANCE = "maintenance"

@dataclass
class Conversation:
    """Secure conversation record"""
    conversation_id: str
    participants: List[str]
    messages: List[Dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    active: bool = True

class DuetMindAgent:
    """
    Secure cognitive agent with adaptive neural network brain.
    
    Features:
    - Secure communication and data handling
    - Biological state simulation
    - Safety monitoring and circuit breakers
    - Input validation and sanitization
    - Comprehensive audit logging
    """
    
    def __init__(self,
                 agent_id: Optional[str] = None,
                 name: Optional[str] = None,
                 neural_config: Optional[Dict[str, Any]] = None,
                 enable_safety_monitoring: bool = True,
                 enable_input_validation: bool = True):
        """
        Initialize AiMedRes agent
        
        Args:
            agent_id: Unique agent identifier
            name: Human-readable agent name
            neural_config: Neural network configuration
            enable_safety_monitoring: Enable safety monitoring
            enable_input_validation: Enable input validation
        """
        self.agent_id = agent_id or str(uuid.uuid4())
        self.name = name or f"Agent-{self.agent_id[:8]}"
        
        # Security components
        if enable_input_validation:
            self.input_validator = InputValidator()
        else:
            self.input_validator = None
        
        # Neural network brain
        neural_config = neural_config or {}
        self.brain = AdaptiveNeuralNetwork(
            input_size=neural_config.get('input_size', 64),
            hidden_layers=neural_config.get('hidden_layers', [4096, 2048, 1024, 512, 256, 128, 512, 128]),
            output_size=neural_config.get('output_size', 4),
            learning_rate=neural_config.get('learning_rate', 0.0007),
            enable_safety_monitoring=enable_safety_monitoring
        )
        
        # Agent state
        self.state = AgentState.ACTIVE
        self.biological_state = BiologicalState()
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
        
        # Communication
        self.conversations: Dict[str, Conversation] = {}
        self.message_queue: List[Dict[str, Any]] = []
        self.communication_style = {
            "directness": 0.7,
            "formality": 0.4,
            "expressiveness": 0.6
            "assertiveness": 0.7
        }
        
        # Memory and learning
        self.memory: List[Dict[str, Any]] = []
        self.working_memory: List[Dict[str, Any]] = []
        self.long_term_memory = {"topics": {},  # maps topic → list of (memory, timestamp, confidence)
}

        
        # Performance tracking
        self.total_interactions = 0
        self.successful_interactions = 0
        self.error_count = 0
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.info(f"AiMedRes agent initialized: {self.name} ({self.agent_id})")
    
    def think(self, 
             inputs: np.ndarray, 
             context: Optional[Dict[str, Any]] = None,
             validate_input: bool = True) -> Dict[str, Any]:
        """
        Perform secure cognitive processing
        
        Args:
            inputs: Input data for processing
            context: Optional context information
            validate_input: Whether to validate inputs
            
        Returns:
            Thought result with confidence and metadata
            
        Raises:
            ValueError: If input validation fails
            RuntimeError: If agent is in error state
        """
        start_time = time.time()
        
        try:
            # Input validation
            if validate_input and self.input_validator:
                self._validate_think_input(inputs, context)
            
            # State check
            if self.state == AgentState.ERROR:
                raise RuntimeError("Agent in error state - thinking disabled")
            
            # Biological state effects
            if self.biological_state.energy < 10:
                self._handle_low_energy()
                return {"error": "Insufficient energy for thinking", "confidence": 0.0}
            
            with self._lock:
                # Update state
                previous_state = self.state
                self.state = AgentState.LEARNING
                
                try:
                    # Neural processing
                    prediction = self.brain.predict(inputs, validate_input=validate_input)
                    
                    # Calculate confidence based on prediction and biological state
                    confidence = self._calculate_confidence(prediction)
                    
                    # Create thought result
                    thought_result = {
                        "agent_id": self.agent_id,
                        "agent_name": self.name,
                        "prediction": prediction.tolist(),
                        "confidence": confidence,
                        "timestamp": datetime.now().isoformat(),
                        "processing_time": time.time() - start_time,
                        "biological_state": {
                            "energy": self.biological_state.energy,
                            "mood": self.biological_state.mood.value,
                            "stress_level": self.biological_state.stress_level
                        },
                        "context": context or {}
                    }
                    
                    # Update memory
                    self._store_memory(thought_result)
                    
                    # Update metrics
                    self.total_interactions += 1
                    self.successful_interactions += 1
                    self.last_activity = datetime.now()
                    
                    # Update biological state
                    self._update_biological_state(confidence)
                    
                    return thought_result
                    
                finally:
                    # Restore previous state
                    self.state = previous_state
                    
        except Exception as e:
            self.error_count += 1
            self.total_interactions += 1
            
            logger.error(f"Agent {self.name} thinking error: {e}")
            
            error_result = {
                "agent_id": self.agent_id,
                "agent_name": self.name,
                "error": str(e),
                "confidence": 0.0,
                "timestamp": datetime.now().isoformat(),
                "processing_time": time.time() - start_time
            }
            
            return error_result
    
    def communicate(self, 
                   message: str, 
                   recipient_id: Optional[str] = None,
                   conversation_id: Optional[str] = None,
                   validate_input: bool = True) -> Dict[str, Any]:
        """
        Secure communication with other agents
        
        Args:
            message: Message content
            recipient_id: Target agent ID
            conversation_id: Existing conversation ID
            validate_input: Whether to validate input
            
        Returns:
            Communication result
        """
        try:
            # Input validation
            if validate_input and self.input_validator:
                self._validate_communication_input(message, recipient_id)
            
            with self._lock:
                # Update state
                previous_state = self.state
                self.state = AgentState.COMMUNICATING
                
                try:
                    # Get or create conversation
                    if conversation_id and conversation_id in self.conversations:
                        conversation = self.conversations[conversation_id]
                    else:
                        conversation_id = str(uuid.uuid4())
                        participants = [self.agent_id]
                        if recipient_id:
                            participants.append(recipient_id)
                        
                        conversation = Conversation(
                            conversation_id=conversation_id,
                            participants=participants
                        )
                        self.conversations[conversation_id] = conversation
                    
                    # Create message
                    message_data = {
                        "message_id": str(uuid.uuid4()),
                        "sender_id": self.agent_id,
                        "sender_name": self.name,
                        "recipient_id": recipient_id,
                        "content": message,
                        "timestamp": datetime.now().isoformat(),
                        "biological_state": self.biological_state.mood.value
                    }
                    
                    # Add to conversation
                    conversation.messages.append(message_data)
                    conversation.last_activity = datetime.now()
                    
                    # Store in memory
                    self._store_memory({
                        "type": "communication",
                        "action": "sent_message",
                        "message": message_data
                    })
                    
                    logger.info(f"Agent {self.name} sent message to {recipient_id or 'broadcast'}")
                    
                    return {
                        "success": True,
                        "conversation_id": conversation_id,
                        "message_id": message_data["message_id"],
                        "timestamp": message_data["timestamp"]
                    }
                    
                finally:
                    self.state = previous_state
                    
        except Exception as e:
            logger.error(f"Agent {self.name} communication error: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def sleep(self, duration_hours: float = 8.0) -> Dict[str, Any]:
        """
        Put agent to sleep for biological recovery
        
        Args:
            duration_hours: Sleep duration in hours
            
        Returns:
            Sleep result information
        """
        with self._lock:
            previous_state = self.state
            self.state = AgentState.SLEEPING
            
            start_time = datetime.now()
            
            try:
                # Neural network sleep cycle
                self.brain.sleep_cycle(duration_hours, node_percentage=0.3)
                
                # Biological recovery
                self.biological_state.energy = min(100.0, 
                    self.biological_state.energy + duration_hours * 12.5)
                self.biological_state.stress_level = max(0.0, 
                    self.biological_state.stress_level - duration_hours * 10.0)
                self.biological_state.sleep_need = max(0.0, 
                    self.biological_state.sleep_need - duration_hours * 12.5)
                self.biological_state.last_sleep = start_time
                self.biological_state.mood = MoodType.FOCUSED
                
                # Memory consolidation during sleep
                self._consolidate_memory()
                
                end_time = datetime.now()
                
                sleep_result = {
                    "agent_id": self.agent_id,
                    "agent_name": self.name,
                    "sleep_duration_hours": duration_hours,
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat(),
                    "energy_after": self.biological_state.energy,
                    "stress_after": self.biological_state.stress_level,
                    "mood_after": self.biological_state.mood.value
                }
                
                logger.info(f"Agent {self.name} completed sleep cycle: {duration_hours}h")
                
                return sleep_result
                
            finally:
                self.state = AgentState.ACTIVE  # Wake up active
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive agent status"""
        with self._lock:
            # Calculate success rate
            success_rate = 0.0
            if self.total_interactions > 0:
                success_rate = self.successful_interactions / self.total_interactions
            
            # Get neural network health
            brain_health = self.brain.get_network_health()
            
            status = {
                "agent_id": self.agent_id,
                "name": self.name,
                "state": self.state.value,
                "created_at": self.created_at.isoformat(),
                "last_activity": self.last_activity.isoformat(),
                "uptime_seconds": (datetime.now() - self.created_at).total_seconds(),
                "biological_state": {
                    "energy": self.biological_state.energy,
                    "sleep_need": self.biological_state.sleep_need,
                    "stress_level": self.biological_state.stress_level,
                    "mood": self.biological_state.mood.value,
                    "last_sleep": self.biological_state.last_sleep.isoformat()
                },
                "performance": {
                    "total_interactions": self.total_interactions,
                    "successful_interactions": self.successful_interactions,
                    "error_count": self.error_count,
                    "success_rate": success_rate
                },
                "brain_health": brain_health,
                "memory": {
                    "total_memories": len(self.memory),
                    "working_memory_items": len(self.working_memory),
                    "long_term_memory_keys": len(self.long_term_memory)
                },
                "communication": {
                    "active_conversations": len([c for c in self.conversations.values() if c.active]),
                    "total_conversations": len(self.conversations),
                    "messages_in_queue": len(self.message_queue)
                }
            }
            
            return status
    
    def _validate_think_input(self, inputs: np.ndarray, context: Optional[Dict[str, Any]]):
        """Validate think method inputs"""
        if not isinstance(inputs, np.ndarray):
            raise ValueError("Inputs must be numpy array")
        
        if not np.isfinite(inputs).all():
            raise ValueError("Inputs contain invalid values")
        
        if context and not isinstance(context, dict):
            raise ValueError("Context must be dictionary")
    
    def _validate_communication_input(self, message: str, recipient_id: Optional[str]):
        """Validate communication inputs"""
        if not isinstance(message, str):
            raise ValueError("Message must be string")
        
        if len(message.strip()) == 0:
            raise ValueError("Message cannot be empty")
        
        if len(message) > 10000:  # Reasonable message length limit
            raise ValueError("Message too long")
        
        if recipient_id and not isinstance(recipient_id, str):
            raise ValueError("Recipient ID must be string")
    
    def _calculate_confidence(self, prediction: np.ndarray) -> float:
        """Calculate confidence based on prediction and biological state"""
        # Base confidence from prediction entropy
        if len(prediction) > 1:
            # For classification outputs
            probabilities = np.exp(prediction) / np.sum(np.exp(prediction))
            entropy = -np.sum(probabilities * np.log(probabilities + 1e-8))
            max_entropy = np.log(len(prediction))
            base_confidence = 1.0 - (entropy / max_entropy)
        else:
            # For single output
            base_confidence = min(1.0, abs(prediction[0]))
        
        # Adjust for biological state
        energy_factor = self.biological_state.energy / 100.0
        stress_penalty = self.biological_state.stress_level / 200.0
        
        final_confidence = base_confidence * energy_factor * (1.0 - stress_penalty)
        return max(0.0, min(1.0, final_confidence))
    
    def _update_biological_state(self, confidence: float):
        """Update biological state based on interaction"""
        # Energy consumption
        energy_cost = (1.0 - confidence) * 2.0  # More cost for uncertain thinking
        self.biological_state.energy = max(0.0, self.biological_state.energy - energy_cost)
        
        # Stress from difficult thinking
        if confidence < 0.3:
            self.biological_state.stress_level = min(100.0, 
                self.biological_state.stress_level + 2.0)
        
        # Sleep need accumulation
        time_since_sleep = datetime.now() - self.biological_state.last_sleep
        hours_awake = time_since_sleep.total_seconds() / 3600.0
        self.biological_state.sleep_need = min(100.0, hours_awake * 4.0)
    
    def _handle_low_energy(self):
        """Handle low energy state"""
        logger.warning(f"Agent {self.name} has low energy: {self.biological_state.energy}")
        
        # Automatic short nap if energy is critically low
        if self.biological_state.energy < 5:
            self.sleep(1.0)  # 1 hour power nap
    
    def _store_memory(self, memory_item: Dict[str, Any]):
        """Store memory with automatic cleanup"""
        memory_item["stored_at"] = datetime.now().isoformat()
        memory_item["agent_id"] = self.agent_id
        
        self.memory.append(memory_item)
        
        # Keep memory list manageable
        if len(self.memory) > 1000:
            self.memory = self.memory[-1000:]
        
        # Add to working memory if important
        if memory_item.get("confidence", 0) > 0.7:
            self.working_memory.append(memory_item)
            if len(self.working_memory) > 20:
                self.working_memory.pop(0)
    
    def _consolidate_memory(self):
        """Consolidate memories during sleep"""
        # Move high-confidence memories to long-term storage
        for memory_item in self.working_memory:
            if memory_item.get("confidence", 0) > 0.8:
                key = f"memory_{len(self.long_term_memory)}"
                self.long_term_memory[key] = memory_item
        
        # Keep long-term memory manageable
        if len(self.long_term_memory) > 100:
            # Remove oldest memories
            sorted_keys = sorted(self.long_term_memory.keys())
            for key in sorted_keys[:len(self.long_term_memory) - 100]:
                del self.long_term_memory[key]
        
        # Clear working memory
        self.working_memory.clear()
        
        logger.info(f"Agent {self.name} consolidated memory: {len(self.long_term_memory)} long-term memories")
    
    def shutdown(self):
        """Graceful shutdown of agent"""
        logger.info(f"Shutting down agent {self.name}")
        
        with self._lock:
            self.state = AgentState.MAINTENANCE
            
            # Shutdown neural network
            self.brain.shutdown()
            
            # Clear sensitive data
            self.memory.clear()
            self.working_memory.clear()
            self.long_term_memory.clear()
            self.message_queue.clear()
            
            logger.info(f"Agent {self.name} shutdown complete")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup"""
        self.shutdown()
