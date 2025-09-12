#!/usr/bin/env python3
"""
Live Agent Reasoning with pgvector Semantic Memory Integration.
Enhances agent reasoning loops with real-time semantic memory retrieval and context injection.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
import json
import numpy as np
from dataclasses import dataclass

from .embed_memory import AgentMemoryStore
from ..audit.event_chain import AuditEventChain

# Setup logging  
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ReasoningContext:
    """Context information for reasoning operations."""
    agent_id: str
    session_id: str
    query: str
    retrieved_memories: List[Dict[str, Any]]
    memory_weights: List[float]
    reasoning_type: str = "contextual"
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)


@dataclass
class ReasoningResult:
    """Result of a reasoning operation."""
    response: str
    confidence: float
    context_used: bool
    memory_count: int
    reasoning_steps: List[str]
    new_memories: List[str]
    session_id: str
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)


class LiveReasoningAgent:
    """
    Enhanced reasoning agent with integrated pgvector semantic memory.
    """
    
    def __init__(self, agent_id: str, memory_store: AgentMemoryStore,
                 audit_chain: Optional[AuditEventChain] = None,
                 similarity_threshold: float = 0.7,
                 max_context_memories: int = 5):
        """
        Initialize live reasoning agent.
        
        Args:
            agent_id: Unique identifier for the agent
            memory_store: Vector memory store for semantic retrieval
            audit_chain: Audit system for logging reasoning operations
            similarity_threshold: Minimum similarity for memory retrieval
            max_context_memories: Maximum number of memories to use in context
        """
        self.agent_id = agent_id
        self.memory_store = memory_store
        self.audit_chain = audit_chain
        self.similarity_threshold = similarity_threshold
        self.max_context_memories = max_context_memories
        
        # Create agent session
        self.session_id = self.memory_store.create_session(
            agent_name=agent_id,
            metadata={"reasoning_agent": True, "created": datetime.now(timezone.utc).isoformat()}
        )
        
        logger.info(f"Initialized LiveReasoningAgent {agent_id} with session {self.session_id}")
    
    def reason_with_context(self, query: str, reasoning_type: str = "medical_consultation") -> ReasoningResult:
        """
        Perform reasoning with semantic memory context retrieval.
        
        Args:
            query: The question or problem to reason about
            reasoning_type: Type of reasoning (medical_consultation, diagnosis, etc.)
            
        Returns:
            ReasoningResult with response and metadata
        """
        try:
            # Retrieve relevant memories
            memories = self.memory_store.retrieve_memories(
                session_id=self.session_id,
                query=query,
                limit=self.max_context_memories,
                min_importance=0.3
            )
            
            # Filter memories by similarity (if available in future pgvector implementation)
            relevant_memories = self._filter_relevant_memories(memories, query)
            
            # Create reasoning context
            context = ReasoningContext(
                agent_id=self.agent_id,
                session_id=self.session_id,
                query=query,
                retrieved_memories=relevant_memories,
                memory_weights=self._calculate_memory_weights(relevant_memories, query),
                reasoning_type=reasoning_type
            )
            
            # Perform contextual reasoning
            reasoning_result = self._perform_reasoning(context)
            
            # Store new reasoning as memory
            self._store_reasoning_memory(query, reasoning_result, context)
            
            # Log audit event
            if self.audit_chain:
                self._log_reasoning_event(context, reasoning_result)
            
            return reasoning_result
            
        except Exception as e:
            logger.error(f"Error in contextual reasoning: {e}")
            return ReasoningResult(
                response=f"Error in reasoning: {str(e)}",
                confidence=0.0,
                context_used=False,
                memory_count=0,
                reasoning_steps=["error"],
                new_memories=[],
                session_id=self.session_id
            )
    
    def _filter_relevant_memories(self, memories: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """
        Filter memories for relevance to the current query.
        
        Args:
            memories: Retrieved memories
            query: Current query
            
        Returns:
            Filtered list of relevant memories
        """
        # In a full pgvector implementation, this would use cosine similarity
        # For now, we use importance scoring and content matching
        relevant = []
        
        query_lower = query.lower()
        query_keywords = set(query_lower.split())
        
        for memory in memories:
            content_lower = memory['content'].lower()
            content_keywords = set(content_lower.split())
            
            # Simple keyword overlap scoring
            overlap = len(query_keywords.intersection(content_keywords))
            overlap_score = overlap / max(len(query_keywords), 1)
            
            # Combine with importance score
            relevance_score = (overlap_score * 0.6) + (memory['importance_score'] * 0.4)
            
            if relevance_score >= self.similarity_threshold:
                memory['relevance_score'] = relevance_score
                relevant.append(memory)
        
        # Sort by relevance score
        relevant.sort(key=lambda m: m.get('relevance_score', 0), reverse=True)
        
        return relevant[:self.max_context_memories]
    
    def _calculate_memory_weights(self, memories: List[Dict[str, Any]], query: str) -> List[float]:
        """
        Calculate importance weights for retrieved memories.
        
        Args:
            memories: Retrieved memories
            query: Current query
            
        Returns:
            List of weights (0-1) for each memory
        """
        if not memories:
            return []
        
        weights = []
        for memory in memories:
            # Base weight from importance score
            base_weight = memory['importance_score']
            
            # Boost weight for recent memories
            hours_old = (datetime.now() - memory['created_at']).total_seconds() / 3600
            recency_boost = max(0, 1.0 - (hours_old / 168))  # Decay over 1 week
            
            # Boost weight for frequently accessed memories
            access_boost = min(0.2, memory['access_count'] * 0.01)
            
            # Boost for relevance score if available
            relevance_boost = memory.get('relevance_score', 0.5) * 0.3
            
            final_weight = min(1.0, base_weight + recency_boost * 0.2 + access_boost + relevance_boost * 0.3)
            weights.append(final_weight)
        
        return weights
    
    def _perform_reasoning(self, context: ReasoningContext) -> ReasoningResult:
        """
        Perform the actual reasoning with context.
        
        Args:
            context: Reasoning context with retrieved memories
            
        Returns:
            ReasoningResult
        """
        reasoning_steps = []
        new_memories = []
        
        # Step 1: Analyze query
        reasoning_steps.append(f"Analyzing query: '{context.query}'")
        
        # Step 2: Review context memories
        if context.retrieved_memories:
            reasoning_steps.append(f"Retrieved {len(context.retrieved_memories)} relevant memories")
            
            # Extract key insights from memories
            insights = []
            for i, memory in enumerate(context.retrieved_memories):
                weight = context.memory_weights[i] if i < len(context.memory_weights) else 0.5
                
                if weight > 0.7:  # High-importance memory
                    insight = f"Key insight: {memory['content'][:100]}..."
                    insights.append(insight)
                    reasoning_steps.append(f"High-confidence memory (weight: {weight:.2f}): {memory['memory_type']}")
        
        # Step 3: Generate contextual response based on reasoning type
        if context.reasoning_type == "medical_consultation":
            response = self._generate_medical_response(context, insights)
        elif context.reasoning_type == "diagnosis":
            response = self._generate_diagnosis_response(context, insights)
        else:
            response = self._generate_general_response(context, insights)
        
        reasoning_steps.append(f"Generated {context.reasoning_type} response")
        
        # Step 4: Calculate confidence based on context quality
        confidence = self._calculate_confidence(context)
        
        # Step 5: Create new memories from this reasoning
        reasoning_memory = f"Reasoned about '{context.query}' with {len(context.retrieved_memories)} context memories"
        new_memories.append(reasoning_memory)
        
        return ReasoningResult(
            response=response,
            confidence=confidence,
            context_used=len(context.retrieved_memories) > 0,
            memory_count=len(context.retrieved_memories),
            reasoning_steps=reasoning_steps,
            new_memories=new_memories,
            session_id=context.session_id
        )
    
    def _generate_medical_response(self, context: ReasoningContext, insights: List[str]) -> str:
        """Generate medical consultation response."""
        base_response = f"Based on the query '{context.query}', "
        
        if insights:
            base_response += f"and considering {len(insights)} relevant clinical insights, "
            base_response += "I recommend a comprehensive evaluation including: "
            
            # Extract medical recommendations from insights
            recommendations = []
            for insight in insights[:3]:  # Top 3 insights
                if "MMSE" in insight or "cognitive" in insight.lower():
                    recommendations.append("cognitive assessment")
                elif "APOE" in insight or "genetic" in insight.lower():
                    recommendations.append("genetic risk evaluation")
                elif "biomarker" in insight.lower():
                    recommendations.append("biomarker analysis")
                else:
                    recommendations.append("clinical evaluation")
            
            if recommendations:
                base_response += f"{', '.join(set(recommendations))}. "
        
        base_response += "Please consult with a qualified healthcare professional for personalized medical advice."
        return base_response
    
    def _generate_diagnosis_response(self, context: ReasoningContext, insights: List[str]) -> str:
        """Generate diagnostic response."""
        response = f"For the diagnostic question '{context.query}', "
        
        if insights:
            response += f"based on {len(insights)} related case memories, "
            response += "the key factors to consider include clinical presentation, "
            response += "patient history, and relevant biomarkers. "
        
        response += "A multifactorial assessment approach is recommended."
        return response
    
    def _generate_general_response(self, context: ReasoningContext, insights: List[str]) -> str:
        """Generate general reasoning response."""
        response = f"Regarding '{context.query}', "
        
        if insights:
            response += f"drawing from {len(insights)} relevant experiences, "
            response += "the analysis suggests considering multiple perspectives "
            response += "and evidence-based approaches. "
        
        response += "Further investigation may be beneficial for a comprehensive understanding."
        return response
    
    def _calculate_confidence(self, context: ReasoningContext) -> float:
        """Calculate confidence score for reasoning result."""
        base_confidence = 0.5
        
        # Boost confidence based on memory count and quality
        if context.retrieved_memories:
            memory_boost = min(0.3, len(context.retrieved_memories) * 0.1)
            
            # Average weight of retrieved memories
            avg_weight = sum(context.memory_weights) / len(context.memory_weights) if context.memory_weights else 0.5
            weight_boost = avg_weight * 0.2
            
            base_confidence += memory_boost + weight_boost
        
        return min(1.0, base_confidence)
    
    def _store_reasoning_memory(self, query: str, result: ReasoningResult, context: ReasoningContext):
        """Store the reasoning process as new memories."""
        try:
            # Store the reasoning result
            self.memory_store.store_memory(
                session_id=context.session_id,
                content=f"Query: {query}. Response: {result.response[:200]}...",
                memory_type="reasoning",
                importance=result.confidence,
                metadata={
                    "reasoning_type": context.reasoning_type,
                    "context_memories_used": len(context.retrieved_memories),
                    "confidence": result.confidence
                }
            )
            
            # Store any new insights as separate memories
            for memory_content in result.new_memories:
                self.memory_store.store_memory(
                    session_id=context.session_id,
                    content=memory_content,
                    memory_type="experience",
                    importance=0.6
                )
                
        except Exception as e:
            logger.error(f"Error storing reasoning memory: {e}")
    
    def _log_reasoning_event(self, context: ReasoningContext, result: ReasoningResult):
        """Log reasoning event to audit chain."""
        try:
            self.audit_chain.log_event(
                event_type="agent_reasoning",
                entity_type="agent_session",
                entity_id=context.session_id,
                event_data={
                    "agent_id": self.agent_id,
                    "query": context.query,
                    "reasoning_type": context.reasoning_type,
                    "memories_retrieved": len(context.retrieved_memories),
                    "confidence": result.confidence,
                    "context_used": result.context_used,
                    "reasoning_duration": "< 1s"  # Would measure actual duration in production
                },
                user_id=f"agent_{self.agent_id}"
            )
        except Exception as e:
            logger.error(f"Error logging reasoning event: {e}")
    
    def get_reasoning_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get history of reasoning operations for this agent.
        
        Args:
            limit: Maximum number of reasoning events to return
            
        Returns:
            List of reasoning events
        """
        if not self.audit_chain:
            return []
        
        return self.audit_chain.get_entity_audit_trail("agent_session", self.session_id, limit)
    
    def end_session(self):
        """End the agent reasoning session."""
        self.memory_store.end_session(self.session_id)
        logger.info(f"Ended reasoning session for agent {self.agent_id}")


def demo_live_reasoning():
    """
    Demonstrate live agent reasoning with semantic memory integration.
    """
    logger.info("Starting live agent reasoning demonstration...")
    
    try:
        # Initialize components
        memory_store = AgentMemoryStore("sqlite:///live_reasoning_demo.db")
        audit_chain = AuditEventChain("sqlite:///reasoning_audit.db")
        
        # Create reasoning agent
        agent = LiveReasoningAgent(
            agent_id="demo_medical_agent",
            memory_store=memory_store,
            audit_chain=audit_chain,
            similarity_threshold=0.6
        )
        
        # Pre-populate with some medical knowledge
        medical_knowledge = [
            {
                "content": "MMSE scores below 24 may indicate cognitive impairment requiring further evaluation",
                "type": "knowledge",
                "importance": 0.9
            },
            {
                "content": "APOE4 genotype significantly increases Alzheimer's disease risk, especially in combination with other factors",
                "type": "knowledge", 
                "importance": 0.85
            },
            {
                "content": "Early intervention and lifestyle modifications can help slow cognitive decline",
                "type": "knowledge",
                "importance": 0.8
            },
            {
                "content": "Comprehensive assessment should include cognitive testing, biomarkers, and family history",
                "type": "reasoning",
                "importance": 0.75
            }
        ]
        
        for knowledge in medical_knowledge:
            memory_store.store_memory(
                session_id=agent.session_id,
                content=knowledge["content"],
                memory_type=knowledge["type"],
                importance=knowledge["importance"]
            )
        
        # Demonstrate reasoning with different queries
        queries = [
            "What should be considered for a patient with MMSE score of 22 and APOE4 positive?",
            "How should we approach early-stage cognitive assessment?",
            "What are the key factors in Alzheimer's risk evaluation?"
        ]
        
        print("\n=== Live Agent Reasoning Demo ===")
        
        for i, query in enumerate(queries, 1):
            print(f"\n--- Reasoning Query {i} ---")
            print(f"Query: {query}")
            
            result = agent.reason_with_context(query, "medical_consultation")
            
            print(f"Response: {result.response}")
            print(f"Confidence: {result.confidence:.2f}")
            print(f"Context memories used: {result.memory_count}")
            print(f"Reasoning steps: {len(result.reasoning_steps)}")
        
        # Show reasoning history
        history = agent.get_reasoning_history()
        print(f"\nReasoning history: {len(history)} events logged")
        
        # End session
        agent.end_session()
        
        print("\n✅ Live reasoning demonstration completed!")
        
    except Exception as e:
        logger.error(f"Error in live reasoning demo: {e}")


if __name__ == "__main__":
    demo_live_reasoning()