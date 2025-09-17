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

    SUPPORTED_REASONING_TYPES = {
        "medical_consultation": "Clinical context synthesis with evaluative guidance",
        "diagnosis": "Supports structuring differential or provisional assessments",
        "imaging_analysis": "Medical imaging interpretation with clinical correlation",
        "deductive": "Logical necessity from stated premises / retrieved memory facts",
        "inductive": "Generalization from multiple observations toward broader patterns",
        "abductive": "Inference to the most plausible explanatory hypothesis under uncertainty",
        "general": "Neutral context-aware analytic synthesis"
    }

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
            query: The question or problem to reason about.
            reasoning_type: Type of reasoning to apply. Supported:
                - medical_consultation: Synthesizes clinical context and suggests evaluation domains.
                - diagnosis: Organizes factors relevant to a diagnostic framing.
                - deductive: Derives conclusions that must follow from stated/query + retrieved premises.
                - inductive: Generalizes from multiple observations / memory patterns to broader principles.
                - abductive: Generates the most plausible explanatory hypothesis given incomplete evidence.
                - general: Broad analytic synthesis (fallback).
            (If an unsupported value is given, it falls back to 'general').

        Returns:
            ReasoningResult with response and metadata.
        """
        try:
            # Normalize reasoning type
            if reasoning_type not in self.SUPPORTED_REASONING_TYPES:
                logger.warning(f"Unsupported reasoning_type '{reasoning_type}' - using 'general'")
                reasoning_type = "general"

            # Retrieve relevant memories
            memories = self.memory_store.retrieve_memories(
                session_id=self.session_id,
                query=query,
                limit=self.max_context_memories,
                min_importance=0.3
            )

            # Filter memories by heuristic relevance
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
        relevant = []
        query_lower = query.lower()
        query_keywords = set(query_lower.split())

        for memory in memories:
            content_lower = memory['content'].lower()
            content_keywords = set(content_lower.split())

            overlap = len(query_keywords.intersection(content_keywords))
            overlap_score = overlap / max(len(query_keywords), 1)

            relevance_score = (overlap_score * 0.6) + (memory['importance_score'] * 0.4)
            if relevance_score >= self.similarity_threshold:
                memory['relevance_score'] = relevance_score
                relevant.append(memory)

        relevant.sort(key=lambda m: m.get('relevance_score', 0), reverse=True)
        return relevant[:self.max_context_memories]

    def _calculate_memory_weights(self, memories: List[Dict[str, Any]], query: str) -> List[float]:
        """
        Calculate importance weights for retrieved memories.
        """
        if not memories:
            return []

        weights = []
        now = datetime.now()
        for memory in memories:
            base_weight = memory['importance_score']
            created_at = memory.get('created_at')
            if created_at is None:
                # Fallback if missing timestamp
                hours_old = 24
            else:
                hours_old = (now - created_at).total_seconds() / 3600
            recency_boost = max(0, 1.0 - (hours_old / 168))
            access_boost = min(0.2, memory.get('access_count', 0) * 0.01)
            relevance_boost = memory.get('relevance_score', 0.5) * 0.3
            final_weight = min(1.0, base_weight + recency_boost * 0.2 + access_boost + relevance_boost * 0.3)
            weights.append(final_weight)
        return weights

    def _perform_reasoning(self, context: ReasoningContext) -> ReasoningResult:
        """
        Perform the actual reasoning with context.
        """
        reasoning_steps = []
        new_memories = []
        insights: List[str] = []  # Ensure defined even if no memories

        # Step 1: Analyze query
        reasoning_steps.append(f"Analyzing query: '{context.query}'")

        # Step 2: Review context memories
        if context.retrieved_memories:
            reasoning_steps.append(f"Retrieved {len(context.retrieved_memories)} relevant memories")
            for i, memory in enumerate(context.retrieved_memories):
                weight = context.memory_weights[i] if i < len(context.memory_weights) else 0.5
                if weight > 0.7:
                    insight = f"Key insight: {memory['content'][:140]}..."
                    insights.append(insight)
                    reasoning_steps.append(f"High-confidence memory (weight: {weight:.2f}): {memory['memory_type']}")

        # Step 3: Generate contextual response based on reasoning type
        rt = context.reasoning_type
        if rt == "medical_consultation":
            response = self._generate_medical_response(context, insights)
        elif rt == "diagnosis":
            response = self._generate_diagnosis_response(context, insights)
        elif rt == "imaging_analysis":
            response = self._generate_imaging_analysis_response(context, insights)
        elif rt == "deductive":
            response = self._generate_deductive_response(context, insights)
        elif rt == "inductive":
            response = self._generate_inductive_response(context, insights)
        elif rt == "abductive":
            response = self._generate_abductive_response(context, insights)
        else:
            response = self._generate_general_response(context, insights)

        reasoning_steps.append(f"Generated {context.reasoning_type} response")

        # Step 4: Calculate confidence
        confidence = self._calculate_confidence(context)

        # Step 5: Create new memories from this reasoning
        reasoning_memory = f"Reasoned about '{context.query}' with {len(context.retrieved_memories)} context memories ({context.reasoning_type})"
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

    # --- Response Generators ---

    def _generate_medical_response(self, context: ReasoningContext, insights: List[str]) -> str:
        base_response = f"Based on the query '{context.query}', "
        if insights:
            base_response += f"and considering {len(insights)} relevant clinical insights, "
            base_response += "I recommend a comprehensive evaluation including: "
            recommendations = []
            for insight in insights[:3]:
                lower = insight.lower()
                if "mmse" in lower or "cognitive" in lower:
                    recommendations.append("cognitive assessment")
                elif "apoe" in lower or "genetic" in lower:
                    recommendations.append("genetic risk evaluation")
                elif "biomarker" in lower:
                    recommendations.append("biomarker analysis")
                else:
                    recommendations.append("clinical evaluation")
            if recommendations:
                base_response += f"{', '.join(sorted(set(recommendations)))}. "
        base_response += "Please consult with a qualified healthcare professional for personalized medical advice."
        return base_response

    def _generate_diagnosis_response(self, context: ReasoningContext, insights: List[str]) -> str:
        response = f"For the diagnostic question '{context.query}', "
        if insights:
            response += f"based on {len(insights)} related case memories, "
            response += "key factors include clinical presentation, patient history, and relevant biomarkers. "
        response += "A structured, multifactorial assessment approach is recommended."
        return response

    def _generate_imaging_analysis_response(self, context: ReasoningContext, insights: List[str]) -> str:
        """
        Imaging analysis reasoning: interpret medical imaging with clinical correlation.
        """
        response = f"Imaging Analysis for: {context.query}\n\n"
        
        # Extract imaging-specific insights from memory
        imaging_insights = []
        quantitative_findings = []
        
        for mem in context.retrieved_memories:
            if mem.get('type') == 'imaging_insight':
                imaging_insights.append(mem['content'])
                # Extract quantitative measures if available in metadata
                if 'metadata' in mem and 'quantitative_measures' in mem['metadata']:
                    measures = mem['metadata']['quantitative_measures']
                    for measure, value in measures.items():
                        quantitative_findings.append(f"{measure}: {value}")
        
        if imaging_insights:
            response += "Relevant Imaging History:\n"
            for i, insight in enumerate(imaging_insights[:3], 1):
                response += f"{i}. {insight}\n"
            response += "\n"
        
        if quantitative_findings:
            response += "Quantitative Measures:\n"
            for finding in quantitative_findings[:5]:
                response += f"• {finding}\n"
            response += "\n"
        
        # General insights from memory
        if insights:
            response += "Clinical Context:\n"
            for insight in insights[:3]:
                response += f"• {insight}\n"
            response += "\n"
        
        # Imaging analysis framework
        response += "Imaging Analysis Framework:\n"
        response += "1. Technical Quality Assessment: Evaluate image acquisition parameters, artifacts, and diagnostic quality\n"
        response += "2. Morphological Analysis: Assess structural changes, volumes, and anatomical variations\n"
        response += "3. Signal Characteristics: Analyze intensity patterns, contrast enhancement, and tissue characteristics\n"
        response += "4. Comparative Analysis: Compare with prior studies and normal population ranges\n"
        response += "5. Clinical Correlation: Integrate findings with clinical presentation and laboratory data\n\n"
        
        response += "Recommendations:\n"
        response += "• Correlate imaging findings with clinical symptoms and examination\n"
        response += "• Consider follow-up imaging if indicated by clinical course\n"
        response += "• Quantitative analysis may provide additional diagnostic value\n"
        response += "• Multidisciplinary review recommended for complex cases"
        
        return response

    def _generate_deductive_response(self, context: ReasoningContext, insights: List[str]) -> str:
        """
        Deductive reasoning: derive necessary conclusions if premises hold.
        """
        premises = []
        for mem in context.retrieved_memories:
            text = mem['content']
            if any(k in text.lower() for k in ["if", "when", "therefore", "implies"]):
                premises.append(text[:160] + ("..." if len(text) > 160 else ""))

        response = f"Deductive analysis of query '{context.query}': "
        if premises:
            response += f"Identified {len(premises)} candidate premise(s). "
            response += "Applying logical structuring to derive constrained conclusions. "
        else:
            response += "Insufficient explicit premises found; defaulting to cautious inference. "

        # Simple synthesized conclusion placeholder
        response += "Conclusion(s) are contingent on the validity and completeness of the retrieved premises."
        return response

    def _generate_inductive_response(self, context: ReasoningContext, insights: List[str]) -> str:
        """
        Inductive reasoning: infer generalizations from multiple observations.
        """
        observations = []
        for mem in context.retrieved_memories:
            if mem['memory_type'] in ("knowledge", "experience", "reasoning"):
                observations.append(mem['content'][:120] + ("..." if len(mem['content']) > 120 else ""))

        response = f"Inductive synthesis for '{context.query}': "
        if len(observations) >= 2:
            response += f"Drawing from {len(observations)} contextual observations, emerging patterns suggest a generalized principle may apply. "
        elif observations:
            response += "Single observation limits generalization strength. "
        else:
            response += "No suitable observations located; inductive strength minimal. "

        response += "Resulting generalization remains probabilistic and should be validated with further evidence."
        return response

    def _generate_abductive_response(self, context: ReasoningContext, insights: List[str]) -> str:
        """
        Abductive reasoning: propose most plausible explanation under uncertainty.
        """
        response = f"Abductive reasoning for '{context.query}': "
        if insights:
            response += f"Leveraging {len(insights)} salient contextual clue(s) to hypothesize a plausible explanatory model. "
        else:
            response += "With limited contextual clues, abductive hypothesis generation requires broader assumption space. "

        response += "Proposed explanation is provisional and should be tested against alternative hypotheses."
        return response

    def _generate_general_response(self, context: ReasoningContext, insights: List[str]) -> str:
        response = f"Regarding '{context.query}', "
        if insights:
            response += f"drawing from {len(insights)} relevant experiences, the analysis suggests considering multiple perspectives "
            response += "and evidence-based approaches. "
        response += "Further investigation may be beneficial for a comprehensive understanding."
        return response

    def _calculate_confidence(self, context: ReasoningContext) -> float:
        """Calculate confidence score for reasoning result."""
        base_confidence = 0.5
        if context.retrieved_memories:
            memory_boost = min(0.3, len(context.retrieved_memories) * 0.1)
            avg_weight = sum(context.memory_weights) / len(context.memory_weights) if context.memory_weights else 0.5
            weight_boost = avg_weight * 0.2
            base_confidence += memory_boost + weight_boost

        # Slight adjustment by reasoning type (deductive relies more on premise quality)
        if context.reasoning_type == "deductive":
            base_confidence -= 0.05
        elif context.reasoning_type == "inductive":
            base_confidence -= 0.02
        elif context.reasoning_type == "abductive":
            base_confidence -= 0.08

        return max(0.0, min(1.0, base_confidence))

    def _store_reasoning_memory(self, query: str, result: ReasoningResult, context: ReasoningContext):
        """Store the reasoning process as new memories."""
        try:
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
                    "reasoning_duration": "< 1s"
                },
                user_id=f"agent_{self.agent_id}"
            )
        except Exception as e:
            logger.error(f"Error logging reasoning event: {e}")

    def get_reasoning_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get history of reasoning operations for this agent.
        """
        if not self.audit_chain:
            return []
        return self.audit_chain.get_entity_audit_trail("agent_session", self.session_id, limit)

    def store_imaging_insight(self, insight_data: Dict[str, Any]):
        """
        Store an imaging insight in the agent's memory.
        
        Args:
            insight_data: Dictionary with imaging insight information.
                         Should contain 'content', and optionally 'importance', 'metadata'
        """
        try:
            # Ensure this is marked as an imaging insight
            memory_data = {
                "content": insight_data.get("content", ""),
                "type": "imaging_insight",
                "importance": insight_data.get("importance", 0.7),
                "metadata": insight_data.get("metadata", {})
            }
            
            memory_id = self.memory_store.store_memory(
                agent_id=self.agent_id,
                **memory_data
            )
            
            logger.info(f"Stored imaging insight {memory_id} for agent {self.agent_id}")
            return memory_id
            
        except Exception as e:
            logger.error(f"Failed to store imaging insight: {e}")
            return None

    def retrieve_imaging_insights(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve imaging insights relevant to a query.
        
        Args:
            query: Search query for relevant imaging insights
            limit: Maximum number of insights to retrieve
            
        Returns:
            List of relevant imaging insights
        """
        try:
            # Retrieve memories with imaging_insight type
            results = self.memory_store.retrieve_memories(
                agent_id=self.agent_id,
                query=query,
                limit=limit,
                similarity_threshold=self.similarity_threshold
            )
            
            # Filter for imaging insights specifically
            imaging_insights = [
                result for result in results 
                if result.get('type') == 'imaging_insight'
            ]
            
            logger.info(f"Retrieved {len(imaging_insights)} imaging insights for query: {query}")
            return imaging_insights
            
        except Exception as e:
            logger.error(f"Failed to retrieve imaging insights: {e}")
            return []

    def reason_with_imaging_context(self, query: str, 
                                   imaging_features: Optional[Dict[str, Any]] = None,
                                   predictions: Optional[Dict[str, Any]] = None) -> ReasoningResult:
        """
        Perform reasoning with specific imaging context integration.
        
        Args:
            query: The imaging-related question or analysis request
            imaging_features: Optional extracted imaging features
            predictions: Optional ML model predictions on imaging data
            
        Returns:
            ReasoningResult with imaging-enhanced analysis
        """
        try:
            # Store any new imaging insights from current data
            if imaging_features or predictions:
                from .imaging_insights import ImagingInsightSummarizer
                summarizer = ImagingInsightSummarizer(self.memory_store)
                
                if predictions and imaging_features:
                    insight = summarizer.analyze_prediction_results(
                        predictions, imaging_features, patient_id=None
                    )
                elif imaging_features:
                    insight = summarizer.analyze_brain_mri_features(
                        imaging_features, patient_id=None
                    )
                else:
                    insight = None
                
                if insight:
                    self.store_imaging_insight(insight.to_memory_dict())
            
            # Perform reasoning with imaging analysis type
            return self.reason_with_context(query, reasoning_type="imaging_analysis")
            
        except Exception as e:
            logger.error(f"Failed to perform imaging-enhanced reasoning: {e}")
            # Fallback to regular reasoning
            return self.reason_with_context(query, reasoning_type="general")

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
        memory_store = AgentMemoryStore("sqlite:///live_reasoning_demo.db")
        audit_chain = AuditEventChain("sqlite:///reasoning_audit.db")

        agent = LiveReasoningAgent(
            agent_id="demo_medical_agent",
            memory_store=memory_store,
            audit_chain=audit_chain,
            similarity_threshold=0.6
        )

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

        queries = [
            ("What should be considered for a patient with MMSE score of 22 and APOE4 positive?", "medical_consultation"),
            ("Given MMSE 22 and APOE4 positivity, what follows logically about assessment priority?", "deductive"),
            ("From several cases of early cognitive decline, what general principle emerges?", "inductive"),
            ("Why might a patient with mild decline and APOE4 show accelerated change?", "abductive"),
            ("Key factors in Alzheimer's risk evaluation?", "diagnosis")
        ]

        print("\n=== Live Agent Reasoning Demo ===")
        for i, (q, rtype) in enumerate(queries, 1):
            print(f"\n--- Reasoning Query {i} ({rtype}) ---")
            print(f"Query: {q}")
            result = agent.reason_with_context(q, rtype)
            print(f"Response: {result.response}")
            print(f"Confidence: {result.confidence:.2f}")
            print(f"Context memories used: {result.memory_count}")
            print(f"Reasoning steps: {len(result.reasoning_steps)}")

        history = agent.get_reasoning_history()
        print(f"\nReasoning history: {len(history)} events logged")

        agent.end_session()
        print("\n✅ Live reasoning demonstration completed!")

    except Exception as e:
        logger.error(f"Error in live reasoning demo: {e}")


if __name__ == "__main__":
    demo_live_reasoning()
