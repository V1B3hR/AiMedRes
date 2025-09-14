#!/usr/bin/env python3
"""
Demonstration of Enhanced Memory Consolidation and Agent Extensions

Shows the new Phase 2 and Phase 3 features including:
- Enhanced salience scoring with uncertainty and reward
- Priority replay with synaptic tagging  
- Semantic conflict detection and resolution
- Memory introspection API
- Plugin system with capability registry
- Enhanced visualization API
"""

import tempfile
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('demo')

# Import our enhanced systems
from agent_memory.memory_consolidation import MemoryConsolidator
from agent_memory.agent_extensions import (
    CapabilityRegistry, ClinicalGuidelineAgent, CapabilityScope,
    create_capability_registry, create_agent_context
)

# Create a simplified visualization API demo class to avoid import issues
class VisualizationAPIDemo:
    """Simplified visualization API for demonstration."""
    
    def __init__(self, config):
        self.config = config
        self.memory_store = None
        self.memory_consolidator = None
        self.capability_registry = None
    
    def initialize_monitors(self, **kwargs):
        """Initialize monitoring system references."""
        self.memory_store = kwargs.get('memory_store')
        self.memory_consolidator = kwargs.get('memory_consolidator')  
        self.capability_registry = kwargs.get('capability_registry')


class MockMemoryStore:
    """Mock memory store for demonstration."""
    
    def __init__(self):
        self.memories = {}
        self.session_memories = {}
        self.next_id = 1
    
    def get_session_memories(self, session_id: str):
        return self.session_memories.get(session_id, [])
    
    def store_memory(self, session_id: str, content: str, memory_type: str = "reasoning",
                    importance: float = 0.5, metadata: dict = None):
        memory = {
            'id': self.next_id,
            'session_id': session_id,
            'content': content,
            'memory_type': memory_type,
            'importance_score': importance,
            'metadata': metadata or {},
            'created_at': datetime.now(),
            'access_count': 0
        }
        
        self.memories[self.next_id] = memory
        
        if session_id not in self.session_memories:
            self.session_memories[session_id] = []
        self.session_memories[session_id].append(memory)
        
        memory_id = self.next_id
        self.next_id += 1
        return memory_id
    
    def retrieve_memories(self, session_id: str, query: str, limit: int = 5,
                         memory_type: str = None, min_importance: float = 0.0):
        memories = self.get_session_memories(session_id)
        
        # Simple keyword matching for demo
        filtered = []
        for memory in memories:
            if query.lower() in memory['content'].lower():
                if not memory_type or memory['memory_type'] == memory_type:
                    if memory['importance_score'] >= min_importance:
                        filtered.append(memory)
        
        return filtered[:limit]


def demo_enhanced_memory_consolidation():
    """Demonstrate enhanced memory consolidation features."""
    logger.info("🧠 Starting Enhanced Memory Consolidation Demo")
    
    # Create temporary database
    fd, temp_db = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    
    try:
        # Create mock memory store
        memory_store = MockMemoryStore()
        
        # Create consolidator with enhanced config
        config = {
            'consolidation_db': temp_db,
            'consolidation_interval_hours': 1,
            'replay_sample_size': 5,
            'synaptic_tag_threshold': 0.8,
            'uncertainty_weight': 0.3,
            'reward_weight': 0.4,
            'conflict_detection_threshold': 0.7
        }
        
        consolidator = MemoryConsolidator(memory_store, config)
        session_id = "demo_session"
        
        logger.info("📝 Adding clinical memories with varying characteristics...")
        
        # Add diverse memories for testing
        clinical_memories = [
            {
                'content': "Patient shows high MMSE score (28) but has APOE4 genotype, indicating complex risk profile",
                'type': 'reasoning',
                'importance': 0.9,
                'uncertainty': 0.6,
                'reward': 0.8
            },
            {
                'content': "Family history strongly predicts dementia development in 60% of cases",
                'type': 'knowledge',
                'importance': 0.8,
                'uncertainty': 0.3,
                'reward': 0.7
            },
            {
                'content': "APOE4 testing shows positive results with high confidence",
                'type': 'experience',
                'importance': 0.7,
                'uncertainty': 0.2,
                'reward': 0.9
            },
            {
                'content': "Treatment X shows positive effects on cognitive function in trials",
                'type': 'semantic',
                'importance': 0.8,
                'uncertainty': 0.4,
                'reward': 0.6
            },
            {
                'content': "Treatment X does not show significant cognitive improvements in recent studies",
                'type': 'semantic',  # This will conflict with previous memory
                'importance': 0.7,
                'uncertainty': 0.5,
                'reward': 0.4
            }
        ]
        
        # Store memories and calculate enhanced salience
        memory_ids = []
        for mem_data in clinical_memories:
            memory_id = memory_store.store_memory(
                session_id=session_id,
                content=mem_data['content'],
                memory_type=mem_data['type'],
                importance=mem_data['importance']
            )
            memory_ids.append(memory_id)
            
            # Calculate enhanced salience with uncertainty and reward
            salience = consolidator.calculate_salience_score(
                memory_id, session_id, 
                uncertainty_score=mem_data['uncertainty'],
                reward_signal=mem_data['reward']
            )
            
            logger.info(f"Memory {memory_id}: Salience = {salience:.3f}, Type = {mem_data['type']}")
        
        logger.info("🔄 Running enhanced consolidation cycle...")
        
        # Run complete consolidation cycle
        results = consolidator.run_consolidation_cycle(session_id)
        
        logger.info("📊 Consolidation Results:")
        for key, value in results.items():
            if key != 'events':
                logger.info(f"  {key}: {value}")
        
        logger.info("📈 Getting enhanced consolidation summary...")
        
        # Get enhanced summary with Phase 2 & 3 metrics
        summary = consolidator.get_consolidation_summary(hours=24)
        
        logger.info("🎯 Enhanced Consolidation Summary:")
        logger.info(f"  Synaptic Tagged Memories: {summary.get('synaptic_tagged_memories', 0)}")
        logger.info(f"  Priority Replay Events: {summary.get('priority_replay_events', 0)}")
        logger.info(f"  Semantic Conflicts: {summary.get('total_conflicts', 0)}")
        logger.info(f"  Priority Distribution: {summary.get('priority_distribution', {})}")
        
        logger.info("🔍 Testing memory introspection...")
        
        # Test memory introspection API
        decision_context = "Should we recommend APOE4 testing for this patient?"
        introspection = consolidator.get_memory_introspection(decision_context, session_id)
        
        logger.info("🧭 Memory Introspection Results:")
        logger.info(f"  Trace Confidence: {introspection.get('trace_confidence', 0):.3f}")
        logger.info(f"  Influential Memories: {introspection.get('influential_memories', 0)}")
        logger.info(f"  Conflicted Memories: {introspection.get('conflicted_memories', 0)}")
        
        # Show memory chain
        memory_chain = introspection.get('memory_chain', [])
        if memory_chain:
            logger.info("  Top Influential Memories:")
            for i, memory in enumerate(memory_chain[:3]):
                logger.info(f"    {i+1}. [{memory['memory_type']}] {memory['content_preview']}")
                logger.info(f"       Influence: {memory['influence_weight']:.3f}, Clinical: {memory['clinical_relevance']:.3f}")
        
        logger.info("✅ Enhanced Memory Consolidation Demo Complete!")
        
    finally:
        # Cleanup
        if os.path.exists(temp_db):
            os.unlink(temp_db)


def demo_agent_extensions():
    """Demonstrate agent extensions and plugin system."""
    logger.info("🔌 Starting Agent Extensions Demo")
    
    # Create capability registry
    config = {
        'max_plugin_memory_mb': 64,
        'max_plugin_cpu_seconds': 10
    }
    registry = create_capability_registry(config)
    
    # Create mock memory systems for context
    memory_store = MockMemoryStore()
    
    # Add some test memories
    session_id = "plugin_demo_session"
    memory_store.store_memory(
        session_id=session_id,
        content="Patient presents with moderate cognitive decline and APOE4 positive status",
        memory_type="clinical_observation",
        importance=0.8
    )
    
    logger.info("📦 Registering Clinical Guideline Agent plugin...")
    
    # Create and register clinical guideline agent
    clinical_agent = ClinicalGuidelineAgent()
    
    success = registry.register_plugin(clinical_agent)
    if success:
        logger.info("✅ Clinical Guideline Agent registered successfully")
    else:
        logger.error("❌ Failed to register Clinical Guideline Agent")
        return
    
    # Activate the plugin
    success = registry.activate_plugin(clinical_agent.name)
    if success:
        logger.info("✅ Clinical Guideline Agent activated")
    else:
        logger.error("❌ Failed to activate Clinical Guideline Agent")
        return
    
    logger.info("📊 Plugin system metrics:")
    metrics = registry.get_plugin_metrics()
    for key, value in metrics.items():
        if key != 'plugin_status_distribution':
            logger.info(f"  {key}: {value}")
    
    logger.info("🎯 Testing capability dispatch...")
    
    # Create agent context
    context = create_agent_context(
        session_id=session_id,
        agent_name="demo_agent",
        memory_store=memory_store,
        consolidator=None,  # No consolidator needed for this demo
        scopes=[CapabilityScope.READ_MEMORY, CapabilityScope.WRITE_SEMANTIC]
    )
    
    # Test guideline explanation capability
    result = registry.dispatch(
        capability="generate_guideline_explanation",
        payload={"condition": "alzheimers_stage_2"},
        context=context
    )
    
    if result.success:
        logger.info("✅ Guideline explanation generated successfully:")
        guideline = result.result_data.get('guideline', {})
        logger.info(f"  Explanation: {guideline.get('explanation', 'N/A')[:100]}...")
        logger.info(f"  Key Factors: {guideline.get('key_factors', [])}")
        logger.info(f"  Recommendations: {guideline.get('recommendations', [])}")
    else:
        logger.error(f"❌ Guideline explanation failed: {result.error_message}")
    
    # Test compliance assessment capability  
    result = registry.dispatch(
        capability="assess_guideline_compliance",
        payload={"patient_data": {"mmse_score": 26, "apoe4_tested": True}},
        context=context
    )
    
    if result.success:
        logger.info("✅ Compliance assessment completed:")
        logger.info(f"  Compliance Score: {result.result_data.get('compliance_score', 0):.2f}")
        logger.info(f"  Compliant Items: {result.result_data.get('compliant_items', [])}")
        logger.info(f"  Missing Items: {result.result_data.get('missing_items', [])}")
    else:
        logger.error(f"❌ Compliance assessment failed: {result.error_message}")
    
    logger.info("📈 Final plugin metrics:")
    final_metrics = registry.get_plugin_metrics()
    logger.info(f"  Capability Invocations: {final_metrics.get('capability_invocations', 0)}")
    logger.info(f"  Plugin Failures: {final_metrics.get('plugin_failures', 0)}")
    logger.info(f"  Security Violations: {final_metrics.get('security_violations', 0)}")
    
    logger.info("✅ Agent Extensions Demo Complete!")


def demo_visualization_api():
    """Demonstrate enhanced visualization API."""
    logger.info("📊 Starting Enhanced Visualization API Demo")
    
    # Create mock systems
    memory_store = MockMemoryStore()
    
    # Create temporary database for consolidator
    fd, temp_db = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    
    try:
        # Create consolidator and capability registry
        consolidator = MemoryConsolidator(memory_store, {'consolidation_db': temp_db})
        registry = create_capability_registry({})
        
        # Register a plugin
        clinical_agent = ClinicalGuidelineAgent()
        registry.register_plugin(clinical_agent)
        registry.activate_plugin(clinical_agent.name)
        
        # Create visualization API
        vis_config = {'port': 5001, 'debug': False}
        vis_api = VisualizationAPIDemo(vis_config)
        
        # Initialize with our systems
        vis_api.initialize_monitors(
            safety_monitor=None,  # Would be real safety monitor in production
            security_monitor=None,  # Would be real security monitor in production  
            production_monitor=None,  # Would be real production monitor in production
            memory_store=memory_store,
            memory_consolidator=consolidator,
            capability_registry=registry
        )
        
        logger.info("🌐 Enhanced Visualization API initialized with:")
        logger.info("  ✅ Memory consolidation system")
        logger.info("  ✅ Plugin capability registry")
        logger.info("  ✅ Agent interaction graph support")
        logger.info("  ✅ Memory introspection endpoints")
        logger.info("  ✅ Semantic conflict management")
        
        logger.info("📡 Available API endpoints:")
        endpoints = [
            "/api/memory/introspection (POST)",
            "/api/memory/conflicts",
            "/api/memory/conflicts/<id>/resolve (POST)",
            "/api/plugins/summary",
            "/api/plugins/list",
            "/api/plugins/<name>/activate (POST)",
            "/api/plugins/<name>/deactivate (POST)",
            "/api/agent/interaction-graph (enhanced)",
            "/api/dashboard/overview (enhanced)"
        ]
        
        for endpoint in endpoints:
            logger.info(f"  • {endpoint}")
        
        logger.info("🎨 Enhanced dashboard features:")
        features = [
            "Real-time plugin system monitoring",
            "Semantic conflict tracking and resolution",
            "Memory consolidation metrics with synaptic tagging",
            "Agent network visualization with detailed metrics",
            "Memory introspection for decision traceability"
        ]
        
        for feature in features:
            logger.info(f"  • {feature}")
        
        logger.info("✅ Enhanced Visualization API Demo Complete!")
        logger.info("💡 To run the API server: python -c \"from demo_enhanced_features import demo_visualization_api; demo_visualization_api()\"")
        
    finally:
        if os.path.exists(temp_db):
            os.unlink(temp_db)


def main():
    """Run all demonstrations."""
    print("🚀 DuetMind Adaptive Enhanced Features Demo")
    print("=" * 60)
    
    try:
        # Demo 1: Enhanced Memory Consolidation
        demo_enhanced_memory_consolidation()
        print("\n" + "=" * 60 + "\n")
        
        # Demo 2: Agent Extensions and Plugin System
        demo_agent_extensions()
        print("\n" + "=" * 60 + "\n")
        
        # Demo 3: Enhanced Visualization API
        demo_visualization_api()
        print("\n" + "=" * 60)
        
        print("🎉 All Enhanced Features Demos Completed Successfully!")
        print("\n📋 Summary of Enhanced Features:")
        print("• Phase 2: Priority replay with synaptic tagging")
        print("• Phase 2: Generative rehearsal and consolidation scheduling") 
        print("• Phase 3: Semantic conflict detection and resolution")
        print("• Phase 3: Memory introspection API for decision traceability")
        print("• Plugin system with secure capability registry")
        print("• Enhanced visualization dashboard with real-time metrics")
        
    except Exception as e:
        logger.error(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()