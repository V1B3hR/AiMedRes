# Enhanced Memory Consolidation and Agent Extensions

This document describes the implementation of enhanced memory consolidation algorithms, visualization tools, and API for custom agent behaviors as specified in the problem statement.

## 🧠 Phase 2: Enhanced Memory Consolidation Algorithms

### Implemented Features

#### 1. Enhanced Salience Scoring
- **Expanded Formula**: `S = αnovelty + βfrequency + γclinical_importance + δreward_delta + εuncertainty`
- **Synaptic Tagging**: High-reward memories (>0.5) with salience >0.8 receive synaptic tags for priority consolidation
- **New Metadata Fields**: Added uncertainty_score, reward_signal, synaptic_tag, conflict_flag, source_references

#### 2. Priority Replay System
- **Weighted Sampling**: Priority = novelty × uncertainty × reward (with synaptic boost)
- **Sample Size**: Configurable replay sample size (default 50 memories)
- **Replay Events**: Tracked in database with priority scores and metadata
- **Biological Plausibility**: Implements priority-based memory replay during consolidation

#### 3. Generative Rehearsal
- **Periodic Summarization**: Runs every 24 hours (configurable)
- **High-Salience Focus**: Summarizes memories with salience >0.6
- **Structured Summaries**: Groups memories by type and extracts key clinical concepts
- **Semantic Storage**: Rehearsal summaries stored as new semantic memories

#### 4. Advanced Consolidation Scheduling
- **Idle Time Detection**: Configurable consolidation intervals
- **Episode-Triggered**: Consolidation triggered by memory count thresholds
- **Circadian Simulation**: Time-based scheduling with rehearsal intervals

## 🔍 Phase 3: Semantic Conflict Resolution and Memory Introspection

### Implemented Features

#### 1. Semantic Conflict Detection
- **Pattern Matching**: Detects contradictions using keyword analysis
- **Confidence Scoring**: Assigns confidence scores to detected conflicts
- **Types Supported**: 'contradiction', 'inconsistency', 'outdated'
- **Database Tracking**: Conflicts stored with resolution status and metadata

#### 2. Conflict Resolution System
- **Manual Resolution**: Human-in-the-loop conflict resolution
- **Winning Memory Selection**: Optionally designate winning memory
- **Influence Reduction**: Losing memories have reduced consolidation priority
- **Status Tracking**: Conflicts marked as 'pending', 'resolved', or 'escalated'

#### 3. Memory Introspection API
- **Decision Traceability**: `get_memory_introspection(decision_context, session_id)`
- **Influence Weighting**: Memories weighted by priority × clinical_relevance × synaptic_tag
- **Conflict Awareness**: Conflicted memories have reduced influence (×0.7)
- **Confidence Scoring**: Overall trace confidence based on total influence

#### 4. Enhanced Metrics
- **Consolidation Summary**: Now includes synaptic tagged counts, replay events, conflicts
- **Priority Distribution**: High/medium/low priority memory counts
- **Conflict Statistics**: Pending vs resolved conflict tracking

## 📊 Phase 4: Enhanced Visualization Tools

### New API Endpoints

#### Memory Introspection
- `POST /api/memory/introspection` - Get decision trace and memory chain
- `GET /api/memory/conflicts` - List semantic conflicts by status
- `POST /api/memory/conflicts/<id>/resolve` - Resolve specific conflicts

#### Plugin System Monitoring
- `GET /api/plugins/summary` - Plugin system metrics
- `GET /api/plugins/list` - List all registered plugins
- `POST /api/plugins/<name>/activate` - Activate plugin
- `POST /api/plugins/<name>/deactivate` - Deactivate plugin

#### Enhanced Agent Graph
- `GET /api/agent/interaction-graph` - Enhanced with plugin agents and detailed metrics
- Real-time metrics: checks_per_minute, accuracy_last_hour, consolidations_last_hour

### Dashboard Enhancements
- **Plugin System Card**: Shows active/total plugins and capability counts
- **Agent Network Card**: Shows active agents and connection latency
- **Semantic Conflicts Section**: Lists pending conflicts with confidence scores
- **Enhanced Memory Card**: Shows consolidation events and synaptic tagged memories

## 🔌 Phase 5: API for Custom Agent Behaviors and Extensions

### Core Components

#### 1. Capability Registry
- **Plugin Management**: Registration, activation, deactivation lifecycle
- **Capability Mapping**: Maps capability names to plugin handlers
- **Metrics Tracking**: Invocations, failures, security violations
- **Lifecycle Hooks**: on_register, on_pre_message, on_post_message, on_error

#### 2. Agent Plugin Interface
```python
class AgentPlugin(Protocol):
    name: str
    version: str
    manifest: PluginManifest
    
    def capabilities(self) -> List[str]
    def handle(message: AgentMessage, context: AgentContext) -> AgentResult
    def healthcheck(self) -> Dict[str, Any]
    def on_register(registry: CapabilityRegistry) -> bool
```

#### 3. Security and Sandboxing
- **Capability Scopes**: READ_MEMORY, WRITE_MEMORY, WRITE_SEMANTIC, etc.
- **Policy Engine**: Enforces scope-based permissions and safety state checks
- **Sandbox Execution**: Resource limits (memory, CPU time, file handles)
- **Memory Accessor**: Safe, scoped access to memory systems

#### 4. Example Plugin: ClinicalGuidelineAgent
- **Capabilities**: generate_guideline_explanation, assess_guideline_compliance
- **Scopes Required**: READ_MEMORY, WRITE_SEMANTIC
- **Features**: Provides clinical guideline analysis and compliance assessment

### Plugin Manifest Example
```python
PluginManifest(
    name="clinical_guideline_agent",
    version="1.0.0",
    description="Provides clinical guideline analysis",
    author="DuetMind Core Team",
    required_scopes=[CapabilityScope.READ_MEMORY, CapabilityScope.WRITE_SEMANTIC],
    requires_api=">=1.0.0,<2.0.0",
    capabilities=["generate_guideline_explanation", "assess_guideline_compliance"],
    sandbox_required=True
)
```

## 🧪 Testing and Validation

### Test Coverage
1. **Enhanced Memory Consolidation Tests** (`test_enhanced_memory_consolidation.py`)
   - Salience scoring with uncertainty and reward
   - Synaptic tagging behavior
   - Priority replay execution
   - Generative rehearsal timing and content
   - Semantic conflict detection and resolution
   - Memory introspection API functionality

2. **Agent Extensions Tests** (`test_agent_extensions.py`)
   - Plugin registration and lifecycle management
   - Capability dispatch and security
   - Memory accessor permissions
   - Policy engine enforcement
   - Sandbox execution and timeouts

### Demo Script
Run `python demo_enhanced_features.py` to see all features in action:
- Enhanced memory consolidation with conflict detection
- Plugin system registration and capability dispatch
- Visualization API with monitoring capabilities

## 📈 Metrics and Monitoring

### Enhanced Consolidation Metrics
- **Synaptic Tagged Memories**: Count of high-priority memories
- **Priority Replay Events**: Number of replay cycles executed
- **Semantic Conflicts**: Total, pending, and resolved conflict counts
- **Priority Distribution**: Memory distribution across priority levels

### Plugin System Metrics
- **Plugin Registrations**: Total plugins registered
- **Capability Invocations**: Total capability calls made
- **Plugin Failures**: Failed plugin executions
- **Security Violations**: Policy enforcement violations
- **Active Plugins**: Currently active plugin count

## 🚀 Usage Examples

### Memory Consolidation with Enhanced Features
```python
from agent_memory.memory_consolidation import MemoryConsolidator

consolidator = MemoryConsolidator(memory_store, config)

# Enhanced salience scoring
salience = consolidator.calculate_salience_score(
    memory_id, session_id, 
    uncertainty_score=0.6, 
    reward_signal=0.8
)

# Run enhanced consolidation
results = consolidator.run_consolidation_cycle(session_id)

# Get memory introspection
introspection = consolidator.get_memory_introspection(
    "Should we test for APOE4?", session_id
)
```

### Plugin System Usage
```python
from agent_memory.agent_extensions import (
    create_capability_registry, ClinicalGuidelineAgent
)

# Create registry and register plugin
registry = create_capability_registry(config)
plugin = ClinicalGuidelineAgent()
registry.register_plugin(plugin)
registry.activate_plugin(plugin.name)

# Dispatch capability
result = registry.dispatch(
    "generate_guideline_explanation",
    {"condition": "alzheimers_stage_2"},
    context
)
```

### Enhanced Visualization API
```python
from api.visualization_api import VisualizationAPI

api = VisualizationAPI(config)
api.initialize_monitors(
    memory_consolidator=consolidator,
    capability_registry=registry
)

# Access enhanced endpoints:
# POST /api/memory/introspection
# GET /api/memory/conflicts
# GET /api/plugins/summary
```

## 🏗️ Architecture Improvements

### Database Schema Enhancements
- **memory_metadata**: Added uncertainty_score, reward_signal, synaptic_tag, conflict_flag
- **semantic_conflicts**: New table for conflict tracking and resolution
- **replay_events**: New table for priority replay event logging
- **Enhanced Indexes**: Performance optimization for new query patterns

### Biological Plausibility
- **Synaptic Tagging**: Models biological synaptic tagging for memory consolidation
- **Priority Replay**: Implements hippocampal replay mechanisms
- **Consolidation Scheduling**: Models sleep-like consolidation phases

### Security Architecture
- **Scope-Based Permissions**: Fine-grained capability access control
- **Policy Enforcement**: Global safety state awareness
- **Sandbox Execution**: Resource-limited plugin execution environment
- **Memory Access Control**: Safe, filtered memory access for plugins

## 📋 Future Enhancements

### Planned Improvements
1. **Advanced NLP**: Replace simple pattern matching with transformer-based conflict detection
2. **Semantic Versioning**: Full semantic versioning support for plugin compatibility
3. **Distributed Execution**: Plugin execution across multiple nodes
4. **Advanced Sandboxing**: Container-based plugin isolation
5. **Real-time Dashboard**: WebSocket-based real-time monitoring
6. **Plugin Marketplace**: Registry for discovering and installing plugins

### Scalability Considerations
- **Database Optimization**: Indexes and query optimization for large memory stores
- **Batch Processing**: Bulk operations for large-scale consolidation
- **Caching**: Memory and conflict caching for improved performance
- **Load Balancing**: Plugin execution load distribution

## 🎯 Summary

This implementation provides a comprehensive enhancement to the DuetMind Adaptive memory consolidation system with:

✅ **Phase 2 Complete**: Priority replay, synaptic tagging, generative rehearsal
✅ **Phase 3 Complete**: Semantic conflict resolution, memory introspection API  
✅ **Enhanced Visualization**: Real-time monitoring with conflict and plugin management
✅ **Plugin Architecture**: Secure, extensible agent behavior system
✅ **Comprehensive Testing**: Full test coverage for all new features
✅ **Working Demo**: Complete demonstration of all enhanced capabilities

The system now provides biologically-inspired memory consolidation with advanced conflict resolution, comprehensive visualization tools, and a secure plugin architecture for extending agent behaviors.