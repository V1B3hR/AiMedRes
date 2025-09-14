"""
Tests for Agent Extensions and Plugin System

Tests the capability registry, plugin management, security, and sandbox execution.
"""

import pytest
import tempfile
import json
from datetime import datetime
from unittest.mock import Mock, patch

from agent_memory.agent_extensions import (
    CapabilityRegistry, AgentPlugin, PluginManifest, CapabilityScope,
    PluginStatus, AgentMessage, AgentContext, AgentResult,
    MemoryAccessor, PolicyEngine, SandboxExecutor,
    ClinicalGuidelineAgent, create_capability_registry, create_agent_context
)


class TestPlugin(AgentPlugin):
    """Test plugin for testing purposes."""
    
    def __init__(self, name: str = "test_plugin", fail_healthcheck: bool = False):
        self._name = name
        self._version = "1.0.0"
        self._fail_healthcheck = fail_healthcheck
        self._active = False
        self._manifest = PluginManifest(
            name=self._name,
            version=self._version,
            description="Test plugin for unit testing",
            author="Test Author",
            required_scopes=[CapabilityScope.READ_MEMORY],
            requires_api=">=1.0.0",
            capabilities=["test_capability", "another_capability"],
            sandbox_required=False
        )
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def version(self) -> str:
        return self._version
    
    @property
    def manifest(self) -> PluginManifest:
        return self._manifest
    
    def capabilities(self) -> list:
        return self._manifest.capabilities
    
    def handle(self, message: AgentMessage, context: AgentContext) -> AgentResult:
        if message.message_type == "test_capability":
            return AgentResult(
                success=True,
                result_data={"test_result": "success", "message_id": message.message_id}
            )
        elif message.message_type == "another_capability":
            return AgentResult(
                success=True,
                result_data={"capability": "another_capability"}
            )
        else:
            return AgentResult(
                success=False,
                result_data={},
                error_message=f"Unknown capability: {message.message_type}"
            )
    
    def healthcheck(self) -> dict:
        return {
            'healthy': not self._fail_healthcheck,
            'status': 'active' if self._active else 'inactive',
            'timestamp': datetime.now().isoformat()
        }
    
    def on_register(self, registry) -> bool:
        return True
    
    def on_activate(self) -> bool:
        self._active = True
        return True
    
    def on_deactivate(self) -> bool:
        self._active = False
        return True


@pytest.fixture
def mock_memory_store():
    """Mock memory store for testing."""
    store = Mock()
    store.retrieve_memories.return_value = [
        {
            'id': 1,
            'content': 'Test memory content',
            'memory_type': 'reasoning',
            'importance_score': 0.7,
            'created_at': datetime.now()
        }
    ]
    store.store_memory.return_value = 123
    return store


@pytest.fixture
def mock_consolidator():
    """Mock consolidator for testing."""
    consolidator = Mock()
    consolidator.get_consolidation_summary.return_value = {
        'total_consolidation_events': 5,
        'synaptic_tagged_memories': 3
    }
    return consolidator


@pytest.fixture
def capability_registry():
    """Create capability registry for testing."""
    config = {
        'max_plugin_memory_mb': 64,
        'max_plugin_cpu_seconds': 10
    }
    return create_capability_registry(config)


class TestCapabilityRegistry:
    """Test capability registry functionality."""
    
    def test_plugin_registration(self, capability_registry):
        """Test successful plugin registration."""
        plugin = TestPlugin("test_plugin_1")
        
        success = capability_registry.register_plugin(plugin)
        assert success
        
        assert "test_plugin_1" in capability_registry.plugins
        assert capability_registry.plugin_status["test_plugin_1"] == PluginStatus.REGISTERED
        
        # Check capabilities are mapped
        assert "test_capability" in capability_registry.capability_mapping
        assert capability_registry.capability_mapping["test_capability"] == "test_plugin_1"
    
    def test_duplicate_plugin_registration(self, capability_registry):
        """Test that duplicate plugin registration fails."""
        plugin1 = TestPlugin("duplicate_plugin")
        plugin2 = TestPlugin("duplicate_plugin")
        
        success1 = capability_registry.register_plugin(plugin1)
        success2 = capability_registry.register_plugin(plugin2)
        
        assert success1
        assert not success2  # Second registration should fail
    
    def test_conflicting_capability_registration(self, capability_registry):
        """Test that conflicting capabilities fail registration."""
        plugin1 = TestPlugin("plugin1")
        plugin2 = TestPlugin("plugin2")
        # Both have "test_capability" - should conflict
        
        success1 = capability_registry.register_plugin(plugin1)
        success2 = capability_registry.register_plugin(plugin2)
        
        assert success1
        assert not success2  # Second registration should fail due to capability conflict
    
    def test_plugin_activation(self, capability_registry):
        """Test plugin activation."""
        plugin = TestPlugin("activation_test")
        
        capability_registry.register_plugin(plugin)
        success = capability_registry.activate_plugin("activation_test")
        
        assert success
        assert capability_registry.plugin_status["activation_test"] == PluginStatus.ACTIVE
        assert plugin._active  # Plugin should know it's active
    
    def test_plugin_activation_failure(self, capability_registry):
        """Test plugin activation failure due to health check."""
        plugin = TestPlugin("failing_plugin", fail_healthcheck=True)
        
        capability_registry.register_plugin(plugin)
        success = capability_registry.activate_plugin("failing_plugin")
        
        assert not success
        assert capability_registry.plugin_status["failing_plugin"] == PluginStatus.FAILED
    
    def test_plugin_deactivation(self, capability_registry):
        """Test plugin deactivation."""
        plugin = TestPlugin("deactivation_test")
        
        capability_registry.register_plugin(plugin)
        capability_registry.activate_plugin("deactivation_test")
        
        success = capability_registry.deactivate_plugin("deactivation_test")
        
        assert success
        assert capability_registry.plugin_status["deactivation_test"] == PluginStatus.INACTIVE
        assert not plugin._active


class TestCapabilityDispatch:
    """Test capability dispatch functionality."""
    
    def test_successful_dispatch(self, capability_registry, mock_memory_store, mock_consolidator):
        """Test successful capability dispatch."""
        plugin = TestPlugin("dispatch_test")
        capability_registry.register_plugin(plugin)
        capability_registry.activate_plugin("dispatch_test")
        
        # Create context
        context = create_agent_context(
            session_id="test_session",
            agent_name="test_agent",
            memory_store=mock_memory_store,
            consolidator=mock_consolidator,
            scopes=[CapabilityScope.READ_MEMORY]
        )
        
        # Dispatch capability
        result = capability_registry.dispatch(
            capability="test_capability",
            payload={"test_data": "value"},
            context=context
        )
        
        assert result.success
        assert "test_result" in result.result_data
        assert result.result_data["test_result"] == "success"
    
    def test_dispatch_unknown_capability(self, capability_registry, mock_memory_store, mock_consolidator):
        """Test dispatch of unknown capability."""
        context = create_agent_context(
            session_id="test_session",
            agent_name="test_agent", 
            memory_store=mock_memory_store,
            consolidator=mock_consolidator,
            scopes=[CapabilityScope.READ_MEMORY]
        )
        
        result = capability_registry.dispatch(
            capability="unknown_capability",
            payload={},
            context=context
        )
        
        assert not result.success
        assert "not registered" in result.error_message
    
    def test_dispatch_inactive_plugin(self, capability_registry, mock_memory_store, mock_consolidator):
        """Test dispatch to inactive plugin."""
        plugin = TestPlugin("inactive_test")
        capability_registry.register_plugin(plugin)
        # Don't activate plugin
        
        context = create_agent_context(
            session_id="test_session",
            agent_name="test_agent",
            memory_store=mock_memory_store,
            consolidator=mock_consolidator,
            scopes=[CapabilityScope.READ_MEMORY]
        )
        
        result = capability_registry.dispatch(
            capability="test_capability",
            payload={},
            context=context
        )
        
        assert not result.success
        assert "not active" in result.error_message
    
    def test_metrics_tracking(self, capability_registry, mock_memory_store, mock_consolidator):
        """Test that metrics are tracked correctly."""
        plugin = TestPlugin("metrics_test")
        capability_registry.register_plugin(plugin)
        capability_registry.activate_plugin("metrics_test")
        
        context = create_agent_context(
            session_id="test_session",
            agent_name="test_agent",
            memory_store=mock_memory_store,
            consolidator=mock_consolidator,
            scopes=[CapabilityScope.READ_MEMORY]
        )
        
        initial_metrics = capability_registry.get_plugin_metrics()
        
        # Successful dispatch
        capability_registry.dispatch("test_capability", {}, context)
        
        # Failed dispatch
        capability_registry.dispatch("unknown_capability", {}, context)
        
        final_metrics = capability_registry.get_plugin_metrics()
        
        assert final_metrics['capability_invocations'] > initial_metrics['capability_invocations']
        assert final_metrics['plugin_failures'] > initial_metrics['plugin_failures']


class TestMemoryAccessor:
    """Test memory accessor security."""
    
    def test_read_memory_with_permission(self, mock_memory_store, mock_consolidator):
        """Test reading memory with proper permissions."""
        accessor = MemoryAccessor(
            mock_memory_store, mock_consolidator, 
            [CapabilityScope.READ_MEMORY]
        )
        
        memories = accessor.read_memories("test_session", "test query")
        
        assert len(memories) > 0
        mock_memory_store.retrieve_memories.assert_called_once()
    
    def test_read_memory_without_permission(self, mock_memory_store, mock_consolidator):
        """Test reading memory without proper permissions."""
        accessor = MemoryAccessor(
            mock_memory_store, mock_consolidator,
            []  # No permissions
        )
        
        with pytest.raises(PermissionError):
            accessor.read_memories("test_session", "test query")
    
    def test_store_memory_with_permission(self, mock_memory_store, mock_consolidator):
        """Test storing memory with proper permissions."""
        accessor = MemoryAccessor(
            mock_memory_store, mock_consolidator,
            [CapabilityScope.WRITE_MEMORY]
        )
        
        memory_id = accessor.store_memory("test_session", "test content")
        
        assert memory_id == 123
        mock_memory_store.store_memory.assert_called_once()
    
    def test_store_memory_without_permission(self, mock_memory_store, mock_consolidator):
        """Test storing memory without proper permissions."""
        accessor = MemoryAccessor(
            mock_memory_store, mock_consolidator,
            []  # No permissions
        )
        
        with pytest.raises(PermissionError):
            accessor.store_memory("test_session", "test content")
    
    def test_importance_limiting(self, mock_memory_store, mock_consolidator):
        """Test that plugin memories have limited importance."""
        accessor = MemoryAccessor(
            mock_memory_store, mock_consolidator,
            [CapabilityScope.WRITE_MEMORY]
        )
        
        # Try to store high importance memory
        accessor.store_memory("test_session", "test content", importance=0.95)
        
        # Should be limited to 0.8
        call_args = mock_memory_store.store_memory.call_args
        assert call_args[1]['importance'] <= 0.8


class TestPolicyEngine:
    """Test policy engine functionality."""
    
    def test_capability_allowed_safe_state(self):
        """Test capability allowed in safe state."""
        policy = PolicyEngine({})
        
        allowed = policy.check_capability_allowed(
            "memory_query", 
            [CapabilityScope.READ_MEMORY]
        )
        
        assert allowed
    
    def test_capability_blocked_critical_state(self):
        """Test capability blocked in critical state."""
        policy = PolicyEngine({})
        policy.update_safety_state("CRITICAL")
        
        allowed = policy.check_capability_allowed(
            "memory_query",
            [CapabilityScope.READ_MEMORY]
        )
        
        assert not allowed
    
    def test_capability_blocked_insufficient_scope(self):
        """Test capability blocked with insufficient scope.""" 
        policy = PolicyEngine({})
        
        allowed = policy.check_capability_allowed(
            "memory_query",
            []  # No scopes
        )
        
        assert not allowed
    
    def test_globally_blocked_capability(self):
        """Test globally blocked capability."""
        policy = PolicyEngine({})
        policy.blocked_capabilities.add("blocked_capability")
        
        allowed = policy.check_capability_allowed(
            "blocked_capability",
            [CapabilityScope.READ_MEMORY]
        )
        
        assert not allowed


class TestSandboxExecutor:
    """Test sandbox execution."""
    
    def test_successful_execution(self):
        """Test successful method execution."""
        executor = SandboxExecutor({'max_plugin_cpu_seconds': 5})
        plugin = TestPlugin()
        
        message = AgentMessage(
            message_id="test",
            sender_id="test",
            recipient_id="test",
            message_type="test_capability",
            payload={},
            timestamp=datetime.now()
        )
        
        context = Mock()
        
        result = executor.execute_plugin_method(plugin, 'handle', message, context)
        
        assert result.success
    
    def test_timeout_handling(self):
        """Test timeout handling in sandbox."""
        executor = SandboxExecutor({'max_plugin_cpu_seconds': 1})
        
        # Create a plugin method that takes too long
        class SlowPlugin:
            def slow_method(self):
                import time
                time.sleep(2)  # Longer than timeout
                return "done"
        
        plugin = SlowPlugin()
        
        with pytest.raises(TimeoutError):
            executor.execute_plugin_method(plugin, 'slow_method')


class TestClinicalGuidelineAgent:
    """Test the example clinical guideline agent."""
    
    def test_guideline_explanation(self, mock_memory_store, mock_consolidator):
        """Test guideline explanation capability."""
        agent = ClinicalGuidelineAgent()
        
        message = AgentMessage(
            message_id="test",
            sender_id="test",
            recipient_id="clinical_guideline_agent",
            message_type="generate_guideline_explanation",
            payload={"condition": "alzheimers_stage_2"},
            timestamp=datetime.now()
        )
        
        context = create_agent_context(
            session_id="test_session",
            agent_name="test_agent",
            memory_store=mock_memory_store,
            consolidator=mock_consolidator,
            scopes=[CapabilityScope.READ_MEMORY, CapabilityScope.WRITE_SEMANTIC]
        )
        
        result = agent.handle(message, context)
        
        assert result.success
        assert "guideline" in result.result_data
        assert "alzheimers_stage_2" in result.result_data["condition"]
    
    def test_compliance_assessment(self, mock_memory_store, mock_consolidator):
        """Test compliance assessment capability."""
        agent = ClinicalGuidelineAgent()
        
        message = AgentMessage(
            message_id="test",
            sender_id="test",
            recipient_id="clinical_guideline_agent",
            message_type="assess_guideline_compliance",
            payload={"patient_data": {}},
            timestamp=datetime.now()
        )
        
        context = create_agent_context(
            session_id="test_session",
            agent_name="test_agent",
            memory_store=mock_memory_store,
            consolidator=mock_consolidator,
            scopes=[CapabilityScope.READ_MEMORY, CapabilityScope.WRITE_SEMANTIC]
        )
        
        result = agent.handle(message, context)
        
        assert result.success
        assert "compliance_score" in result.result_data
        assert isinstance(result.result_data["compliance_score"], float)
    
    def test_unknown_capability(self, mock_memory_store, mock_consolidator):
        """Test handling of unknown capability."""
        agent = ClinicalGuidelineAgent()
        
        message = AgentMessage(
            message_id="test",
            sender_id="test",
            recipient_id="clinical_guideline_agent",
            message_type="unknown_capability",
            payload={},
            timestamp=datetime.now()
        )
        
        context = create_agent_context(
            session_id="test_session",
            agent_name="test_agent",
            memory_store=mock_memory_store,
            consolidator=mock_consolidator,
            scopes=[CapabilityScope.READ_MEMORY]
        )
        
        result = agent.handle(message, context)
        
        assert not result.success
        assert "Unknown capability" in result.error_message


class TestLifecycleHooks:
    """Test lifecycle hooks functionality."""
    
    def test_registration_hook(self, capability_registry):
        """Test registration lifecycle hook."""
        hook_called = False
        
        def registration_hook(plugin):
            nonlocal hook_called
            hook_called = True
            assert plugin.name == "hook_test"
        
        capability_registry.add_lifecycle_hook('on_register', registration_hook)
        
        plugin = TestPlugin("hook_test")
        capability_registry.register_plugin(plugin)
        
        assert hook_called
    
    def test_pre_message_hook(self, capability_registry, mock_memory_store, mock_consolidator):
        """Test pre-message lifecycle hook."""
        hook_called = False
        
        def pre_message_hook(message, context):
            nonlocal hook_called
            hook_called = True
            assert message.message_type == "test_capability"
        
        capability_registry.add_lifecycle_hook('on_pre_message', pre_message_hook)
        
        plugin = TestPlugin("hook_test")
        capability_registry.register_plugin(plugin)
        capability_registry.activate_plugin("hook_test")
        
        context = create_agent_context(
            session_id="test_session",
            agent_name="test_agent",
            memory_store=mock_memory_store,
            consolidator=mock_consolidator,
            scopes=[CapabilityScope.READ_MEMORY]
        )
        
        capability_registry.dispatch("test_capability", {}, context)
        
        assert hook_called


@pytest.mark.integration  
class TestFullPluginSystem:
    """Integration tests for the complete plugin system."""
    
    def test_complete_plugin_lifecycle(self, mock_memory_store, mock_consolidator):
        """Test complete plugin lifecycle from registration to execution."""
        # Create registry
        registry = create_capability_registry({})
        
        # Create and register plugin
        plugin = TestPlugin("integration_test")
        success = registry.register_plugin(plugin)
        assert success
        
        # Activate plugin
        success = registry.activate_plugin("integration_test")
        assert success
        
        # Create context
        context = create_agent_context(
            session_id="integration_session",
            agent_name="integration_agent",
            memory_store=mock_memory_store,
            consolidator=mock_consolidator,
            scopes=[CapabilityScope.READ_MEMORY]
        )
        
        # Execute capability
        result = registry.dispatch("test_capability", {"test": "data"}, context)
        assert result.success
        
        # Check metrics
        metrics = registry.get_plugin_metrics()
        assert metrics['total_plugins'] == 1
        assert metrics['active_plugins'] == 1
        assert metrics['capability_invocations'] > 0
        
        # Deactivate plugin
        success = registry.deactivate_plugin("integration_test")
        assert success
        
        # Verify deactivation
        final_metrics = registry.get_plugin_metrics()
        assert final_metrics['active_plugins'] == 0
    
    def test_multiple_plugins_interaction(self, mock_memory_store, mock_consolidator):
        """Test interaction between multiple plugins."""
        registry = create_capability_registry({})
        
        # Register multiple plugins with different capabilities
        plugin1 = TestPlugin("plugin1")
        plugin1._manifest.capabilities = ["capability1"]
        
        plugin2 = TestPlugin("plugin2") 
        plugin2._manifest.capabilities = ["capability2"]
        
        registry.register_plugin(plugin1)
        registry.register_plugin(plugin2)
        
        registry.activate_plugin("plugin1")
        registry.activate_plugin("plugin2")
        
        context = create_agent_context(
            session_id="multi_test",
            agent_name="multi_agent",
            memory_store=mock_memory_store,
            consolidator=mock_consolidator,
            scopes=[CapabilityScope.READ_MEMORY]
        )
        
        # Both capabilities should work
        result1 = registry.dispatch("capability1", {}, context)
        result2 = registry.dispatch("capability2", {}, context)
        
        assert result1.success
        assert result2.success
        
        # Check final metrics
        metrics = registry.get_plugin_metrics()
        assert metrics['total_plugins'] == 2
        assert metrics['active_plugins'] == 2
        assert metrics['total_capabilities'] == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])