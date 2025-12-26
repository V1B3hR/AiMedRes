"""
API for Custom Agent Behaviors and Extensions

Provides a plugin architecture that allows external developers or internal modules
to add new agent types, tools, validators, or memory transforms securely and
versionably.

Core Features:
- Capability Registry for managing agent capabilities
- Agent Interface protocol for plugins
- Sandbox execution environment
- Lifecycle hooks and dependency injection
- Versioning and compatibility management
- Security and policy enforcement
"""

import hashlib
import importlib
import inspect
import json
import logging
import subprocess
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Union

logger = logging.getLogger("duetmind.agent.extensions")


class CapabilityScope(Enum):
    """Security scopes for agent capabilities."""

    READ_MEMORY = "read_memory"
    WRITE_MEMORY = "write_memory"
    WRITE_SEMANTIC = "write_semantic"
    INVOKE_EXTERNAL_API = "invoke_external_api"
    ACCESS_SAFETY_MONITOR = "access_safety_monitor"
    MODIFY_AGENT_STATE = "modify_agent_state"
    SYSTEM_ADMIN = "system_admin"


class PluginStatus(Enum):
    """Plugin lifecycle status."""

    REGISTERED = "registered"
    ACTIVE = "active"
    INACTIVE = "inactive"
    DEPRECATED = "deprecated"
    FAILED = "failed"
    SANDBOXED = "sandboxed"


@dataclass
class AgentMessage:
    """Message structure for agent communication."""

    message_id: str
    sender_id: str
    recipient_id: str
    message_type: str
    payload: Dict[str, Any]
    timestamp: datetime
    correlation_id: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class AgentContext:
    """Context provided to agent plugins."""

    session_id: str
    agent_name: str
    safety_state: str
    memory_accessor: Any  # Will be injected
    logger: logging.Logger
    config: Dict[str, Any]
    scopes: List[CapabilityScope]
    correlation_id: Optional[str] = None


@dataclass
class AgentResult:
    """Result returned by agent handlers."""

    success: bool
    result_data: Dict[str, Any]
    error_message: Optional[str] = None
    next_actions: List[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.next_actions is None:
            self.next_actions = []
        if self.metadata is None:
            self.metadata = {}


@dataclass
class PluginManifest:
    """Plugin manifest declaration."""

    name: str
    version: str
    description: str
    author: str
    required_scopes: List[CapabilityScope]
    requires_api: str  # Semantic version range
    capabilities: List[str]
    dependencies: List[str] = None
    config_schema: Dict[str, Any] = None
    sandbox_required: bool = True

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.config_schema is None:
            self.config_schema = {}


class AgentPlugin(Protocol):
    """Protocol that all agent plugins must implement."""

    @property
    def name(self) -> str:
        """Plugin name."""
        ...

    @property
    def version(self) -> str:
        """Plugin version."""
        ...

    @property
    def manifest(self) -> PluginManifest:
        """Plugin manifest."""
        ...

    def capabilities(self) -> List[str]:
        """Return list of capabilities this plugin provides."""
        ...

    def handle(self, message: AgentMessage, context: AgentContext) -> AgentResult:
        """Handle an agent message."""
        ...

    def healthcheck(self) -> Dict[str, Any]:
        """Return health status of the plugin."""
        ...

    def on_register(self, registry: "CapabilityRegistry") -> bool:
        """Called when plugin is registered."""
        ...

    def on_activate(self) -> bool:
        """Called when plugin is activated."""
        ...

    def on_deactivate(self) -> bool:
        """Called when plugin is deactivated."""
        ...


class MemoryAccessor:
    """Safe memory accessor for plugins."""

    def __init__(self, memory_store, consolidator, allowed_scopes: List[CapabilityScope]):
        self.memory_store = memory_store
        self.consolidator = consolidator
        self.allowed_scopes = allowed_scopes

    def read_memories(self, session_id: str, query: str, limit: int = 10) -> List[Dict]:
        """Read memories if scope allows."""
        if CapabilityScope.READ_MEMORY not in self.allowed_scopes:
            raise PermissionError("Plugin does not have read_memory scope")

        return self.memory_store.retrieve_memories(
            session_id=session_id, query=query, limit=min(limit, 50)  # Hard limit for safety
        )

    def store_memory(
        self, session_id: str, content: str, memory_type: str = "plugin", importance: float = 0.5
    ) -> Optional[int]:
        """Store memory if scope allows."""
        if CapabilityScope.WRITE_MEMORY not in self.allowed_scopes:
            raise PermissionError("Plugin does not have write_memory scope")

        return self.memory_store.store_memory(
            session_id=session_id,
            content=content,
            memory_type=memory_type,
            importance=min(importance, 0.8),  # Limit plugin memory importance
            metadata={"source": "plugin"},
        )

    def get_consolidation_info(self, session_id: str) -> Dict[str, Any]:
        """Get consolidation information."""
        if CapabilityScope.READ_MEMORY not in self.allowed_scopes:
            raise PermissionError("Plugin does not have read_memory scope")

        return self.consolidator.get_consolidation_summary(hours=24)


class PolicyEngine:
    """Policy engine for plugin security."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.blocked_capabilities = set()
        self.global_safety_state = "SAFE"

    def check_capability_allowed(self, capability: str, scopes: List[CapabilityScope]) -> bool:
        """Check if capability is allowed given current state."""

        # Block all capabilities in critical safety state
        if self.global_safety_state == "CRITICAL":
            logger.warning(f"Blocking capability {capability} due to critical safety state")
            return False

        # Check if capability is globally blocked
        if capability in self.blocked_capabilities:
            logger.warning(f"Capability {capability} is globally blocked")
            return False

        # Check scope requirements (simplified)
        required_scope_mapping = {
            "memory_query": CapabilityScope.READ_MEMORY,
            "memory_store": CapabilityScope.WRITE_MEMORY,
            "semantic_analysis": CapabilityScope.WRITE_SEMANTIC,
            "external_call": CapabilityScope.INVOKE_EXTERNAL_API,
        }

        required_scope = required_scope_mapping.get(capability)
        if required_scope and required_scope not in scopes:
            logger.warning(f"Capability {capability} requires scope {required_scope}")
            return False

        return True

    def update_safety_state(self, state: str):
        """Update global safety state."""
        self.global_safety_state = state
        logger.info(f"Policy engine updated safety state to {state}")


class SandboxExecutor:
    """Sandbox execution environment for plugins."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.resource_limits = {
            "max_memory_mb": config.get("max_plugin_memory_mb", 128),
            "max_cpu_seconds": config.get("max_plugin_cpu_seconds", 30),
            "max_file_handles": config.get("max_plugin_file_handles", 10),
        }

    def execute_plugin_method(self, plugin: AgentPlugin, method_name: str, *args, **kwargs) -> Any:
        """Execute plugin method in sandbox."""
        try:
            # Get the method
            method = getattr(plugin, method_name)

            # Simple timeout execution (could be enhanced with proper sandboxing)
            result = self._execute_with_timeout(method, args, kwargs)
            return result

        except Exception as e:
            logger.error(f"Sandbox execution error for {plugin.name}.{method_name}: {e}")
            raise

    def _execute_with_timeout(self, method: Callable, args: tuple, kwargs: dict) -> Any:
        """Execute method with timeout."""
        result = None
        exception = None

        def target():
            nonlocal result, exception
            try:
                result = method(*args, **kwargs)
            except Exception as e:
                exception = e

        thread = threading.Thread(target=target)
        thread.start()
        thread.join(timeout=self.resource_limits["max_cpu_seconds"])

        if thread.is_alive():
            # Thread is still running, timeout occurred
            logger.error(
                f"Plugin method timed out after {self.resource_limits['max_cpu_seconds']} seconds"
            )
            raise TimeoutError("Plugin method execution timed out")

        if exception:
            raise exception

        return result


class CapabilityRegistry:
    """Central registry for agent capabilities and plugins."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.plugins: Dict[str, AgentPlugin] = {}
        self.plugin_status: Dict[str, PluginStatus] = {}
        self.capability_mapping: Dict[str, str] = {}  # capability -> plugin_name
        self.policy_engine = PolicyEngine(config)
        self.sandbox_executor = SandboxExecutor(config)
        self.lifecycle_hooks: Dict[str, List[Callable]] = {
            "on_register": [],
            "on_pre_message": [],
            "on_post_message": [],
            "on_error": [],
            "on_consolidation_cycle": [],
        }

        # Metrics
        self.metrics = {
            "plugin_registrations": 0,
            "capability_invocations": 0,
            "plugin_failures": 0,
            "security_violations": 0,
        }

        logger.info("Capability Registry initialized")

    def register_plugin(self, plugin: AgentPlugin) -> bool:
        """Register a new plugin."""
        try:
            # Validate plugin
            if not self._validate_plugin(plugin):
                return False

            # Check if plugin already exists
            if plugin.name in self.plugins:
                logger.warning(f"Plugin {plugin.name} already registered")
                return False

            # Verify API compatibility
            if not self._check_api_compatibility(plugin.manifest.requires_api):
                logger.error(f"Plugin {plugin.name} API compatibility check failed")
                return False

            # Register capabilities
            for capability in plugin.capabilities():
                if capability in self.capability_mapping:
                    logger.error(
                        f"Capability {capability} already registered by {self.capability_mapping[capability]}"
                    )
                    return False
                self.capability_mapping[capability] = plugin.name

            # Store plugin
            self.plugins[plugin.name] = plugin
            self.plugin_status[plugin.name] = PluginStatus.REGISTERED

            # Run registration hook
            try:
                if hasattr(plugin, "on_register"):
                    if not plugin.on_register(self):
                        logger.error(f"Plugin {plugin.name} registration hook failed")
                        self._unregister_plugin(plugin.name)
                        return False
            except Exception as e:
                logger.error(f"Plugin {plugin.name} registration hook error: {e}")
                self._unregister_plugin(plugin.name)
                return False

            # Run global registration hooks
            for hook in self.lifecycle_hooks["on_register"]:
                try:
                    hook(plugin)
                except Exception as e:
                    logger.error(f"Global registration hook error: {e}")

            self.metrics["plugin_registrations"] += 1
            logger.info(f"Plugin {plugin.name} v{plugin.version} registered successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to register plugin {plugin.name}: {e}")
            return False

    def _validate_plugin(self, plugin: AgentPlugin) -> bool:
        """Validate plugin meets requirements."""
        # Check required methods
        required_methods = ["name", "version", "capabilities", "handle", "healthcheck"]
        for method in required_methods:
            if not hasattr(plugin, method):
                logger.error(f"Plugin missing required method: {method}")
                return False

        # Check manifest
        if not hasattr(plugin, "manifest") or not isinstance(plugin.manifest, PluginManifest):
            logger.error("Plugin missing or invalid manifest")
            return False

        # Validate capabilities are not empty
        capabilities = plugin.capabilities()
        if not capabilities or not isinstance(capabilities, list):
            logger.error("Plugin must provide non-empty capabilities list")
            return False

        return True

    def _check_api_compatibility(self, version_requirement: str) -> bool:
        """Check if plugin API version is compatible."""
        # Simplified version check (could use semantic versioning library)
        current_api_version = "1.0.0"

        # Basic compatibility check
        if ">=" in version_requirement:
            required = version_requirement.split(">=")[1].split(",")[0].strip()
            return current_api_version >= required
        elif "==" in version_requirement:
            required = version_requirement.split("==")[1].strip()
            return current_api_version == required
        elif "<" in version_requirement:
            # Handle <2.0.0 case
            return True  # Always compatible for now

        return True  # Default allow

    def dispatch(
        self, capability: str, payload: Dict[str, Any], context: AgentContext
    ) -> AgentResult:
        """Dispatch capability request to appropriate plugin."""
        try:
            # Check if capability is registered
            if capability not in self.capability_mapping:
                return AgentResult(
                    success=False,
                    result_data={},
                    error_message=f"Capability {capability} not registered",
                )

            plugin_name = self.capability_mapping[capability]
            plugin = self.plugins[plugin_name]

            # Check plugin status
            if self.plugin_status[plugin_name] != PluginStatus.ACTIVE:
                return AgentResult(
                    success=False,
                    result_data={},
                    error_message=f"Plugin {plugin_name} is not active",
                )

            # Check policy
            if not self.policy_engine.check_capability_allowed(capability, context.scopes):
                self.metrics["security_violations"] += 1
                return AgentResult(
                    success=False,
                    result_data={},
                    error_message=f"Capability {capability} not allowed by policy",
                )

            # Create message
            message = AgentMessage(
                message_id=str(uuid.uuid4()),
                sender_id="capability_registry",
                recipient_id=plugin_name,
                message_type=capability,
                payload=payload,
                timestamp=datetime.now(),
                correlation_id=context.correlation_id,
            )

            # Run pre-message hooks
            for hook in self.lifecycle_hooks["on_pre_message"]:
                try:
                    hook(message, context)
                except Exception as e:
                    logger.error(f"Pre-message hook error: {e}")

            # Execute plugin
            if plugin.manifest.sandbox_required:
                result = self.sandbox_executor.execute_plugin_method(
                    plugin, "handle", message, context
                )
            else:
                result = plugin.handle(message, context)

            # Run post-message hooks
            for hook in self.lifecycle_hooks["on_post_message"]:
                try:
                    hook(message, context, result)
                except Exception as e:
                    logger.error(f"Post-message hook error: {e}")

            self.metrics["capability_invocations"] += 1
            return result

        except Exception as e:
            logger.error(f"Error dispatching capability {capability}: {e}")
            self.metrics["plugin_failures"] += 1

            # Run error hooks
            for hook in self.lifecycle_hooks["on_error"]:
                try:
                    hook(capability, e, context)
                except Exception as hook_error:
                    logger.error(f"Error hook failed: {hook_error}")

            return AgentResult(
                success=False, result_data={}, error_message=f"Plugin execution error: {str(e)}"
            )

    def activate_plugin(self, plugin_name: str) -> bool:
        """Activate a registered plugin."""
        if plugin_name not in self.plugins:
            logger.error(f"Plugin {plugin_name} not registered")
            return False

        plugin = self.plugins[plugin_name]

        try:
            # Run health check
            health = plugin.healthcheck()
            if not health.get("healthy", False):
                logger.error(f"Plugin {plugin_name} health check failed: {health}")
                return False

            # Run activation hook
            if hasattr(plugin, "on_activate"):
                if not plugin.on_activate():
                    logger.error(f"Plugin {plugin_name} activation hook failed")
                    return False

            self.plugin_status[plugin_name] = PluginStatus.ACTIVE
            logger.info(f"Plugin {plugin_name} activated")
            return True

        except Exception as e:
            logger.error(f"Failed to activate plugin {plugin_name}: {e}")
            self.plugin_status[plugin_name] = PluginStatus.FAILED
            return False

    def deactivate_plugin(self, plugin_name: str) -> bool:
        """Deactivate a plugin."""
        if plugin_name not in self.plugins:
            return False

        plugin = self.plugins[plugin_name]

        try:
            if hasattr(plugin, "on_deactivate"):
                plugin.on_deactivate()

            self.plugin_status[plugin_name] = PluginStatus.INACTIVE
            logger.info(f"Plugin {plugin_name} deactivated")
            return True

        except Exception as e:
            logger.error(f"Failed to deactivate plugin {plugin_name}: {e}")
            return False

    def _unregister_plugin(self, plugin_name: str):
        """Internal method to unregister a plugin."""
        if plugin_name in self.plugins:
            # Remove capability mappings
            capabilities_to_remove = []
            for capability, mapped_plugin in self.capability_mapping.items():
                if mapped_plugin == plugin_name:
                    capabilities_to_remove.append(capability)

            for capability in capabilities_to_remove:
                del self.capability_mapping[capability]

            # Remove plugin
            del self.plugins[plugin_name]
            if plugin_name in self.plugin_status:
                del self.plugin_status[plugin_name]

    def get_plugin_metrics(self) -> Dict[str, Any]:
        """Get plugin system metrics."""
        status_counts = {}
        for status in self.plugin_status.values():
            status_counts[status.value] = status_counts.get(status.value, 0) + 1

        return {
            **self.metrics,
            "total_plugins": len(self.plugins),
            "active_plugins": status_counts.get("active", 0),
            "total_capabilities": len(self.capability_mapping),
            "plugin_status_distribution": status_counts,
            "generated_at": datetime.now().isoformat(),
        }

    def add_lifecycle_hook(self, hook_type: str, hook_func: Callable):
        """Add a lifecycle hook."""
        if hook_type in self.lifecycle_hooks:
            self.lifecycle_hooks[hook_type].append(hook_func)
        else:
            logger.error(f"Unknown hook type: {hook_type}")


# Example Plugin Implementation
class ClinicalGuidelineAgent(AgentPlugin):
    """Example plugin that provides clinical guideline capabilities."""

    def __init__(self):
        self._name = "clinical_guideline_agent"
        self._version = "1.0.0"
        self._manifest = PluginManifest(
            name=self._name,
            version=self._version,
            description="Provides clinical guideline analysis and recommendations",
            author="DuetMind Core Team",
            required_scopes=[CapabilityScope.READ_MEMORY, CapabilityScope.WRITE_SEMANTIC],
            requires_api=">=1.0.0,<2.0.0",
            capabilities=["generate_guideline_explanation", "assess_guideline_compliance"],
            dependencies=[],
            sandbox_required=True,
        )
        self._active = False

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> str:
        return self._version

    @property
    def manifest(self) -> PluginManifest:
        return self._manifest

    def capabilities(self) -> List[str]:
        return self._manifest.capabilities

    def handle(self, message: AgentMessage, context: AgentContext) -> AgentResult:
        """Handle guideline-related requests."""
        try:
            capability = message.message_type
            payload = message.payload

            if capability == "generate_guideline_explanation":
                return self._generate_explanation(payload, context)
            elif capability == "assess_guideline_compliance":
                return self._assess_compliance(payload, context)
            else:
                return AgentResult(
                    success=False, result_data={}, error_message=f"Unknown capability: {capability}"
                )

        except Exception as e:
            return AgentResult(
                success=False, result_data={}, error_message=f"Plugin error: {str(e)}"
            )

    def _generate_explanation(self, payload: Dict[str, Any], context: AgentContext) -> AgentResult:
        """Generate clinical guideline explanation."""
        condition = payload.get("condition", "")

        # Simplified guideline logic
        guidelines = {
            "alzheimers_stage_2": {
                "explanation": "For Stage 2 Alzheimer's, guidelines recommend cognitive assessment with MMSE, genetic testing for APOE4, and consideration of cholinesterase inhibitors.",
                "key_factors": ["MMSE score", "APOE4 status", "functional decline rate"],
                "recommendations": [
                    "Regular cognitive monitoring",
                    "Caregiver support",
                    "Safety assessment",
                ],
            }
        }

        guideline = guidelines.get(
            condition.lower(),
            {
                "explanation": f"General guidelines for {condition} require individualized assessment.",
                "key_factors": ["Patient history", "Current symptoms", "Risk factors"],
                "recommendations": ["Consult specialist", "Follow standard protocols"],
            },
        )

        return AgentResult(
            success=True,
            result_data={
                "condition": condition,
                "guideline": guideline,
                "generated_by": self.name,
                "timestamp": datetime.now().isoformat(),
            },
        )

    def _assess_compliance(self, payload: Dict[str, Any], context: AgentContext) -> AgentResult:
        """Assess guideline compliance."""
        # Simplified compliance check
        return AgentResult(
            success=True,
            result_data={
                "compliance_score": 0.85,
                "compliant_items": ["MMSE performed", "Risk factors assessed"],
                "missing_items": ["APOE4 testing"],
                "assessed_by": self.name,
            },
        )

    def healthcheck(self) -> Dict[str, Any]:
        """Return health status."""
        return {
            "healthy": True,  # Always healthy for demo purposes
            "status": "active" if self._active else "ready",
            "last_check": datetime.now().isoformat(),
        }

    def on_register(self, registry: CapabilityRegistry) -> bool:
        """Called when plugin is registered."""
        logger.info(
            f"Clinical Guideline Agent registered with {len(self.capabilities())} capabilities"
        )
        return True

    def on_activate(self) -> bool:
        """Called when plugin is activated."""
        self._active = True
        logger.info("Clinical Guideline Agent activated")
        return True

    def on_deactivate(self) -> bool:
        """Called when plugin is deactivated."""
        self._active = False
        logger.info("Clinical Guideline Agent deactivated")
        return True


def create_capability_registry(config: Dict[str, Any]) -> CapabilityRegistry:
    """Factory function to create capability registry."""
    return CapabilityRegistry(config)


def create_agent_context(
    session_id: str, agent_name: str, memory_store, consolidator, scopes: List[CapabilityScope]
) -> AgentContext:
    """Factory function to create agent context."""
    return AgentContext(
        session_id=session_id,
        agent_name=agent_name,
        safety_state="SAFE",
        memory_accessor=MemoryAccessor(memory_store, consolidator, scopes),
        logger=logger,
        config={},
        scopes=scopes,
        correlation_id=str(uuid.uuid4()),
    )
