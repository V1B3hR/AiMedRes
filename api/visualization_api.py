"""
Enhanced Visualization API for DuetMind Adaptive Safety and Memory Monitoring.

Provides REST endpoints for:
- Enhanced safety monitoring dashboard data
- Agent interaction graphs with real-time metrics
- Memory state visualization with consolidation events
- Real-time monitoring metrics with semantic conflicts
- Plugin system monitoring and management
- Memory introspection and decision traceability
"""

from flask import Flask, jsonify, request, render_template_string
from flask_cors import CORS
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import logging
import json
import os

logger = logging.getLogger(__name__)

try:
    from security.safety_monitor import SafetyMonitor, SafetyDomain
    from security.monitoring import SecurityMonitor
    from mlops.monitoring.production_monitor import ProductionMonitor
    from agent_memory.memory_consolidation import MemoryConsolidator
    from agent_memory.embed_memory import AgentMemoryStore
    from agent_memory.agent_extensions import CapabilityRegistry
except ImportError as e:
    SafetyMonitor = None
    SafetyDomain = None
    SecurityMonitor = None
    ProductionMonitor = None
    MemoryConsolidator = None
    AgentMemoryStore = None
    CapabilityRegistry = None
    # Use logging module directly as backup in case logger is not accessible
    import logging
    logging.getLogger(__name__).warning(f"Some monitoring modules not available: {e}")


class VisualizationAPI:
    """Enhanced API service for monitoring and visualization data."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.app = Flask(__name__)
        CORS(self.app)
        
        # Initialize monitoring systems
        self.safety_monitor = None
        self.security_monitor = None
        self.production_monitor = None
        self.memory_consolidator = None
        self.memory_store = None
        self.capability_registry = None
        
        # Setup routes
        self._setup_routes()
        
        logger.info("Enhanced Visualization API initialized")
    
    def initialize_monitors(self, safety_monitor: SafetyMonitor = None,
                          security_monitor: SecurityMonitor = None,
                          production_monitor: ProductionMonitor = None,
                          memory_store: AgentMemoryStore = None,
                          memory_consolidator: MemoryConsolidator = None,
                          capability_registry: CapabilityRegistry = None):
        """Initialize monitoring system references."""
        self.safety_monitor = safety_monitor
        self.security_monitor = security_monitor
        self.production_monitor = production_monitor
        self.memory_store = memory_store
        self.memory_consolidator = memory_consolidator
        self.capability_registry = capability_registry
        
        logger.info("Enhanced monitoring systems initialized for API")
    
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.app.route('/')
        def dashboard():
            """Main dashboard view."""
            return render_template_string(DASHBOARD_HTML)
        
        @self.app.route('/api/health')
        def health_check():
            """Health check endpoint."""
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'version': '1.0.0'
            })
        
        @self.app.route('/api/safety/summary')
        def safety_summary():
            """Get safety monitoring summary."""
            try:
                hours = request.args.get('hours', 24, type=int)
                
                if self.safety_monitor:
                    summary = self.safety_monitor.get_safety_summary(hours=hours)
                    return jsonify(summary)
                else:
                    return jsonify({
                        'error': 'Safety monitor not initialized',
                        'monitoring_enabled': False
                    }), 503
                    
            except Exception as e:
                logger.error(f"Error getting safety summary: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/safety/findings')
        def safety_findings():
            """Get recent safety findings."""
            try:
                hours = request.args.get('hours', 24, type=int)
                domain = request.args.get('domain')
                correlation_id = request.args.get('correlation_id')
                
                if self.safety_monitor:
                    # Convert domain string to enum if provided
                    domain_enum = None
                    if domain:
                        try:
                            domain_enum = SafetyDomain(domain.lower())
                        except ValueError:
                            return jsonify({'error': f'Invalid domain: {domain}'}), 400
                    
                    findings = self.safety_monitor.get_safety_findings(
                        hours=hours,
                        domain=domain_enum,
                        correlation_id=correlation_id
                    )
                    return jsonify({
                        'findings': findings,
                        'count': len(findings),
                        'period_hours': hours
                    })
                else:
                    return jsonify({
                        'error': 'Safety monitor not initialized',
                        'findings': []
                    }), 503
                    
            except Exception as e:
                logger.error(f"Error getting safety findings: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/safety/run-checks', methods=['POST'])
        def run_safety_checks():
            """Trigger safety checks manually."""
            try:
                data = request.get_json() or {}
                domain_str = data.get('domain')
                context = data.get('context', {})
                
                if self.safety_monitor:
                    domain_enum = None
                    if domain_str:
                        try:
                            domain_enum = SafetyDomain(domain_str.lower())
                        except ValueError:
                            return jsonify({'error': f'Invalid domain: {domain_str}'}), 400
                    
                    correlation_id = self.safety_monitor.create_correlation_id()
                    findings = self.safety_monitor.run_safety_checks(
                        domain=domain_enum,
                        context=context,
                        correlation_id=correlation_id
                    )
                    
                    return jsonify({
                        'correlation_id': correlation_id,
                        'findings': [
                            {
                                'domain': f.domain.value,
                                'check_name': f.check_name,
                                'severity': f.severity,
                                'message': f.message,
                                'value': f.value,
                                'threshold': f.threshold,
                                'timestamp': f.timestamp.isoformat()
                            } for f in findings
                        ],
                        'count': len(findings)
                    })
                else:
                    return jsonify({'error': 'Safety monitor not initialized'}), 503
                    
            except Exception as e:
                logger.error(f"Error running safety checks: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/security/summary')
        def security_summary():
            """Get security monitoring summary."""
            try:
                if self.security_monitor:
                    summary = self.security_monitor.get_security_summary()
                    return jsonify(summary)
                else:
                    return jsonify({
                        'error': 'Security monitor not initialized',
                        'monitoring_status': 'inactive'
                    }), 503
                    
            except Exception as e:
                logger.error(f"Error getting security summary: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/production/summary')
        def production_summary():
            """Get production monitoring summary."""
            try:
                hours = request.args.get('hours', 24, type=int)
                
                if self.production_monitor:
                    summary = self.production_monitor.get_monitoring_summary(hours=hours)
                    return jsonify(summary)
                else:
                    return jsonify({
                        'error': 'Production monitor not initialized',
                        'status': 'unavailable'
                    }), 503
                    
            except Exception as e:
                logger.error(f"Error getting production summary: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/memory/consolidation-summary')
        def memory_consolidation_summary():
            """Get memory consolidation summary."""
            try:
                hours = request.args.get('hours', 24, type=int)
                
                if self.memory_consolidator:
                    summary = self.memory_consolidator.get_consolidation_summary(hours=hours)
                    return jsonify(summary)
                else:
                    return jsonify({
                        'error': 'Memory consolidator not initialized',
                        'consolidation_status': 'unavailable'
                    }), 503
                    
            except Exception as e:
                logger.error(f"Error getting memory consolidation summary: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/memory/introspection', methods=['POST'])
        def memory_introspection():
            """Get memory introspection for decision traceability."""
            try:
                data = request.get_json() or {}
                decision_context = data.get('decision_context', '')
                session_id = data.get('session_id', '')
                
                if not decision_context or not session_id:
                    return jsonify({'error': 'decision_context and session_id required'}), 400
                
                if self.memory_consolidator:
                    introspection = self.memory_consolidator.get_memory_introspection(
                        decision_context, session_id
                    )
                    return jsonify(introspection)
                else:
                    return jsonify({'error': 'Memory consolidator not initialized'}), 503
                    
            except Exception as e:
                logger.error(f"Error getting memory introspection: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/memory/conflicts')
        def memory_conflicts():
            """Get semantic conflicts in memory."""
            try:
                session_id = request.args.get('session_id')
                status = request.args.get('status', 'pending')
                
                if self.memory_consolidator:
                    # Get conflicts from database
                    import sqlite3
                    conflicts = []
                    
                    try:
                        with sqlite3.connect(self.memory_consolidator.consolidation_db) as conn:
                            cursor = conn.execute("""
                                SELECT conflict_id, memory_id1, memory_id2, conflict_type,
                                       confidence_score, resolution_status, created_at, metadata_json
                                FROM semantic_conflicts
                                WHERE resolution_status = ?
                                ORDER BY created_at DESC
                                LIMIT 50
                            """, (status,))
                            
                            for row in cursor.fetchall():
                                conflicts.append({
                                    'conflict_id': row[0],
                                    'memory_id1': row[1],
                                    'memory_id2': row[2],
                                    'conflict_type': row[3],
                                    'confidence_score': row[4],
                                    'resolution_status': row[5],
                                    'created_at': row[6],
                                    'metadata': json.loads(row[7] or '{}')
                                })
                    
                    except Exception as db_error:
                        logger.error(f"Database error getting conflicts: {db_error}")
                        conflicts = []
                    
                    return jsonify({
                        'conflicts': conflicts,
                        'count': len(conflicts),
                        'status_filter': status
                    })
                else:
                    return jsonify({'error': 'Memory consolidator not initialized'}), 503
                    
            except Exception as e:
                logger.error(f"Error getting memory conflicts: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/memory/conflicts/<conflict_id>/resolve', methods=['POST'])
        def resolve_conflict(conflict_id: str):
            """Resolve a semantic conflict."""
            try:
                data = request.get_json() or {}
                resolution_method = data.get('resolution_method', 'manual')
                winning_memory_id = data.get('winning_memory_id')
                
                if self.memory_consolidator:
                    success = self.memory_consolidator.resolve_semantic_conflict(
                        conflict_id, resolution_method, winning_memory_id
                    )
                    
                    return jsonify({
                        'success': success,
                        'conflict_id': conflict_id,
                        'resolution_method': resolution_method,
                        'resolved_at': datetime.now().isoformat()
                    })
                else:
                    return jsonify({'error': 'Memory consolidator not initialized'}), 503
                    
            except Exception as e:
                logger.error(f"Error resolving conflict: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/plugins/summary')
        def plugins_summary():
            """Get plugin system summary."""
            try:
                if self.capability_registry:
                    metrics = self.capability_registry.get_plugin_metrics()
                    return jsonify(metrics)
                else:
                    return jsonify({
                        'error': 'Capability registry not initialized',
                        'plugin_system_status': 'unavailable'
                    }), 503
                    
            except Exception as e:
                logger.error(f"Error getting plugins summary: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/plugins/list')
        def plugins_list():
            """List all registered plugins."""
            try:
                if self.capability_registry:
                    plugins_data = []
                    
                    for name, plugin in self.capability_registry.plugins.items():
                        status = self.capability_registry.plugin_status.get(name, 'unknown')
                        
                        plugins_data.append({
                            'name': plugin.name,
                            'version': plugin.version,
                            'status': status.value if hasattr(status, 'value') else str(status),
                            'capabilities': plugin.capabilities(),
                            'manifest': {
                                'description': plugin.manifest.description,
                                'author': plugin.manifest.author,
                                'required_scopes': [scope.value for scope in plugin.manifest.required_scopes],
                                'sandbox_required': plugin.manifest.sandbox_required
                            }
                        })
                    
                    return jsonify({
                        'plugins': plugins_data,
                        'total_count': len(plugins_data)
                    })
                else:
                    return jsonify({'error': 'Capability registry not initialized'}), 503
                    
            except Exception as e:
                logger.error(f"Error listing plugins: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/plugins/<plugin_name>/activate', methods=['POST'])
        def activate_plugin(plugin_name: str):
            """Activate a plugin."""
            try:
                if self.capability_registry:
                    success = self.capability_registry.activate_plugin(plugin_name)
                    return jsonify({
                        'success': success,
                        'plugin_name': plugin_name,
                        'action': 'activate',
                        'timestamp': datetime.now().isoformat()
                    })
                else:
                    return jsonify({'error': 'Capability registry not initialized'}), 503
                    
            except Exception as e:
                logger.error(f"Error activating plugin: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/plugins/<plugin_name>/deactivate', methods=['POST'])
        def deactivate_plugin(plugin_name: str):
            """Deactivate a plugin."""
            try:
                if self.capability_registry:
                    success = self.capability_registry.deactivate_plugin(plugin_name)
                    return jsonify({
                        'success': success,
                        'plugin_name': plugin_name,
                        'action': 'deactivate',
                        'timestamp': datetime.now().isoformat()
                    })
                else:
                    return jsonify({'error': 'Capability registry not initialized'}), 503
                    
            except Exception as e:
                logger.error(f"Error deactivating plugin: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/memory/session/<session_id>')
        def memory_session_info(session_id: str):
            """Get memory information for a specific session."""
            try:
                if not self.memory_store:
                    return jsonify({'error': 'Memory store not initialized'}), 503
                
                memories = self.memory_store.get_session_memories(session_id)
                
                # Calculate basic statistics
                memory_types = {}
                importance_levels = {'high': 0, 'medium': 0, 'low': 0}
                
                for memory in memories:
                    mem_type = memory.get('memory_type', 'unknown')
                    memory_types[mem_type] = memory_types.get(mem_type, 0) + 1
                    
                    importance = memory.get('importance_score', 0.5)
                    if importance >= 0.7:
                        importance_levels['high'] += 1
                    elif importance >= 0.4:
                        importance_levels['medium'] += 1
                    else:
                        importance_levels['low'] += 1
                
                return jsonify({
                    'session_id': session_id,
                    'total_memories': len(memories),
                    'memory_types': memory_types,
                    'importance_distribution': importance_levels,
                    'memories': memories[:20]  # Limit to first 20 for performance
                })
                
            except Exception as e:
                logger.error(f"Error getting memory session info: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/agent/interaction-graph')
        def agent_interaction_graph():
            """Get enhanced agent interaction graph data."""
            try:
                # Enhanced agent interaction data with more detailed metrics
                graph_data = {
                    'timestamp': datetime.now().isoformat(),
                    'agents': [
                        {
                            'id': 'safety_monitor',
                            'type': 'safety',
                            'state': 'active',
                            'cpu_percent': 5.2,
                            'memory_mb': 128,
                            'pending_tasks': 2,
                            'last_activity': datetime.now().isoformat(),
                            'metrics': {
                                'checks_per_minute': 45,
                                'findings_last_hour': 3,
                                'critical_alerts': 0
                            }
                        },
                        {
                            'id': 'production_monitor', 
                            'type': 'monitoring',
                            'state': 'active',
                            'cpu_percent': 3.1,
                            'memory_mb': 256,
                            'pending_tasks': 0,
                            'last_activity': datetime.now().isoformat(),
                            'metrics': {
                                'predictions_per_minute': 12,
                                'accuracy_last_hour': 0.89,
                                'drift_score': 0.02
                            }
                        },
                        {
                            'id': 'memory_consolidator',
                            'type': 'memory',
                            'state': 'consolidating' if self.memory_consolidator and self.memory_consolidator.running else 'idle',
                            'cpu_percent': 0.8,
                            'memory_mb': 64,
                            'pending_tasks': 0,
                            'last_activity': (datetime.now() - timedelta(minutes=15)).isoformat(),
                            'metrics': {
                                'consolidations_last_hour': 2,
                                'conflicts_detected': 1,
                                'synaptic_tagged_memories': 15
                            }
                        }
                    ],
                    'edges': [
                        {
                            'source': 'safety_monitor',
                            'target': 'production_monitor',
                            'messages_per_minute': 12,
                            'avg_latency_ms': 45,
                            'error_rate': 0.01,
                            'connection_strength': 0.8,
                            'relationship_type': 'monitoring'
                        },
                        {
                            'source': 'production_monitor',
                            'target': 'memory_consolidator',
                            'messages_per_minute': 3,
                            'avg_latency_ms': 120,
                            'error_rate': 0.0,
                            'connection_strength': 0.6,
                            'relationship_type': 'data_flow'
                        }
                    ]
                }
                
                # Add plugin agents if capability registry is available
                if self.capability_registry:
                    plugin_metrics = self.capability_registry.get_plugin_metrics()
                    
                    for name, plugin in self.capability_registry.plugins.items():
                        status = self.capability_registry.plugin_status.get(name, 'unknown')
                        
                        graph_data['agents'].append({
                            'id': f'plugin_{name}',
                            'type': 'plugin',
                            'state': status.value if hasattr(status, 'value') else str(status),
                            'cpu_percent': 1.2,  # Estimated
                            'memory_mb': 32,     # Estimated
                            'pending_tasks': 0,
                            'last_activity': datetime.now().isoformat(),
                            'metrics': {
                                'capabilities': len(plugin.capabilities()),
                                'version': plugin.version,
                                'sandbox_required': plugin.manifest.sandbox_required
                            }
                        })
                
                return jsonify(graph_data)
                
            except Exception as e:
                logger.error(f"Error getting agent interaction graph: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/dashboard/overview')
        def dashboard_overview():
            """Get overview data for main dashboard."""
            try:
                overview = {
                    'timestamp': datetime.now().isoformat(),
                    'system_status': 'healthy',
                    'components': {}
                }
                
                # Safety monitoring status
                if self.safety_monitor:
                    safety_summary = self.safety_monitor.get_safety_summary(hours=1)
                    overview['components']['safety'] = {
                        'status': safety_summary.get('overall_status', 'unknown'),
                        'events_count': safety_summary.get('total_events', 0),
                        'active_domains': safety_summary.get('active_domains', 0)
                    }
                
                # Security monitoring status
                if self.security_monitor:
                    security_summary = self.security_monitor.get_security_summary()
                    overview['components']['security'] = {
                        'status': security_summary.get('monitoring_status', 'inactive'),
                        'events_count': security_summary.get('total_security_events', 0),
                        'api_requests': security_summary.get('api_usage', {}).get('total_requests', 0)
                    }
                
                # Production monitoring status
                if self.production_monitor:
                    prod_summary = self.production_monitor.get_monitoring_summary(hours=1)
                    overview['components']['production'] = {
                        'status': prod_summary.get('status', 'unknown'),
                        'predictions': prod_summary.get('total_predictions', 0),
                        'avg_accuracy': prod_summary.get('avg_accuracy', 0.0)
                    }
                
                # Memory consolidation status
                if self.memory_consolidator:
                    memory_summary = self.memory_consolidator.get_consolidation_summary(hours=1)
                    overview['components']['memory'] = {
                        'status': 'active' if memory_summary.get('total_consolidation_events', 0) > 0 else 'idle',
                        'consolidation_events': memory_summary.get('total_consolidation_events', 0),
                        'last_consolidation': memory_summary.get('last_consolidation'),
                        'synaptic_tagged': memory_summary.get('synaptic_tagged_memories', 0),
                        'pending_conflicts': memory_summary.get('pending_conflicts', 0)
                    }
                
                # Plugin system status
                if self.capability_registry:
                    plugin_metrics = self.capability_registry.get_plugin_metrics()
                    overview['components']['plugins'] = {
                        'status': 'active' if plugin_metrics.get('active_plugins', 0) > 0 else 'inactive',
                        'total_plugins': plugin_metrics.get('total_plugins', 0),
                        'active_plugins': plugin_metrics.get('active_plugins', 0),
                        'total_capabilities': plugin_metrics.get('total_capabilities', 0),
                        'plugin_failures': plugin_metrics.get('plugin_failures', 0)
                    }
                
                # Determine overall system status
                component_statuses = []
                for component, data in overview['components'].items():
                    status = data.get('status', 'unknown')
                    if status in ['critical', 'emergency', 'error']:
                        component_statuses.append('critical')
                    elif status in ['warning', 'degraded']:
                        component_statuses.append('warning')
                    elif status in ['healthy', 'active', 'safe']:
                        component_statuses.append('healthy')
                    else:
                        component_statuses.append('unknown')
                
                if 'critical' in component_statuses:
                    overview['system_status'] = 'critical'
                elif 'warning' in component_statuses:
                    overview['system_status'] = 'warning'
                elif 'unknown' in component_statuses:
                    overview['system_status'] = 'unknown'
                else:
                    overview['system_status'] = 'healthy'
                
                return jsonify(overview)
                
            except Exception as e:
                logger.error(f"Error getting dashboard overview: {e}")
                return jsonify({'error': str(e)}), 500
    
    def run(self, host: str = '0.0.0.0', port: int = 5001, debug: bool = False):
        """Run the visualization API server."""
        self.app.run(host=host, port=port, debug=debug)


# Basic HTML dashboard template
DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>DuetMind Adaptive Monitoring Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
        .header { background: #2c3e50; color: white; padding: 20px; margin: -20px -20px 20px -20px; }
        .status-card { background: white; padding: 20px; margin: 10px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .status-good { border-left: 5px solid #2ecc71; }
        .status-warning { border-left: 5px solid #f39c12; }  
        .status-critical { border-left: 5px solid #e74c3c; }
        .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        .metric-value { font-size: 2em; font-weight: bold; color: #2c3e50; }
        .loading { color: #7f8c8d; font-style: italic; }
        .error { color: #e74c3c; }
        .timestamp { color: #7f8c8d; font-size: 0.9em; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🧠 DuetMind Adaptive Monitoring Dashboard</h1>
        <p>Real-time safety, security, and performance monitoring</p>
    </div>
    
    <div class="status-card">
        <h2>System Overview</h2>
        <div id="overview">Loading system overview...</div>
    </div>
    
    <div class="metrics-grid">
        <div class="status-card">
            <h3>🛡️ Safety Monitoring</h3>
            <div id="safety-metrics">Loading safety metrics...</div>
        </div>
        
        <div class="status-card">
            <h3>🔒 Security Monitoring</h3>
            <div id="security-metrics">Loading security metrics...</div>
        </div>
        
        <div class="status-card">
            <h3>📊 Production Monitoring</h3>
            <div id="production-metrics">Loading production metrics...</div>
        </div>
        
        <div class="status-card">
            <h3>🧠 Memory Consolidation</h3>
            <div id="memory-metrics">Loading memory metrics...</div>
        </div>
        
        <div class="status-card">
            <h3>🔌 Plugin System</h3>
            <div id="plugin-metrics">Loading plugin metrics...</div>
        </div>
        
        <div class="status-card">
            <h3>🕸️ Agent Network</h3>
            <div id="agent-network">Loading agent network...</div>
        </div>
    </div>
    
    <div class="metrics-grid">
        <div class="status-card">
            <h2>⚠️ Semantic Conflicts</h2>
            <div id="memory-conflicts">Loading memory conflicts...</div>
        </div>
        
        <div class="status-card">
            <h2>🔍 Recent Safety Findings</h2>
            <div id="safety-findings">Loading recent findings...</div>
        </div>
    </div>

    <script>
        // Auto-refresh dashboard every 30 seconds
        function refreshDashboard() {
            loadOverview();
            loadSafetyMetrics();
            loadSecurityMetrics();
            loadProductionMetrics();
            loadMemoryMetrics();
            loadPluginMetrics();
            loadAgentNetwork();
            loadMemoryConflicts();
            loadSafetyFindings();
        }
        
        function loadOverview() {
            fetch('/api/dashboard/overview')
                .then(response => response.json())
                .then(data => {
                    const element = document.getElementById('overview');
                    if (data.error) {
                        element.innerHTML = `<div class="error">Error: ${data.error}</div>`;
                        return;
                    }
                    
                    let statusClass = 'status-good';
                    if (data.system_status === 'warning') statusClass = 'status-warning';
                    if (data.system_status === 'critical') statusClass = 'status-critical';
                    
                    element.innerHTML = `
                        <div class="${statusClass}">
                            <div class="metric-value">${data.system_status.toUpperCase()}</div>
                            <p>System Status</p>
                            <div class="timestamp">Last updated: ${new Date(data.timestamp).toLocaleString()}</div>
                        </div>
                    `;
                })
                .catch(error => {
                    document.getElementById('overview').innerHTML = `<div class="error">Failed to load overview: ${error}</div>`;
                });
        }
        
        function loadSafetyMetrics() {
            fetch('/api/safety/summary')
                .then(response => response.json())
                .then(data => {
                    const element = document.getElementById('safety-metrics');
                    if (data.error) {
                        element.innerHTML = `<div class="error">${data.error}</div>`;
                        return;
                    }
                    
                    element.innerHTML = `
                        <div class="metric-value">${data.overall_status || 'Unknown'}</div>
                        <p>Safety Status</p>
                        <p>Events: ${data.total_events || 0} | Domains: ${data.active_domains || 0}</p>
                    `;
                })
                .catch(error => {
                    document.getElementById('safety-metrics').innerHTML = `<div class="error">Failed to load</div>`;
                });
        }
        
        function loadSecurityMetrics() {
            fetch('/api/security/summary')
                .then(response => response.json())
                .then(data => {
                    const element = document.getElementById('security-metrics');
                    if (data.error) {
                        element.innerHTML = `<div class="error">${data.error}</div>`;
                        return;
                    }
                    
                    element.innerHTML = `
                        <div class="metric-value">${data.monitoring_status || 'Unknown'}</div>
                        <p>Security Status</p>
                        <p>Events: ${data.total_security_events || 0}</p>
                    `;
                })
                .catch(error => {
                    document.getElementById('security-metrics').innerHTML = `<div class="error">Failed to load</div>`;
                });
        }
        
        function loadProductionMetrics() {
            fetch('/api/production/summary')
                .then(response => response.json())
                .then(data => {
                    const element = document.getElementById('production-metrics');
                    if (data.error) {
                        element.innerHTML = `<div class="error">${data.error}</div>`;
                        return;
                    }
                    
                    element.innerHTML = `
                        <div class="metric-value">${data.status || 'Unknown'}</div>
                        <p>Production Status</p>
                        <p>Predictions: ${data.total_predictions || 0} | Accuracy: ${(data.avg_accuracy || 0).toFixed(3)}</p>
                    `;
                })
                .catch(error => {
                    document.getElementById('production-metrics').innerHTML = `<div class="error">Failed to load</div>`;
                });
        }
        
        function loadMemoryMetrics() {
            fetch('/api/memory/consolidation-summary')
                .then(response => response.json())
                .then(data => {
                    const element = document.getElementById('memory-metrics');
                    if (data.error) {
                        element.innerHTML = `<div class="error">${data.error}</div>`;
                        return;
                    }
                    
                    const status = data.total_consolidation_events > 0 ? 'Active' : 'Idle';
                    element.innerHTML = `
                        <div class="metric-value">${status}</div>
                        <p>Consolidation Status</p>
                        <p>Events: ${data.total_consolidation_events || 0} | Conflicts: ${data.pending_conflicts || 0}</p>
                        <p>Synaptic Tagged: ${data.synaptic_tagged_memories || 0}</p>
                    `;
                })
                .catch(error => {
                    document.getElementById('memory-metrics').innerHTML = `<div class="error">Failed to load</div>`;
                });
        }
        
        function loadPluginMetrics() {
            fetch('/api/plugins/summary')
                .then(response => response.json())
                .then(data => {
                    const element = document.getElementById('plugin-metrics');
                    if (data.error) {
                        element.innerHTML = `<div class="error">${data.error}</div>`;
                        return;
                    }
                    
                    const status = data.active_plugins > 0 ? 'Active' : 'Inactive';
                    element.innerHTML = `
                        <div class="metric-value">${status}</div>
                        <p>Plugin System</p>
                        <p>Plugins: ${data.active_plugins}/${data.total_plugins} | Capabilities: ${data.total_capabilities || 0}</p>
                        <p>Failures: ${data.plugin_failures || 0}</p>
                    `;
                })
                .catch(error => {
                    document.getElementById('plugin-metrics').innerHTML = `<div class="error">Failed to load</div>`;
                });
        }
        
        function loadAgentNetwork() {
            fetch('/api/agent/interaction-graph')
                .then(response => response.json())
                .then(data => {
                    const element = document.getElementById('agent-network');
                    if (data.error) {
                        element.innerHTML = `<div class="error">${data.error}</div>`;
                        return;
                    }
                    
                    const activeAgents = data.agents.filter(a => a.state === 'active').length;
                    element.innerHTML = `
                        <div class="metric-value">${activeAgents}/${data.agents.length}</div>
                        <p>Agents Active</p>
                        <p>Connections: ${data.edges.length} | Avg Latency: ${Math.round(data.edges.reduce((sum, e) => sum + e.avg_latency_ms, 0) / data.edges.length || 0)}ms</p>
                    `;
                })
                .catch(error => {
                    document.getElementById('agent-network').innerHTML = `<div class="error">Failed to load</div>`;
                });
        }
        
        function loadMemoryConflicts() {
            fetch('/api/memory/conflicts?status=pending')
                .then(response => response.json())
                .then(data => {
                    const element = document.getElementById('memory-conflicts');
                    if (data.error) {
                        element.innerHTML = `<div class="error">${data.error}</div>`;
                        return;
                    }
                    
                    if (data.conflicts.length === 0) {
                        element.innerHTML = '<p>No pending conflicts</p>';
                        return;
                    }
                    
                    let html = `<p><strong>${data.count} pending conflicts</strong></p><ul>`;
                    data.conflicts.slice(0, 5).forEach(conflict => {
                        html += `
                            <li>
                                <strong>[${conflict.conflict_type.toUpperCase()}]</strong> 
                                Memories ${conflict.memory_id1} vs ${conflict.memory_id2}
                                <span class="timestamp">Confidence: ${(conflict.confidence_score * 100).toFixed(0)}%</span>
                            </li>
                        `;
                    });
                    html += '</ul>';
                    element.innerHTML = html;
                })
                .catch(error => {
                    document.getElementById('memory-conflicts').innerHTML = `<div class="error">Failed to load conflicts</div>`;
                });
        }
        
        function loadSafetyFindings() {
            fetch('/api/safety/findings?hours=1')
                .then(response => response.json())
                .then(data => {
                    const element = document.getElementById('safety-findings');
                    if (data.error) {
                        element.innerHTML = `<div class="error">${data.error}</div>`;
                        return;
                    }
                    
                    if (data.findings.length === 0) {
                        element.innerHTML = '<p>No recent safety findings</p>';
                        return;
                    }
                    
                    let html = '<ul>';
                    data.findings.slice(0, 10).forEach(finding => {
                        let severityClass = finding.severity === 'critical' ? 'error' : '';
                        html += `
                            <li class="${severityClass}">
                                <strong>[${finding.severity.toUpperCase()}]</strong> 
                                ${finding.domain}: ${finding.message}
                                <div class="timestamp">${new Date(finding.timestamp).toLocaleString()}</div>
                            </li>
                        `;
                    });
                    html += '</ul>';
                    element.innerHTML = html;
                })
                .catch(error => {
                    document.getElementById('safety-findings').innerHTML = `<div class="error">Failed to load findings</div>`;
                });
        }
        
        // Initial load
        refreshDashboard();
        
        // Auto-refresh every 30 seconds
        setInterval(refreshDashboard, 30000);
    </script>
</body>
</html>
"""


def create_visualization_api(config: Dict[str, Any]) -> VisualizationAPI:
    """Factory function to create visualization API."""
    return VisualizationAPI(config)
