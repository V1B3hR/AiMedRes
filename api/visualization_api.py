"""
Visualization API for DuetMind Adaptive Safety and Memory Monitoring.

Provides REST endpoints for:
- Safety monitoring dashboard data
- Agent interaction graphs
- Memory state visualization
- Real-time monitoring metrics
"""

from flask import Flask, jsonify, request, render_template_string
from flask_cors import CORS
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import logging
import json
import os

# Import our monitoring systems
from ..security.safety_monitor import SafetyMonitor, SafetyDomain
from ..security.monitoring import SecurityMonitor
from ..mlops.monitoring.production_monitor import ProductionMonitor
from ..agent_memory.memory_consolidation import MemoryConsolidator
from ..agent_memory.embed_memory import AgentMemoryStore

logger = logging.getLogger('duetmind.visualization.api')


class VisualizationAPI:
    """API service for monitoring and visualization data."""
    
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
        
        # Setup routes
        self._setup_routes()
        
        logger.info("Visualization API initialized")
    
    def initialize_monitors(self, safety_monitor: SafetyMonitor = None,
                          security_monitor: SecurityMonitor = None,
                          production_monitor: ProductionMonitor = None,
                          memory_store: AgentMemoryStore = None,
                          memory_consolidator: MemoryConsolidator = None):
        """Initialize monitoring system references."""
        self.safety_monitor = safety_monitor
        self.security_monitor = security_monitor
        self.production_monitor = production_monitor
        self.memory_store = memory_store
        self.memory_consolidator = memory_consolidator
        
        logger.info("Monitoring systems initialized for API")
    
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
            """Get agent interaction graph data."""
            try:
                # Mock agent interaction data (would be real in production)
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
                            'last_activity': datetime.now().isoformat()
                        },
                        {
                            'id': 'production_monitor', 
                            'type': 'monitoring',
                            'state': 'active',
                            'cpu_percent': 3.1,
                            'memory_mb': 256,
                            'pending_tasks': 0,
                            'last_activity': datetime.now().isoformat()
                        },
                        {
                            'id': 'memory_consolidator',
                            'type': 'memory',
                            'state': 'idle',
                            'cpu_percent': 0.8,
                            'memory_mb': 64,
                            'pending_tasks': 0,
                            'last_activity': (datetime.now() - timedelta(minutes=15)).isoformat()
                        }
                    ],
                    'edges': [
                        {
                            'source': 'safety_monitor',
                            'target': 'production_monitor',
                            'messages_per_minute': 12,
                            'avg_latency_ms': 45,
                            'error_rate': 0.01,
                            'connection_strength': 0.8
                        },
                        {
                            'source': 'production_monitor',
                            'target': 'memory_consolidator',
                            'messages_per_minute': 3,
                            'avg_latency_ms': 120,
                            'error_rate': 0.0,
                            'connection_strength': 0.6
                        }
                    ]
                }
                
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
                        'last_consolidation': memory_summary.get('last_consolidation')
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
    </div>
    
    <div class="status-card">
        <h2>Recent Safety Findings</h2>
        <div id="safety-findings">Loading recent findings...</div>
    </div>

    <script>
        // Auto-refresh dashboard every 30 seconds
        function refreshDashboard() {
            loadOverview();
            loadSafetyMetrics();
            loadSecurityMetrics();
            loadProductionMetrics();
            loadMemoryMetrics();
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
                        <p>Events: ${data.total_consolidation_events || 0}</p>
                    `;
                })
                .catch(error => {
                    document.getElementById('memory-metrics').innerHTML = `<div class="error">Failed to load</div>`;
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