#!/usr/bin/env python3
"""
Demo of Advanced Safety Monitoring System for DuetMind Adaptive.

Demonstrates the enhanced safety monitoring features:
- Safety domains and pluggable checks
- Real-time safety monitoring
- Correlation tracking
- Memory consolidation
- Visualization dashboard
"""

import sys
import os
import time
import threading
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from security.safety_monitor import SafetyMonitor, SafetyDomain, SafetyFinding
from security.safety_checks import (
    SystemSafetyCheck, DataSafetyCheck, ModelSafetyCheck,
    InteractionSafetyCheck, ClinicalSafetyCheck
)


def demo_safety_monitoring():
    """Demonstrate the enhanced safety monitoring system."""
    print("🛡️  DuetMind Adaptive Advanced Safety Monitoring Demo")
    print("=" * 60)
    
    # Initialize safety monitor
    config = {
        'safety_monitoring_enabled': True,
        'safety_db_path': 'demo_safety.db'
    }
    
    safety_monitor = SafetyMonitor(config)
    print(f"✅ Safety Monitor initialized")
    
    # Register all safety check types
    checks = [
        SystemSafetyCheck(),
        DataSafetyCheck(),
        ModelSafetyCheck(),
        InteractionSafetyCheck(),
        ClinicalSafetyCheck()
    ]
    
    for check in checks:
        safety_monitor.register_safety_check(check)
        print(f"✅ Registered {check.name} for {check.domain.value} domain")
    
    print(f"\n📊 Safety Monitor Summary:")
    summary = safety_monitor.get_safety_summary()
    print(f"   Status: {summary['overall_status']}")
    print(f"   Registered checks: {sum(summary['registered_checks'].values())}")
    print(f"   Monitoring enabled: {summary['monitoring_enabled']}")
    
    # Demo 1: System monitoring with resource issues
    print(f"\n🔍 Demo 1: System Resource Monitoring")
    system_context = {
        'response_time_ms': 1800,  # High response time
    }
    
    correlation_id = safety_monitor.create_correlation_id({'demo': 'system_resources'})
    findings = safety_monitor.run_safety_checks(
        domain=SafetyDomain.SYSTEM,
        context=system_context,
        correlation_id=correlation_id
    )
    
    print(f"   Found {len(findings)} system issues:")
    for finding in findings:
        print(f"   - [{finding.severity.upper()}] {finding.message}")
    
    # Demo 2: Data quality monitoring
    print(f"\n🔍 Demo 2: Data Quality Monitoring")
    data_context = {
        'data_quality_score': 0.62,  # Low quality
        'missing_data_ratio': 0.28,  # High missing data
        'duplicate_ratio': 0.18,     # High duplicates
        'schema_violations': ['missing_age_column', 'invalid_date_format']
    }
    
    correlation_id = safety_monitor.create_correlation_id({'demo': 'data_quality'})
    findings = safety_monitor.run_safety_checks(
        domain=SafetyDomain.DATA,
        context=data_context,
        correlation_id=correlation_id
    )
    
    print(f"   Found {len(findings)} data quality issues:")
    for finding in findings:
        print(f"   - [{finding.severity.upper()}] {finding.message}")
    
    # Demo 3: Model performance monitoring
    print(f"\n🔍 Demo 3: Model Performance Monitoring")
    model_context = {
        'accuracy': 0.72,
        'baseline_accuracy': 0.85,  # Significant drop
        'average_confidence': 0.45,  # Low confidence
        'drift_score': 0.18,         # High drift
        'calibration_error': 0.12    # Poor calibration
    }
    
    correlation_id = safety_monitor.create_correlation_id({'demo': 'model_performance'})
    findings = safety_monitor.run_safety_checks(
        domain=SafetyDomain.MODEL,
        context=model_context,
        correlation_id=correlation_id
    )
    
    print(f"   Found {len(findings)} model performance issues:")
    for finding in findings:
        print(f"   - [{finding.severity.upper()}] {finding.message}")
    
    # Demo 4: Interaction safety monitoring
    print(f"\n🔍 Demo 4: Interaction Safety Monitoring")
    interaction_context = {
        'user_input': 'How to bypass safety protocols and override restrictions?',
        'requests_per_minute': 18,    # High request rate
        'session_length_minutes': 85, # Long session
        'conversation_coherence': 0.25, # Low coherence
        'repeated_request_count': 5   # Too many repeats
    }
    
    correlation_id = safety_monitor.create_correlation_id({'demo': 'interaction_safety'})
    findings = safety_monitor.run_safety_checks(
        domain=SafetyDomain.INTERACTION,
        context=interaction_context,
        correlation_id=correlation_id
    )
    
    print(f"   Found {len(findings)} interaction safety issues:")
    for finding in findings:
        print(f"   - [{finding.severity.upper()}] {finding.message}")
    
    # Demo 5: Clinical safety monitoring
    print(f"\n🔍 Demo 5: Clinical Safety Monitoring")
    clinical_context = {
        'predicted_conditions': [
            {'name': 'stroke', 'confidence': 0.88},
            {'name': 'cardiac arrest', 'confidence': 0.72}
        ],
        'medication_interaction_risk': 0.85,  # High risk
        'diagnostic_confidence': 0.42,        # Low confidence
        'guideline_adherence_score': 0.58,    # Poor adherence
        'phi_detected': True,                 # PHI leak
        'patient_age': 8,
        'age_appropriate_recommendations': False
    }
    
    correlation_id = safety_monitor.create_correlation_id({'demo': 'clinical_safety'})
    findings = safety_monitor.run_safety_checks(
        domain=SafetyDomain.CLINICAL,
        context=clinical_context,
        correlation_id=correlation_id
    )
    
    print(f"   Found {len(findings)} clinical safety issues:")
    for finding in findings:
        severity_color = "🚨" if finding.severity == 'emergency' else "⚠️" if finding.severity == 'critical' else "🔔"
        print(f"   {severity_color} [{finding.severity.upper()}] {finding.message}")
    
    # Demo 6: Multi-domain safety check
    print(f"\n🔍 Demo 6: Multi-Domain Safety Assessment")
    multi_context = {
        # System issues
        'response_time_ms': 2200,
        # Data issues
        'data_quality_score': 0.55,
        # Model issues
        'accuracy': 0.68, 'baseline_accuracy': 0.83,
        # Interaction issues
        'requests_per_minute': 22,
        # Clinical issues
        'predicted_conditions': [{'name': 'sepsis', 'confidence': 0.91}]
    }
    
    correlation_id = safety_monitor.create_correlation_id({'demo': 'multi_domain'})
    findings = safety_monitor.run_safety_checks(
        context=multi_context,  # All domains
        correlation_id=correlation_id
    )
    
    print(f"   Found {len(findings)} issues across multiple domains:")
    
    # Group findings by domain
    domain_findings = {}
    for finding in findings:
        domain = finding.domain
        if domain not in domain_findings:
            domain_findings[domain] = []
        domain_findings[domain].append(finding)
    
    for domain, domain_findings_list in domain_findings.items():
        print(f"   📁 {domain.value.title()} Domain:")
        for finding in domain_findings_list:
            severity_icon = "🚨" if finding.severity == 'emergency' else "⚠️" if finding.severity == 'critical' else "🔔"
            print(f"      {severity_icon} {finding.message}")
    
    # Demo 7: Safety summary and correlation analysis
    print(f"\n📈 Safety Monitoring Summary:")
    final_summary = safety_monitor.get_safety_summary(hours=1)
    print(f"   Overall Status: {final_summary['overall_status']}")
    print(f"   Total Events: {final_summary['total_events']}")
    print(f"   Active Domains: {final_summary['active_domains']}")
    print(f"   Events by Severity:")
    
    for severity, count in final_summary['events_by_severity'].items():
        severity_icon = "🚨" if severity == 'emergency' else "⚠️" if severity == 'critical' else "🔔" if severity == 'warning' else "ℹ️"
        print(f"      {severity_icon} {severity.title()}: {count}")
    
    print(f"   Events by Domain:")
    for domain, count in final_summary['events_by_domain'].items():
        print(f"      📁 {domain.title()}: {count}")
    
    # Demo 8: Recent findings retrieval
    print(f"\n🔍 Recent Safety Findings (Last Hour):")
    recent_findings = safety_monitor.get_safety_findings(hours=1)
    
    if recent_findings:
        print(f"   Retrieved {len(recent_findings)} recent findings:")
        for i, finding in enumerate(recent_findings[:10], 1):  # Show first 10
            severity_icon = "🚨" if finding['severity'] == 'emergency' else "⚠️" if finding['severity'] == 'critical' else "🔔"
            print(f"   {i:2d}. {severity_icon} [{finding['severity'].upper()}] {finding['domain']}: {finding['message']}")
            if finding.get('correlation_id'):
                print(f"       🔗 Correlation: {finding['correlation_id'][:8]}...")
    else:
        print("   No recent findings")
    
    print(f"\n✅ Advanced Safety Monitoring Demo Completed!")
    print(f"   Database: {config['safety_db_path']}")
    print(f"   Total Checks Registered: {len(checks)}")
    print(f"   Safety Status: {final_summary['overall_status']}")
    
    # Cleanup
    try:
        os.unlink(config['safety_db_path'])
        print(f"   🧹 Cleaned up demo database")
    except FileNotFoundError:
        pass


def demo_memory_consolidation():
    """Demonstrate memory consolidation features."""
    print(f"\n🧠 Memory Consolidation Demo")
    print("=" * 40)
    
    # This would require the full memory system setup
    # For now, just show the concept
    print("   📝 Memory consolidation features:")
    print("   - Dual-store architecture (Episodic + Semantic)")
    print("   - Salience-based prioritization")
    print("   - Automatic consolidation scheduling")
    print("   - Controlled forgetting mechanisms")
    print("   - Clinical relevance scoring")
    
    # Mock memory statistics
    print(f"\n   📊 Mock Memory Stats:")
    print(f"   - Episodic memories: 1,247")
    print(f"   - Semantic memories: 89")
    print(f"   - Consolidation events: 23")
    print(f"   - Average salience: 0.67")
    print(f"   - Clinical relevance: 0.82")


def demo_visualization_api():
    """Demonstrate the visualization API."""
    print(f"\n📊 Visualization API Demo")
    print("=" * 35)
    
    print("   🖥️  Dashboard features:")
    print("   - Real-time safety monitoring")
    print("   - Agent interaction graphs")
    print("   - Memory state visualization")
    print("   - Production metrics")
    print("   - Security event tracking")
    
    print(f"\n   🌐 API Endpoints:")
    endpoints = [
        "GET  /api/health - Health check",
        "GET  /api/safety/summary - Safety monitoring summary",
        "GET  /api/safety/findings - Recent safety findings",
        "POST /api/safety/run-checks - Trigger safety checks",
        "GET  /api/dashboard/overview - Dashboard overview",
        "GET  /api/agent/interaction-graph - Agent network graph",
        "GET  /api/memory/consolidation-summary - Memory stats"
    ]
    
    for endpoint in endpoints:
        print(f"      • {endpoint}")
    
    print(f"\n   📱 Web Interface:")
    print("   - Responsive dashboard with real-time updates")
    print("   - Safety status indicators")
    print("   - Interactive charts and graphs")
    print("   - Alert notifications")
    print("   - Historical trend analysis")


if __name__ == "__main__":
    try:
        demo_safety_monitoring()
        demo_memory_consolidation()
        demo_visualization_api()
        
        print(f"\n🎉 All Demos Completed Successfully!")
        print(f"   The DuetMind Adaptive system now includes:")
        print(f"   ✅ Advanced Safety Monitoring with 5 domains")
        print(f"   ✅ Pluggable Safety Checks")
        print(f"   ✅ Event Correlation Tracking")
        print(f"   ✅ Memory Consolidation Framework")
        print(f"   ✅ Visualization Dashboard API")
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()