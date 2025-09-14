#!/usr/bin/env python3
"""
Simple Integration Demo: Advanced Safety Monitoring with Existing Systems

Shows how the new safety monitoring integrates with existing DuetMind components
without complex dependencies.
"""

import sys
import os
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from security.safety_monitor import SafetyMonitor, SafetyDomain
from security.safety_checks import SystemSafetyCheck, ClinicalSafetyCheck, DataSafetyCheck
from security.monitoring import SecurityMonitor


class SimpleMedicalAISystem:
    """Simplified medical AI system with integrated safety monitoring."""
    
    def __init__(self):
        # Initialize safety monitoring
        config = {
            'safety_monitoring_enabled': True,
            'security_monitoring_enabled': True,
            'safety_db_path': 'simple_demo_safety.db'
        }
        
        self.safety_monitor = SafetyMonitor(config)
        self.security_monitor = SecurityMonitor(config)
        
        # Setup safety checks
        self.safety_monitor.register_safety_check(SystemSafetyCheck())
        self.safety_monitor.register_safety_check(ClinicalSafetyCheck())
        self.safety_monitor.register_safety_check(DataSafetyCheck())
        
        print("🏥 Simple Medical AI System initialized")
        print(f"   ✅ Safety monitoring enabled")
        print(f"   ✅ Security monitoring enabled")
        print(f"   ✅ {len(self.safety_monitor.safety_checks)} safety check domains registered")
    
    def process_patient(self, patient_data):
        """Process a patient with safety monitoring."""
        print(f"\n👤 Processing Patient: {patient_data['name']}")
        
        # Create correlation ID for tracking
        correlation_id = self.safety_monitor.create_correlation_id({
            'patient_id': patient_data.get('id'),
            'session_type': 'patient_assessment'
        })
        
        # Simulate medical AI processing with safety context
        safety_context = {
            'response_time_ms': 420,  # Simulated processing time
            'data_quality_score': patient_data.get('data_quality', 0.9),
            'predicted_conditions': self._simulate_diagnosis(patient_data),
            'diagnostic_confidence': patient_data.get('confidence', 0.8),
            'medication_interaction_risk': patient_data.get('med_risk', 0.1),
            'phi_detected': 'SSN' in str(patient_data),  # Simple PHI check
            'patient_age': patient_data.get('age', 50),
            'guideline_adherence_score': 0.9
        }
        
        # Run safety checks
        findings = self.safety_monitor.run_safety_checks(
            context=safety_context,
            correlation_id=correlation_id
        )
        
        # Log security event
        self.security_monitor.log_api_request(
            user_id=patient_data.get('clinician', 'unknown'),
            endpoint='patient_assessment',
            method='POST',
            status_code=200,
            response_time=0.42,
            payload_size=len(str(patient_data))
        )
        
        # Process results
        critical_findings = [f for f in findings if f.severity in ['critical', 'emergency']]
        
        result = {
            'patient_id': patient_data.get('id'),
            'diagnosis': safety_context['predicted_conditions'],
            'confidence': safety_context['diagnostic_confidence'],
            'safety_status': self._determine_safety_status(findings),
            'critical_findings': len(critical_findings),
            'requires_review': len(critical_findings) > 0,
            'correlation_id': correlation_id
        }
        
        # Report findings
        if critical_findings:
            print(f"   ⚠️  {len(critical_findings)} critical safety findings detected:")
            for finding in critical_findings[:3]:  # Show first 3
                severity_icon = "🚨" if finding.severity == 'emergency' else "⚠️"
                print(f"      {severity_icon} {finding.message}")
        else:
            print(f"   ✅ No critical safety issues detected")
        
        print(f"   📊 Safety Status: {result['safety_status']}")
        print(f"   🔍 Correlation ID: {correlation_id[:8]}...")
        
        return result
    
    def _simulate_diagnosis(self, patient_data):
        """Simulate medical diagnosis based on patient data."""
        conditions = []
        
        # Simple rule-based diagnosis simulation
        if patient_data.get('age', 0) > 65 and patient_data.get('symptoms'):
            if 'chest_pain' in patient_data['symptoms']:
                conditions.append({'name': 'cardiac assessment needed', 'confidence': 0.7})
            if 'memory_loss' in patient_data['symptoms']:
                conditions.append({'name': 'mild cognitive impairment', 'confidence': 0.6})
            if 'severe_headache' in patient_data['symptoms']:
                conditions.append({'name': 'stroke', 'confidence': 0.85})  # Critical condition
        
        return conditions
    
    def _determine_safety_status(self, findings):
        """Determine overall safety status from findings."""
        if any(f.severity == 'emergency' for f in findings):
            return 'EMERGENCY'
        elif any(f.severity == 'critical' for f in findings):
            return 'CRITICAL'
        elif any(f.severity == 'warning' for f in findings):
            return 'WARNING'
        else:
            return 'SAFE'
    
    def get_system_status(self):
        """Get overall system status."""
        safety_summary = self.safety_monitor.get_safety_summary(hours=1)
        security_summary = self.security_monitor.get_security_summary()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'safety': {
                'status': safety_summary['overall_status'],
                'total_events': safety_summary['total_events'],
                'critical_events': safety_summary['events_by_severity'].get('critical', 0),
                'emergency_events': safety_summary['events_by_severity'].get('emergency', 0)
            },
            'security': {
                'status': security_summary['monitoring_status'],
                'total_events': security_summary['total_security_events']
            }
        }


def run_demo():
    """Run the simple integration demo."""
    print("🏥 Simple Medical AI Safety Integration Demo")
    print("=" * 50)
    
    # Initialize system
    system = SimpleMedicalAISystem()
    
    # Test patients with different risk profiles
    test_patients = [
        {
            'id': 'P001',
            'name': 'John Smith',
            'age': 45,
            'symptoms': ['headache', 'fatigue'],
            'data_quality': 0.95,
            'confidence': 0.8,
            'med_risk': 0.1,
            'clinician': 'DR_JONES'
        },
        {
            'id': 'P002',
            'name': 'Mary Johnson',
            'age': 72,
            'symptoms': ['chest_pain', 'shortness_of_breath'],
            'data_quality': 0.87,
            'confidence': 0.75,
            'med_risk': 0.3,
            'clinician': 'DR_SMITH'
        },
        {
            'id': 'P003',
            'name': 'Robert Wilson',
            'age': 68,
            'symptoms': ['severe_headache', 'vision_changes'],
            'data_quality': 0.6,  # Low data quality
            'confidence': 0.4,    # Low confidence
            'med_risk': 0.8,      # High medication risk
            'clinician': 'DR_BROWN'
        }
    ]
    
    # Process patients
    results = []
    for patient in test_patients:
        result = system.process_patient(patient)
        results.append(result)
    
    # System status summary
    print(f"\n📊 System Status Summary:")
    status = system.get_system_status()
    print(f"   Safety Status: {status['safety']['status']}")
    print(f"   Total Safety Events: {status['safety']['total_events']}")
    print(f"   Critical Events: {status['safety']['critical_events']}")
    print(f"   Emergency Events: {status['safety']['emergency_events']}")
    print(f"   Security Events: {status['security']['total_events']}")
    
    # Results summary
    print(f"\n📈 Processing Results:")
    safe_patients = [r for r in results if r['safety_status'] == 'SAFE']
    critical_patients = [r for r in results if r['safety_status'] in ['CRITICAL', 'EMERGENCY']]
    
    print(f"   Patients Processed: {len(results)}")
    print(f"   Safe Assessments: {len(safe_patients)}")
    print(f"   Critical/Emergency: {len(critical_patients)}")
    print(f"   Requiring Review: {sum(1 for r in results if r['requires_review'])}")
    
    # Show recent safety findings
    print(f"\n🔍 Recent Safety Findings:")
    recent_findings = system.safety_monitor.get_safety_findings(hours=1)
    
    if recent_findings:
        print(f"   Found {len(recent_findings)} recent findings:")
        for i, finding in enumerate(recent_findings[:5], 1):  # Show first 5
            severity_icon = "🚨" if finding['severity'] == 'emergency' else "⚠️" if finding['severity'] == 'critical' else "🔔"
            print(f"   {i}. {severity_icon} [{finding['severity'].upper()}] {finding['domain']}: {finding['message']}")
    else:
        print("   No recent findings")
    
    # Demonstrate correlation tracking
    if results:
        sample_correlation = results[0]['correlation_id']
        chain = system.safety_monitor.get_correlation_chain(sample_correlation)
        print(f"\n🔗 Correlation Chain Example (ID: {sample_correlation[:8]}...):")
        print(f"   Events in chain: {len(chain)}")
        for event in chain:
            print(f"   • {event.domain.value}: {event.message}")
    
    print(f"\n✅ Demo Completed Successfully!")
    print(f"   The system demonstrates:")
    print(f"   • Real-time safety monitoring across multiple domains")
    print(f"   • Integration with existing security monitoring")
    print(f"   • Event correlation for audit trails")
    print(f"   • Automated risk assessment and alerting")
    print(f"   • Medical AI specific safety checks")
    
    # Cleanup
    try:
        os.unlink('simple_demo_safety.db')
        print(f"   🧹 Cleaned up demo database")
    except FileNotFoundError:
        pass


if __name__ == "__main__":
    try:
        run_demo()
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()