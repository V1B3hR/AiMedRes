#!/usr/bin/env python3
"""
Integration Example: DuetMind Adaptive Advanced Safety Monitoring

Shows how to integrate the new safety monitoring features with existing
DuetMind Adaptive components for a complete medical AI safety system.
"""

import sys
import os
from datetime import datetime
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from security.safety_monitor import SafetyMonitor, SafetyDomain
from security.safety_checks import SystemSafetyCheck, ClinicalSafetyCheck
from security.monitoring import SecurityMonitor
from mlops.monitoring.production_monitor import ProductionMonitor
from api.visualization_api import VisualizationAPI


class IntegratedMedicalAISafetySystem:
    """
    Integrated safety system combining existing and new monitoring capabilities.
    
    Demonstrates how to integrate:
    - Advanced Safety Monitoring (new)
    - Security Monitoring (existing)
    - Production Monitoring (existing)
    - Visualization Dashboard (new)
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize existing monitoring systems
        self.security_monitor = SecurityMonitor(config)
        self.production_monitor = ProductionMonitor(
            model_name=config.get('model_name', 'alzheimer_classifier'),
            mlflow_tracking_uri=config.get('mlflow_uri', 'sqlite:///mlflow.db')
        )
        
        # Initialize new advanced safety monitoring
        self.safety_monitor = SafetyMonitor(config)
        
        # Register safety checks for medical AI
        self._setup_medical_safety_checks()
        
        # Initialize visualization API
        self.viz_api = VisualizationAPI(config)
        self.viz_api.initialize_monitors(
            safety_monitor=self.safety_monitor,
            security_monitor=self.security_monitor,
            production_monitor=self.production_monitor
        )
        
        print("🏥 Integrated Medical AI Safety System initialized")
    
    def _setup_medical_safety_checks(self):
        """Setup safety checks specific to medical AI applications."""
        
        # System monitoring for performance and resources
        system_config = {
            'cpu_critical_threshold': 85.0,      # Lower threshold for medical systems
            'memory_critical_threshold': 85.0,    # Lower threshold for medical systems
            'response_time_warning_ms': 500.0,    # Faster response required
            'response_time_critical_ms': 1000.0   # Stricter timing for medical AI
        }
        self.safety_monitor.register_safety_check(SystemSafetyCheck(system_config))
        
        # Clinical safety monitoring
        clinical_config = {
            'critical_conditions': [
                'stroke', 'heart attack', 'cardiac arrest', 'severe bleeding',
                'anaphylaxis', 'respiratory failure', 'sepsis', 'diabetic coma',
                'acute myocardial infarction', 'pulmonary embolism'
            ],
            'medication_interaction_threshold': 0.7,  # Stricter for safety
            'diagnostic_confidence_threshold': 0.8    # Higher confidence required
        }
        self.safety_monitor.register_safety_check(ClinicalSafetyCheck(clinical_config))
        
        print("✅ Medical AI safety checks configured")
    
    def process_patient_assessment(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a patient assessment with integrated safety monitoring.
        
        Args:
            patient_data: Patient information and symptoms
            
        Returns:
            Assessment results with safety validation
        """
        print(f"\n🏥 Processing Patient Assessment")
        print(f"   Patient ID: {patient_data.get('patient_id', 'Unknown')}")
        
        # Create correlation ID for tracking this assessment
        correlation_id = self.safety_monitor.create_correlation_id({
            'type': 'patient_assessment',
            'patient_id': patient_data.get('patient_id'),
            'timestamp': datetime.now().isoformat()
        })
        
        # Simulate model prediction (in real system, this would be actual ML inference)
        prediction_result = self._simulate_medical_prediction(patient_data)
        
        # Prepare safety check context
        safety_context = {
            'response_time_ms': 350,  # Simulated response time
            'predicted_conditions': prediction_result.get('conditions', []),
            'diagnostic_confidence': prediction_result.get('confidence', 0.0),
            'medication_interaction_risk': patient_data.get('medication_risk', 0.0),
            'guideline_adherence_score': prediction_result.get('guideline_adherence', 0.9),
            'phi_detected': self._check_for_phi(prediction_result.get('explanation', '')),
            'patient_age': patient_data.get('age', 50),
            'age_appropriate_recommendations': self._check_age_appropriateness(
                patient_data.get('age', 50), 
                prediction_result.get('recommendations', [])
            )
        }
        
        # Run safety checks
        safety_findings = self.safety_monitor.run_safety_checks(
            context=safety_context,
            correlation_id=correlation_id
        )
        
        # Log to security monitor
        self.security_monitor.log_api_request(
            user_id=patient_data.get('clinician_id', 'unknown'),
            endpoint='patient_assessment',
            method='POST',
            status_code=200,
            response_time=0.35,
            payload_size=len(str(patient_data))
        )
        
        # Determine if assessment is safe to return
        critical_findings = [f for f in safety_findings if f.severity in ['critical', 'emergency']]
        
        assessment_result = {
            'correlation_id': correlation_id,
            'prediction': prediction_result,
            'safety_status': 'EMERGENCY' if any(f.severity == 'emergency' for f in safety_findings) 
                           else 'CRITICAL' if critical_findings
                           else 'SAFE',
            'safety_findings': [
                {
                    'domain': f.domain.value,
                    'severity': f.severity,
                    'message': f.message,
                    'recommendation': f.metadata.get('recommendation', 'monitor')
                } for f in safety_findings
            ],
            'requires_human_review': len(critical_findings) > 0,
            'timestamp': datetime.now().isoformat()
        }
        
        # Handle critical findings
        if critical_findings:
            print(f"⚠️  SAFETY ALERT: {len(critical_findings)} critical findings detected")
            for finding in critical_findings[:3]:  # Show first 3
                severity_icon = "🚨" if finding.severity == 'emergency' else "⚠️"
                print(f"   {severity_icon} {finding.message}")
            
            if any(f.severity == 'emergency' for f in critical_findings):
                print(f"🚨 EMERGENCY: Immediate medical attention recommended")
                assessment_result['immediate_action_required'] = True
        
        print(f"✅ Assessment completed - Status: {assessment_result['safety_status']}")
        return assessment_result
    
    def _simulate_medical_prediction(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate medical AI prediction (replace with actual model in production)."""
        age = patient_data.get('age', 50)
        symptoms = patient_data.get('symptoms', [])
        
        # Simple rule-based simulation
        conditions = []
        confidence = 0.75
        
        if age > 65 and 'chest_pain' in symptoms:
            conditions.append({'name': 'cardiac assessment needed', 'confidence': 0.8})
        
        if 'cognitive_decline' in symptoms:
            conditions.append({'name': 'mild cognitive impairment', 'confidence': 0.7})
        
        if 'severe_headache' in symptoms and 'vision_changes' in symptoms:
            conditions.append({'name': 'stroke', 'confidence': 0.85})
            confidence = 0.85
        
        return {
            'conditions': conditions,
            'confidence': confidence,
            'recommendations': ['clinical_evaluation', 'follow_up_in_2_weeks'],
            'explanation': f"Assessment based on age {age} and symptoms: {', '.join(symptoms)}",
            'guideline_adherence': 0.88
        }
    
    def _check_for_phi(self, text: str) -> bool:
        """Check if text contains Protected Health Information."""
        phi_patterns = ['SSN', 'social security', 'phone:', 'address:', 'DOB:', 'medical record']
        return any(pattern.lower() in text.lower() for pattern in phi_patterns)
    
    def _check_age_appropriateness(self, age: int, recommendations: list) -> bool:
        """Check if recommendations are age-appropriate."""
        if age < 18:
            # Pediatric checks
            adult_only_recommendations = ['colonoscopy', 'mammogram', 'prostate_exam']
            return not any(rec in str(recommendations).lower() for rec in adult_only_recommendations)
        return True
    
    def get_system_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive system dashboard data."""
        
        # Get data from all monitoring systems
        safety_summary = self.safety_monitor.get_safety_summary(hours=24)
        security_summary = self.security_monitor.get_security_summary()
        production_summary = self.production_monitor.get_monitoring_summary(hours=24)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'system_status': safety_summary.get('overall_status', 'UNKNOWN'),
            'safety_monitoring': {
                'status': safety_summary.get('overall_status'),
                'total_events': safety_summary.get('total_events', 0),
                'critical_events': safety_summary.get('events_by_severity', {}).get('critical', 0),
                'emergency_events': safety_summary.get('events_by_severity', {}).get('emergency', 0),
                'active_domains': safety_summary.get('active_domains', 0)
            },
            'security_monitoring': {
                'status': security_summary.get('monitoring_status', 'inactive'),
                'total_events': security_summary.get('total_security_events', 0),
                'api_requests': security_summary.get('api_usage', {}).get('total_requests', 0)
            },
            'production_monitoring': {
                'status': production_summary.get('status', 'unknown'),
                'predictions': production_summary.get('total_predictions', 0),
                'accuracy': production_summary.get('avg_accuracy', 0.0),
                'alerts': len(production_summary.get('alerts', {}))
            },
            'recommendations': self._generate_system_recommendations(
                safety_summary, security_summary, production_summary
            )
        }
    
    def _generate_system_recommendations(self, safety_summary, security_summary, production_summary) -> list:
        """Generate system recommendations based on monitoring data."""
        recommendations = []
        
        # Safety recommendations
        if safety_summary.get('overall_status') == 'EMERGENCY':
            recommendations.append("🚨 URGENT: Review emergency safety findings immediately")
        elif safety_summary.get('overall_status') == 'CRITICAL':
            recommendations.append("⚠️ Review critical safety findings")
        
        # Production recommendations
        if production_summary.get('avg_accuracy', 1.0) < 0.8:
            recommendations.append("📊 Model accuracy below threshold - consider retraining")
        
        # Security recommendations
        if security_summary.get('total_security_events', 0) > 10:
            recommendations.append("🔒 High security event count - review access patterns")
        
        if not recommendations:
            recommendations.append("✅ All systems operating within normal parameters")
        
        return recommendations
    
    def start_monitoring(self):
        """Start all monitoring systems."""
        self.security_monitor.start_monitoring()
        self.safety_monitor.start_monitoring(interval_seconds=30)  # More frequent for medical AI
        self.production_monitor.start_monitoring(interval_seconds=300)
        print("🔄 All monitoring systems started")
    
    def stop_monitoring(self):
        """Stop all monitoring systems."""
        self.security_monitor.stop_monitoring()
        self.safety_monitor.stop_monitoring()
        self.production_monitor.stop_monitoring()
        print("⏹️ All monitoring systems stopped")


def demo_integrated_system():
    """Demonstrate the integrated medical AI safety system."""
    print("🏥 Medical AI Safety System Integration Demo")
    print("=" * 50)
    
    # System configuration
    config = {
        'safety_monitoring_enabled': True,
        'security_monitoring_enabled': True,
        'model_name': 'medical_ai_classifier',
        'safety_db_path': 'medical_safety.db'
    }
    
    # Initialize integrated system
    safety_system = IntegratedMedicalAISafetySystem(config)
    
    # Demo patient assessments
    test_patients = [
        {
            'patient_id': 'P001',
            'age': 72,
            'symptoms': ['chest_pain', 'shortness_of_breath'],
            'medication_risk': 0.3,
            'clinician_id': 'DR001'
        },
        {
            'patient_id': 'P002', 
            'age': 45,
            'symptoms': ['cognitive_decline', 'memory_issues'],
            'medication_risk': 0.1,
            'clinician_id': 'DR002'
        },
        {
            'patient_id': 'P003',
            'age': 38,
            'symptoms': ['severe_headache', 'vision_changes'],
            'medication_risk': 0.85,  # High medication interaction risk
            'clinician_id': 'DR001'
        }
    ]
    
    results = []
    for patient in test_patients:
        result = safety_system.process_patient_assessment(patient)
        results.append(result)
    
    # Show system dashboard
    print(f"\n📊 System Dashboard:")
    dashboard = safety_system.get_system_dashboard()
    print(f"   Overall Status: {dashboard['system_status']}")
    print(f"   Safety Events: {dashboard['safety_monitoring']['total_events']}")
    print(f"   Critical Events: {dashboard['safety_monitoring']['critical_events']}")
    print(f"   Emergency Events: {dashboard['safety_monitoring']['emergency_events']}")
    print(f"   Security Events: {dashboard['security_monitoring']['total_events']}")
    
    print(f"\n💡 System Recommendations:")
    for rec in dashboard['recommendations']:
        print(f"   • {rec}")
    
    # Summary
    safe_assessments = [r for r in results if r['safety_status'] == 'SAFE']
    critical_assessments = [r for r in results if r['safety_status'] in ['CRITICAL', 'EMERGENCY']]
    
    print(f"\n📈 Assessment Summary:")
    print(f"   Total Assessments: {len(results)}")
    print(f"   Safe Assessments: {len(safe_assessments)}")
    print(f"   Critical/Emergency: {len(critical_assessments)}")
    print(f"   Human Review Required: {sum(1 for r in results if r.get('requires_human_review'))}")
    
    print(f"\n✅ Integration Demo Completed Successfully!")
    
    # Cleanup
    try:
        os.unlink(config['safety_db_path'])
        print(f"🧹 Cleaned up demo database")
    except FileNotFoundError:
        pass


if __name__ == "__main__":
    demo_integrated_system()