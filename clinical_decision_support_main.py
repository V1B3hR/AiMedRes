#!/usr/bin/env python3
"""
Clinical Decision Support System - Main Integration Module

This is the main orchestrator that integrates all Clinical Decision Support components:
- Risk stratification engine
- Explainable AI dashboard
- EHR integration
- Regulatory compliance
- Web-based clinical interface

Provides a unified interface for clinicians to interact with the AI system.
"""

import json
import logging
import asyncio
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from pathlib import Path
import uuid
from dataclasses import asdict

# Import all CDSS components
from clinical_decision_support import ClinicalDecisionSupportSystem, RiskAssessment
from explainable_ai_dashboard import DashboardGenerator
from ehr_integration import EHRConnector
from regulatory_compliance import (
    ComplianceDashboard, HIPAAComplianceManager, FDAValidationManager,
    AuditEvent, AuditEventType
)
from specialized_medical_agents import MedicalKnowledgeAgent

# Flask components for web interface
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS

logger = logging.getLogger("ClinicalDecisionSupportMain")


class ClinicalWorkflowOrchestrator:
    """
    Main orchestrator for clinical decision support workflows.
    
    Integrates all components to provide seamless clinical decision support.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize all components
        self.cdss = ClinicalDecisionSupportSystem(config)
        self.dashboard_generator = DashboardGenerator(config)
        self.ehr_connector = EHRConnector(config)
        self.compliance_dashboard = ComplianceDashboard(config)
        self.hipaa_manager = HIPAAComplianceManager(config)
        
        # Workflow tracking
        self.active_sessions = {}
        self.workflow_history = []
        
        logger.info("Clinical Decision Support System initialized")
    
    def process_patient_workflow(self, patient_data: Dict[str, Any], 
                               user_id: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process complete clinical decision support workflow for a patient.
        
        Args:
            patient_data: Patient clinical data
            user_id: Clinician user ID
            session_id: Optional session ID for tracking
            
        Returns:
            Complete workflow results including assessments, dashboard, and recommendations
        """
        
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        workflow_start = datetime.now()
        
        try:
            # 1. Log PHI access for HIPAA compliance
            phi_elements = list(patient_data.keys())
            self.hipaa_manager.log_phi_access(
                user_id=user_id,
                patient_id=patient_data.get('patient_id', 'unknown'),
                data_elements=phi_elements,
                purpose='clinical_decision_support'
            )
            
            # 2. Apply data minimization
            minimized_data = self.hipaa_manager.check_data_minimization(
                patient_data, 'risk_assessment'
            )
            
            # 3. Perform comprehensive risk assessment
            assessments = self.cdss.comprehensive_assessment(minimized_data)
            
            # 4. Generate explainable AI dashboard
            dashboard_data = self.dashboard_generator.generate_patient_dashboard(
                minimized_data, assessments
            )
            
            # 5. Generate clinical summary
            clinical_summary = self.cdss.generate_clinical_summary(assessments)
            
            # 6. Log audit event
            audit_event = AuditEvent(
                event_id=str(uuid.uuid4()),
                event_type=AuditEventType.MODEL_PREDICTION,
                timestamp=datetime.now(),
                user_id=user_id,
                patient_id=patient_data.get('patient_id'),
                resource_accessed='clinical_decision_support_system',
                action_performed='comprehensive_assessment',
                outcome='SUCCESS',
                ip_address=request.remote_addr if request else None,
                user_agent=request.headers.get('User-Agent') if request else None,
                additional_data={
                    'session_id': session_id,
                    'conditions_assessed': list(assessments.keys()),
                    'processing_time_ms': (datetime.now() - workflow_start).total_seconds() * 1000
                }
            )
            
            self.hipaa_manager.log_audit_event(audit_event)
            
            # 7. Prepare workflow results
            workflow_results = {
                'session_id': session_id,
                'patient_id': patient_data.get('patient_id'),
                'timestamp': datetime.now().isoformat(),
                'user_id': user_id,
                'assessments': {
                    condition: asdict(assessment) if hasattr(assessment, '__dict__') else assessment
                    for condition, assessment in assessments.items()
                },
                'clinical_summary': clinical_summary,
                'dashboard': dashboard_data,
                'processing_time_ms': (datetime.now() - workflow_start).total_seconds() * 1000,
                'compliance_status': 'COMPLIANT',
                'next_steps': self._generate_next_steps(assessments, clinical_summary)
            }
            
            # Store session
            self.active_sessions[session_id] = workflow_results
            self.workflow_history.append(workflow_results)
            
            return workflow_results
            
        except Exception as e:
            # Log error event
            error_event = AuditEvent(
                event_id=str(uuid.uuid4()),
                event_type=AuditEventType.MODEL_PREDICTION,
                timestamp=datetime.now(),
                user_id=user_id,
                patient_id=patient_data.get('patient_id'),
                resource_accessed='clinical_decision_support_system',
                action_performed='comprehensive_assessment',
                outcome='FAILURE',
                ip_address=request.remote_addr if request else None,
                user_agent=request.headers.get('User-Agent') if request else None,
                additional_data={
                    'session_id': session_id,
                    'error': str(e),
                    'processing_time_ms': (datetime.now() - workflow_start).total_seconds() * 1000
                },
                risk_level='HIGH'
            )
            
            self.hipaa_manager.log_audit_event(error_event)
            
            logger.error(f"Workflow processing failed: {e}")
            raise
    
    def export_to_ehr(self, session_id: str, ehr_endpoint: str, 
                     format_type: str = 'fhir') -> Dict[str, Any]:
        """
        Export assessment results to EHR system.
        
        Args:
            session_id: Workflow session ID
            ehr_endpoint: EHR system endpoint
            format_type: Export format ('fhir' or 'hl7')
            
        Returns:
            Export status and results
        """
        
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        workflow_results = self.active_sessions[session_id]
        patient_id = workflow_results['patient_id']
        assessments = workflow_results['assessments']
        
        try:
            # Export to EHR
            export_result = self.ehr_connector.export_assessment_results(
                assessments, patient_id, format_type
            )
            
            # Log export event
            audit_event = AuditEvent(
                event_id=str(uuid.uuid4()),
                event_type=AuditEventType.DATA_EXPORT,
                timestamp=datetime.now(),
                user_id=workflow_results['user_id'],
                patient_id=patient_id,
                resource_accessed='ehr_system',
                action_performed='export_assessment_results',
                outcome='SUCCESS',
                ip_address=None,
                user_agent=None,
                additional_data={
                    'session_id': session_id,
                    'ehr_endpoint': ehr_endpoint,
                    'format_type': format_type,
                    'exported_assessments': list(assessments.keys())
                }
            )
            
            self.hipaa_manager.log_audit_event(audit_event)
            
            return {
                'status': 'SUCCESS',
                'session_id': session_id,
                'export_format': format_type,
                'ehr_endpoint': ehr_endpoint,
                'exported_data': export_result,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"EHR export failed: {e}")
            raise
    
    def get_compliance_status(self) -> Dict[str, Any]:
        """Get current regulatory compliance status"""
        return self.compliance_dashboard.generate_compliance_dashboard()
    
    def _generate_next_steps(self, assessments: Dict[str, Any], 
                           clinical_summary: Dict[str, Any]) -> List[str]:
        """Generate recommended next steps for clinician"""
        
        next_steps = []
        
        # High-risk conditions require immediate action
        high_risk_conditions = clinical_summary.get('high_risk_conditions', [])
        if high_risk_conditions:
            next_steps.append(f"Immediate specialist referral for: {', '.join(high_risk_conditions)}")
            next_steps.append("Schedule follow-up within 1-2 weeks")
        
        # Priority interventions
        priority_interventions = clinical_summary.get('priority_interventions', [])[:3]
        if priority_interventions:
            next_steps.append(f"Implement priority interventions: {', '.join(priority_interventions)}")
        
        # Next assessment scheduling
        next_assessment_date = clinical_summary.get('next_assessment_date')
        if next_assessment_date:
            next_steps.append(f"Schedule next assessment for: {next_assessment_date}")
        
        # Documentation requirements
        next_steps.append("Document clinical decision and rationale in patient record")
        next_steps.append("Provide patient education materials for identified risk factors")
        
        return next_steps


class ClinicalWebInterface:
    """
    Web-based interface for the Clinical Decision Support System.
    """
    
    def __init__(self, orchestrator: ClinicalWorkflowOrchestrator):
        self.orchestrator = orchestrator
        self.app = Flask(__name__)
        CORS(self.app)
        
        # Setup routes
        self._setup_routes()
        
    def _setup_routes(self):
        """Setup Flask routes for the web interface"""
        
        @self.app.route('/')
        def index():
            """Main dashboard page"""
            return render_template_string(self._get_dashboard_template())
        
        @self.app.route('/api/assess', methods=['POST'])
        def assess_patient():
            """Process patient assessment"""
            try:
                data = request.get_json()
                patient_data = data.get('patient_data', {})
                user_id = data.get('user_id', 'anonymous')
                
                # Process workflow
                results = self.orchestrator.process_patient_workflow(
                    patient_data, user_id
                )
                
                return jsonify({
                    'success': True,
                    'results': results
                })
                
            except Exception as e:
                logger.error(f"Assessment API error: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @self.app.route('/api/export', methods=['POST'])
        def export_to_ehr():
            """Export results to EHR"""
            try:
                data = request.get_json()
                session_id = data.get('session_id')
                ehr_endpoint = data.get('ehr_endpoint', 'localhost')
                format_type = data.get('format', 'fhir')
                
                result = self.orchestrator.export_to_ehr(
                    session_id, ehr_endpoint, format_type
                )
                
                return jsonify({
                    'success': True,
                    'export_result': result
                })
                
            except Exception as e:
                logger.error(f"Export API error: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @self.app.route('/api/compliance')
        def get_compliance():
            """Get compliance status"""
            try:
                compliance_status = self.orchestrator.get_compliance_status()
                return jsonify({
                    'success': True,
                    'compliance': compliance_status
                })
                
            except Exception as e:
                logger.error(f"Compliance API error: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @self.app.route('/api/sessions/<session_id>')
        def get_session(session_id):
            """Get session results"""
            try:
                if session_id in self.orchestrator.active_sessions:
                    session_data = self.orchestrator.active_sessions[session_id]
                    return jsonify({
                        'success': True,
                        'session': session_data
                    })
                else:
                    return jsonify({
                        'success': False,
                        'error': 'Session not found'
                    }), 404
                    
            except Exception as e:
                logger.error(f"Session API error: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
    
    def _get_dashboard_template(self) -> str:
        """Get HTML template for clinical dashboard"""
        return """
<!DOCTYPE html>
<html>
<head>
    <title>Clinical Decision Support System</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f5f5f5;
        }
        .header {
            background-color: #2c3e50;
            color: white;
            padding: 1rem;
            text-align: center;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
        }
        .card {
            background: white;
            border-radius: 8px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .form-group {
            margin: 1rem 0;
        }
        .form-group label {
            display: block;
            margin-bottom: 0.5rem;
            font-weight: bold;
        }
        .form-group input, .form-group select {
            width: 100%;
            padding: 0.5rem;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 1rem;
        }
        .btn {
            background-color: #3498db;
            color: white;
            padding: 0.75rem 1.5rem;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 1rem;
        }
        .btn:hover {
            background-color: #2980b9;
        }
        .btn-success {
            background-color: #27ae60;
        }
        .btn-success:hover {
            background-color: #229954;
        }
        .results {
            margin-top: 2rem;
        }
        .risk-high {
            background-color: #e74c3c;
            color: white;
            padding: 0.5rem;
            border-radius: 4px;
        }
        .risk-medium {
            background-color: #f39c12;
            color: white;
            padding: 0.5rem;
            border-radius: 4px;
        }
        .risk-low {
            background-color: #27ae60;
            color: white;
            padding: 0.5rem;
            border-radius: 4px;
        }
        .loading {
            text-align: center;
            padding: 2rem;
        }
        .recommendations {
            background-color: #ecf0f1;
            padding: 1rem;
            border-radius: 4px;
            margin: 1rem 0;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Clinical Decision Support System</h1>
        <p>AI-Powered Risk Assessment and Clinical Decision Support</p>
    </div>
    
    <div class="container">
        <div class="card">
            <h2>Patient Assessment</h2>
            <form id="assessmentForm">
                <div class="form-group">
                    <label for="patientId">Patient ID:</label>
                    <input type="text" id="patientId" name="patientId" required>
                </div>
                
                <div class="form-group">
                    <label for="age">Age:</label>
                    <input type="number" id="age" name="age" min="18" max="120" required>
                </div>
                
                <div class="form-group">
                    <label for="gender">Gender:</label>
                    <select id="gender" name="gender" required>
                        <option value="">Select Gender</option>
                        <option value="0">Female</option>
                        <option value="1">Male</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label for="mmse">MMSE Score (0-30):</label>
                    <input type="number" id="mmse" name="mmse" min="0" max="30">
                </div>
                
                <div class="form-group">
                    <label for="cdr">CDR Rating (0-3):</label>
                    <input type="number" id="cdr" name="cdr" min="0" max="3" step="0.5">
                </div>
                
                <div class="form-group">
                    <label for="education">Education (years):</label>
                    <input type="number" id="education" name="education" min="0" max="25">
                </div>
                
                <div class="form-group">
                    <label for="userId">Clinician ID:</label>
                    <input type="text" id="userId" name="userId" required>
                </div>
                
                <button type="submit" class="btn">Perform Assessment</button>
            </form>
        </div>
        
        <div id="results" class="results" style="display: none;"></div>
        
        <div class="card">
            <h2>Compliance Dashboard</h2>
            <button onclick="loadCompliance()" class="btn btn-success">Load Compliance Status</button>
            <div id="complianceResults" style="margin-top: 1rem;"></div>
        </div>
    </div>
    
    <script>
        document.getElementById('assessmentForm').addEventListener('submit', function(e) {
            e.preventDefault();
            performAssessment();
        });
        
        async function performAssessment() {
            const formData = new FormData(document.getElementById('assessmentForm'));
            const patientData = {
                patient_id: formData.get('patientId'),
                Age: parseInt(formData.get('age')),
                'M/F': parseInt(formData.get('gender')),
                MMSE: formData.get('mmse') ? parseInt(formData.get('mmse')) : null,
                CDR: formData.get('cdr') ? parseFloat(formData.get('cdr')) : null,
                EDUC: formData.get('education') ? parseInt(formData.get('education')) : null
            };
            
            const userId = formData.get('userId');
            
            document.getElementById('results').innerHTML = '<div class="loading">Processing assessment...</div>';
            document.getElementById('results').style.display = 'block';
            
            try {
                const response = await fetch('/api/assess', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        patient_data: patientData,
                        user_id: userId
                    })
                });
                
                const data = await response.json();
                
                if (data.success) {
                    displayResults(data.results);
                } else {
                    document.getElementById('results').innerHTML = '<div class="card"><h3>Error</h3><p>' + data.error + '</p></div>';
                }
            } catch (error) {
                document.getElementById('results').innerHTML = '<div class="card"><h3>Error</h3><p>Failed to perform assessment: ' + error.message + '</p></div>';
            }
        }
        
        function displayResults(results) {
            let html = '<div class="card"><h3>Assessment Results</h3>';
            html += '<p><strong>Session ID:</strong> ' + results.session_id + '</p>';
            html += '<p><strong>Processing Time:</strong> ' + Math.round(results.processing_time_ms) + 'ms</p>';
            
            // Clinical Summary
            html += '<h4>Clinical Summary</h4>';
            html += '<p><strong>Overall Risk Score:</strong> ' + results.clinical_summary.overall_risk_score.toFixed(3) + '</p>';
            
            if (results.clinical_summary.high_risk_conditions.length > 0) {
                html += '<p><strong>High Risk Conditions:</strong> <span class="risk-high">' + results.clinical_summary.high_risk_conditions.join(', ') + '</span></p>';
            }
            
            // Individual Assessments
            html += '<h4>Individual Risk Assessments</h4>';
            for (const [condition, assessment] of Object.entries(results.assessments)) {
                const riskClass = assessment.risk_level === 'HIGH' ? 'risk-high' : 
                                 assessment.risk_level === 'MEDIUM' ? 'risk-medium' : 'risk-low';
                
                html += '<div style="margin: 1rem 0; padding: 1rem; border: 1px solid #ddd; border-radius: 4px;">';
                html += '<h5>' + condition.charAt(0).toUpperCase() + condition.slice(1) + ' Assessment</h5>';
                html += '<p><strong>Risk Level:</strong> <span class="' + riskClass + '">' + assessment.risk_level + '</span></p>';
                html += '<p><strong>Risk Score:</strong> ' + assessment.risk_score.toFixed(3) + '</p>';
                html += '<p><strong>Confidence:</strong> ' + assessment.confidence.toFixed(3) + '</p>';
                html += '<p><strong>Interventions:</strong> ' + assessment.interventions.join(', ') + '</p>';
                html += '</div>';
            }
            
            // Next Steps
            html += '<div class="recommendations">';
            html += '<h4>Recommended Next Steps</h4>';
            html += '<ul>';
            for (const step of results.next_steps) {
                html += '<li>' + step + '</li>';
            }
            html += '</ul></div>';
            
            // Export Options
            html += '<div style="margin-top: 2rem;">';
            html += '<button onclick="exportToEHR(\'' + results.session_id + '\')" class="btn">Export to EHR</button>';
            html += '</div>';
            
            html += '</div>';
            
            document.getElementById('results').innerHTML = html;
        }
        
        async function exportToEHR(sessionId) {
            try {
                const response = await fetch('/api/export', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        session_id: sessionId,
                        ehr_endpoint: 'localhost',
                        format: 'fhir'
                    })
                });
                
                const data = await response.json();
                
                if (data.success) {
                    alert('Export successful! Results have been sent to EHR system.');
                } else {
                    alert('Export failed: ' + data.error);
                }
            } catch (error) {
                alert('Export failed: ' + error.message);
            }
        }
        
        async function loadCompliance() {
            document.getElementById('complianceResults').innerHTML = '<div class="loading">Loading compliance status...</div>';
            
            try {
                const response = await fetch('/api/compliance');
                const data = await response.json();
                
                if (data.success) {
                    const compliance = data.compliance;
                    let html = '<h4>Compliance Status</h4>';
                    html += '<p><strong>Overall Score:</strong> ' + compliance.overall_compliance_score + '/100</p>';
                    html += '<p><strong>HIPAA Status:</strong> ' + compliance.hipaa_compliance.status + '</p>';
                    html += '<p><strong>FDA Submission Ready:</strong> ' + (compliance.fda_validation.submission_ready ? 'Yes' : 'No') + '</p>';
                    
                    if (compliance.action_items.length > 0) {
                        html += '<h5>Action Items:</h5><ul>';
                        for (const item of compliance.action_items) {
                            html += '<li>' + item + '</li>';
                        }
                        html += '</ul>';
                    }
                    
                    document.getElementById('complianceResults').innerHTML = html;
                } else {
                    document.getElementById('complianceResults').innerHTML = '<p>Failed to load compliance status: ' + data.error + '</p>';
                }
            } catch (error) {
                document.getElementById('complianceResults').innerHTML = '<p>Failed to load compliance status: ' + error.message + '</p>';
            }
        }
    </script>
</body>
</html>
        """
    
    def run(self, host='0.0.0.0', port=5000, debug=False):
        """Run the web interface"""
        logger.info(f"Starting Clinical Decision Support Web Interface on {host}:{port}")
        self.app.run(host=host, port=port, debug=debug)


def main():
    """Main entry point for the Clinical Decision Support System"""
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Configuration
    config = {
        'organization_id': 'duetmind-adaptive',
        'master_password': 'secure_password_123',
        'audit_db_path': '/tmp/compliance_audit.db',
        'validation_db_path': '/tmp/fda_validation.db',
        'enable_web_interface': True
    }
    
    print("=" * 60)
    print("CLINICAL DECISION SUPPORT SYSTEM")
    print("DuetMind Adaptive - Medical AI Platform")
    print("=" * 60)
    
    try:
        # Initialize orchestrator
        orchestrator = ClinicalWorkflowOrchestrator(config)
        print("✓ Clinical Decision Support System initialized")
        
        # Test with example patient
        example_patient = {
            'patient_id': 'DEMO_PATIENT_001',
            'Age': 78,
            'M/F': 0,  # Female
            'MMSE': 22,
            'CDR': 0.5,
            'EDUC': 14,
            'nWBV': 0.72
        }
        
        print("\nProcessing example patient workflow...")
        results = orchestrator.process_patient_workflow(
            example_patient, 
            user_id='demo_clinician_001'
        )
        
        print(f"✓ Assessment completed in {results['processing_time_ms']:.0f}ms")
        print(f"✓ Session ID: {results['session_id']}")
        print(f"✓ Conditions assessed: {list(results['assessments'].keys())}")
        print(f"✓ High-risk conditions: {results['clinical_summary']['high_risk_conditions']}")
        
        # Test EHR export
        print("\nTesting EHR export...")
        export_result = orchestrator.export_to_ehr(
            results['session_id'], 
            'demo_ehr_endpoint', 
            'fhir'
        )
        print(f"✓ EHR export completed: {export_result['status']}")
        
        # Get compliance status
        print("\nChecking regulatory compliance...")
        compliance_status = orchestrator.get_compliance_status()
        print(f"✓ Compliance score: {compliance_status['overall_compliance_score']}/100")
        print(f"✓ HIPAA status: {compliance_status['hipaa_compliance']['status']}")
        
        if config.get('enable_web_interface', False):
            print("\nStarting web interface...")
            web_interface = ClinicalWebInterface(orchestrator)
            print("✓ Web interface available at: http://localhost:5000")
            print("\nPress Ctrl+C to stop the server")
            web_interface.run(debug=False)
        else:
            print("\n✓ Clinical Decision Support System ready for integration")
            
    except Exception as e:
        logger.error(f"Failed to initialize system: {e}")
        print(f"✗ System initialization failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())