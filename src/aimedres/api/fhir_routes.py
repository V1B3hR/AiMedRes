"""
FHIR Integration API endpoints.

Provides endpoints for:
- FHIR patient data retrieval (sandbox/mock)
- Consent enforcement
- FHIR resource mapping
"""

import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from flask import Blueprint, request, jsonify

from ..security.auth import require_auth

logger = logging.getLogger(__name__)

fhir_bp = Blueprint('fhir', __name__, url_prefix='/api/v1/fhir')


class FHIRMockServer:
    """
    Mock FHIR server for development and testing.
    
    Simulates FHIR R4 API responses.
    """
    
    def __init__(self):
        self.patients = self._initialize_mock_patients()
        self.consent_records = self._initialize_consent_records()
    
    def _initialize_mock_patients(self) -> Dict[str, Any]:
        """Initialize mock patient data."""
        return {
            'patient-001': {
                'resourceType': 'Patient',
                'id': 'patient-001',
                'identifier': [
                    {
                        'system': 'urn:oid:1.2.36.146.595.217.0.1',
                        'value': 'PAT-001-DEIDENTIFIED'
                    }
                ],
                'name': [
                    {
                        'use': 'official',
                        'family': 'REDACTED',
                        'given': ['REDACTED']
                    }
                ],
                'gender': 'male',
                'birthDate': '1950-01-15',
                'address': [
                    {
                        'use': 'home',
                        'city': 'REDACTED',
                        'state': 'CA',
                        'postalCode': 'REDACTED'
                    }
                ],
                'active': True
            },
            'patient-002': {
                'resourceType': 'Patient',
                'id': 'patient-002',
                'identifier': [
                    {
                        'system': 'urn:oid:1.2.36.146.595.217.0.1',
                        'value': 'PAT-002-DEIDENTIFIED'
                    }
                ],
                'name': [
                    {
                        'use': 'official',
                        'family': 'REDACTED',
                        'given': ['REDACTED']
                    }
                ],
                'gender': 'female',
                'birthDate': '1948-05-22',
                'active': True
            },
            'patient-003': {
                'resourceType': 'Patient',
                'id': 'patient-003',
                'identifier': [
                    {
                        'system': 'urn:oid:1.2.36.146.595.217.0.1',
                        'value': 'PAT-003-DEIDENTIFIED'
                    }
                ],
                'name': [
                    {
                        'use': 'official',
                        'family': 'REDACTED',
                        'given': ['REDACTED']
                    }
                ],
                'gender': 'female',
                'birthDate': '1955-11-08',
                'active': True
            }
        }
    
    def _initialize_consent_records(self) -> Dict[str, Any]:
        """Initialize mock consent records."""
        return {
            'patient-001': {
                'consent_given': True,
                'scope': ['research', 'ai_analysis'],
                'granted_date': '2024-12-01T10:00:00Z',
                'expires_date': '2025-12-01T10:00:00Z'
            },
            'patient-002': {
                'consent_given': True,
                'scope': ['research'],
                'granted_date': '2024-11-15T14:30:00Z',
                'expires_date': '2025-11-15T14:30:00Z'
            },
            'patient-003': {
                'consent_given': False,
                'scope': [],
                'granted_date': None,
                'expires_date': None
            }
        }
    
    def check_consent(self, patient_id: str, required_scope: str = 'research') -> bool:
        """
        Check if patient has given consent for the required scope.
        
        Args:
            patient_id: Patient identifier
            required_scope: Required consent scope
        
        Returns:
            True if consent granted
        """
        if patient_id not in self.consent_records:
            return False
        
        consent = self.consent_records[patient_id]
        
        # Check if consent is active
        if not consent['consent_given']:
            return False
        
        # Check if consent scope includes required scope
        if required_scope not in consent['scope']:
            return False
        
        # Check if consent has expired
        if consent['expires_date']:
            expires = datetime.fromisoformat(consent['expires_date'].replace('Z', '+00:00'))
            if datetime.now(expires.tzinfo) > expires:
                return False
        
        return True
    
    def get_patient(self, patient_id: str) -> Optional[Dict[str, Any]]:
        """Get patient resource by ID."""
        return self.patients.get(patient_id)
    
    def list_patients(self, active_only: bool = True) -> List[Dict[str, Any]]:
        """List all patients."""
        patients = list(self.patients.values())
        
        if active_only:
            patients = [p for p in patients if p.get('active', False)]
        
        return patients


# Global FHIR mock server
fhir_server = FHIRMockServer()


@fhir_bp.route('/patients', methods=['GET'])
@require_auth()
def list_fhir_patients():
    """
    List FHIR patients from mock/sandbox server.
    
    Query Parameters:
        active: Filter by active status (default: true)
    
    Returns:
        FHIR Bundle with patient resources
    """
    try:
        active_only = request.args.get('active', 'true').lower() == 'true'
        
        patients = fhir_server.list_patients(active_only=active_only)
        
        # Return FHIR Bundle format
        bundle = {
            'resourceType': 'Bundle',
            'type': 'searchset',
            'total': len(patients),
            'entry': [
                {
                    'resource': patient,
                    'fullUrl': f"Patient/{patient['id']}"
                }
                for patient in patients
            ]
        }
        
        logger.info(f"Listed {len(patients)} FHIR patients")
        
        return jsonify(bundle)
        
    except Exception as e:
        logger.error(f"FHIR patients list error: {e}")
        return jsonify({'error': 'Failed to list patients', 'message': str(e)}), 500


@fhir_bp.route('/patients/<patient_id>', methods=['GET'])
@require_auth()
def get_fhir_patient(patient_id: str):
    """
    Get FHIR patient by ID with consent enforcement.
    
    Args:
        patient_id: Patient identifier
    
    Query Parameters:
        scope: Required consent scope (default: research)
    
    Returns:
        FHIR Patient resource
    """
    try:
        required_scope = request.args.get('scope', 'research')
        
        # Check consent first
        if not fhir_server.check_consent(patient_id, required_scope):
            logger.warning(f"Consent not granted for patient: {patient_id}")
            return jsonify({
                'error': 'Consent not granted',
                'message': f'Patient has not granted consent for {required_scope}'
            }), 403
        
        # Get patient data
        patient = fhir_server.get_patient(patient_id)
        
        if not patient:
            return jsonify({
                'error': 'Patient not found',
                'message': f'Patient with ID {patient_id} not found'
            }), 404
        
        logger.info(f"Retrieved FHIR patient: {patient_id}")
        
        return jsonify(patient)
        
    except Exception as e:
        logger.error(f"FHIR patient retrieval error: {e}")
        return jsonify({'error': 'Failed to retrieve patient', 'message': str(e)}), 500


@fhir_bp.route('/patients/<patient_id>/consent', methods=['GET'])
@require_auth()
def get_patient_consent(patient_id: str):
    """
    Get consent status for a patient.
    
    Args:
        patient_id: Patient identifier
    
    Returns:
        Consent status and details
    """
    try:
        consent = fhir_server.consent_records.get(patient_id)
        
        if not consent:
            return jsonify({
                'error': 'Consent record not found',
                'message': f'No consent record for patient {patient_id}'
            }), 404
        
        return jsonify({
            'patient_id': patient_id,
            'consent_status': consent,
            'retrieved_at': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Consent retrieval error: {e}")
        return jsonify({'error': 'Failed to retrieve consent', 'message': str(e)}), 500


@fhir_bp.route('/patients/<patient_id>/observations', methods=['GET'])
@require_auth()
def get_patient_observations(patient_id: str):
    """
    Get clinical observations for a patient.
    
    Args:
        patient_id: Patient identifier
    
    Returns:
        FHIR Bundle with Observation resources
    """
    try:
        # Check consent
        if not fhir_server.check_consent(patient_id, 'research'):
            return jsonify({'error': 'Consent not granted'}), 403
        
        # Mock observations
        observations = [
            {
                'resourceType': 'Observation',
                'id': f'obs-{patient_id}-001',
                'status': 'final',
                'code': {
                    'coding': [{
                        'system': 'http://loinc.org',
                        'code': '8867-4',
                        'display': 'Heart rate'
                    }]
                },
                'subject': {'reference': f'Patient/{patient_id}'},
                'valueQuantity': {
                    'value': 72,
                    'unit': 'beats/minute'
                }
            },
            {
                'resourceType': 'Observation',
                'id': f'obs-{patient_id}-002',
                'status': 'final',
                'code': {
                    'coding': [{
                        'system': 'http://loinc.org',
                        'code': '8480-6',
                        'display': 'Systolic blood pressure'
                    }]
                },
                'subject': {'reference': f'Patient/{patient_id}'},
                'valueQuantity': {
                    'value': 120,
                    'unit': 'mmHg'
                }
            }
        ]
        
        bundle = {
            'resourceType': 'Bundle',
            'type': 'searchset',
            'total': len(observations),
            'entry': [
                {'resource': obs}
                for obs in observations
            ]
        }
        
        logger.info(f"Retrieved observations for patient: {patient_id}")
        
        return jsonify(bundle)
        
    except Exception as e:
        logger.error(f"Observations retrieval error: {e}")
        return jsonify({'error': 'Failed to retrieve observations', 'message': str(e)}), 500
