# EMR/EHR Integration Guide

## Overview

This guide provides comprehensive instructions for integrating AiMedRes with Electronic Medical Record (EMR) and Electronic Health Record (EHR) systems using industry-standard protocols and connectors.

## Integration Architecture

```
┌─────────────────────┐
│   EMR/EHR System    │
│  (Epic, Cerner,     │
│   Allscripts, etc.) │
└──────────┬──────────┘
           │
    ┌──────┴───────┐
    │  Integration │
    │    Layer     │
    │ (HL7/FHIR)   │
    └──────┬───────┘
           │
┌──────────┴──────────┐
│  AiMedRes Platform  │
│  - PHI Scrubber     │
│  - AI Processing    │
│  - Results Engine   │
└─────────────────────┘
```

## Supported EMR/EHR Systems

### Tier 1 Support (Tested)
- **Epic** - FHIR R4 & HL7 v2.x
- **Cerner** - FHIR R4 & HL7 v2.x
- **Allscripts** - HL7 v2.x

### Tier 2 Support (Compatible)
- **athenahealth** - FHIR R4
- **eClinicalWorks** - HL7 v2.x
- **NextGen Healthcare** - HL7 v2.x
- **Meditech** - HL7 v2.x

### Custom/Generic Support
- Any system supporting FHIR R4 or HL7 v2.x

## Integration Methods

### Method 1: FHIR REST API (Recommended)

Most modern EMR/EHR systems support FHIR REST APIs, making this the preferred integration method.

#### Epic FHIR Integration

```python
from src.aimedres.integration.ehr import FHIRIntegrationEngine
import requests

class EpicFHIRConnector:
    """Epic FHIR R4 integration connector"""
    
    def __init__(self, base_url, client_id, client_secret):
        self.base_url = base_url
        self.client_id = client_id
        self.client_secret = client_secret
        self.access_token = None
        
        # Initialize FHIR engine
        self.fhir_engine = FHIRIntegrationEngine(
            base_url=base_url,
            auth_token=None,  # Will be set after OAuth
            version="R4"
        )
    
    def authenticate(self):
        """OAuth 2.0 authentication with Epic"""
        token_url = f"{self.base_url}/oauth2/token"
        
        payload = {
            'grant_type': 'client_credentials',
            'client_id': self.client_id,
            'client_secret': self.client_secret
        }
        
        response = requests.post(token_url, data=payload)
        response.raise_for_status()
        
        self.access_token = response.json()['access_token']
        self.fhir_engine.auth_token = self.access_token
        
        print("✅ Authenticated with Epic")
    
    def get_patient(self, patient_id):
        """Retrieve patient from Epic"""
        return self.fhir_engine.get_patient(patient_id)
    
    def get_observations(self, patient_id, category=None):
        """Retrieve patient observations"""
        return self.fhir_engine.get_observations(patient_id, category)
    
    def create_diagnostic_report(self, report_data):
        """Send diagnostic report to Epic"""
        return self.fhir_engine.create_resource('DiagnosticReport', report_data)

# Usage
epic_connector = EpicFHIRConnector(
    base_url='https://fhir.epic.com/interconnect-fhir-oauth/api/FHIR/R4',
    client_id=os.getenv('EPIC_CLIENT_ID'),
    client_secret=os.getenv('EPIC_CLIENT_SECRET')
)

epic_connector.authenticate()
patient = epic_connector.get_patient('12345')
```

#### Cerner FHIR Integration

```python
class CernerFHIRConnector:
    """Cerner FHIR R4 integration connector"""
    
    def __init__(self, base_url, api_key):
        self.base_url = base_url
        self.api_key = api_key
        
        self.fhir_engine = FHIRIntegrationEngine(
            base_url=base_url,
            auth_token=api_key,
            version="R4"
        )
    
    def get_patient(self, patient_id):
        """Retrieve patient from Cerner"""
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Accept': 'application/fhir+json'
        }
        
        response = requests.get(
            f"{self.base_url}/Patient/{patient_id}",
            headers=headers
        )
        response.raise_for_status()
        
        return response.json()
    
    def search_patients(self, family_name=None, given_name=None, birthdate=None):
        """Search patients in Cerner"""
        params = {}
        if family_name:
            params['family'] = family_name
        if given_name:
            params['given'] = given_name
        if birthdate:
            params['birthdate'] = birthdate
        
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Accept': 'application/fhir+json'
        }
        
        response = requests.get(
            f"{self.base_url}/Patient",
            params=params,
            headers=headers
        )
        response.raise_for_status()
        
        return response.json()

# Usage
cerner_connector = CernerFHIRConnector(
    base_url='https://fhir.cerner.com/r4/<tenant_id>',
    api_key=os.getenv('CERNER_API_KEY')
)

patient = cerner_connector.get_patient('12345')
```

### Method 2: HL7 v2.x Interface Engine

For legacy systems or those primarily using HL7 v2.x messaging.

#### HL7 Interface Configuration

```python
from deployment.data_integration.standards_interoperability_guide import (
    HL7MessageParser,
    HL7MLLPServer
)

class EMRInterfaceEngine:
    """HL7 interface engine for EMR integration"""
    
    def __init__(self, host='0.0.0.0', port=2575):
        self.parser = HL7MessageParser()
        self.server = HL7MLLPServer(host, port)
        self.message_queue = []
    
    def start_listener(self):
        """Start HL7 MLLP listener"""
        print(f"🚀 Starting HL7 interface engine...")
        self.server.start()
    
    def send_hl7_message(self, target_host, target_port, message):
        """Send HL7 message to EMR"""
        import socket
        
        try:
            # Connect to EMR's HL7 interface
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((target_host, target_port))
            
            # Send with MLLP framing
            mllp_message = b'\x0b' + message.encode('utf-8') + b'\x1c\r'
            sock.sendall(mllp_message)
            
            # Receive ACK
            ack_data = sock.recv(1024)
            sock.close()
            
            if b'MSA|AA' in ack_data:
                print("✅ Message acknowledged by EMR")
                return True
            else:
                print("❌ Message rejected by EMR")
                return False
                
        except Exception as e:
            print(f"❌ Failed to send HL7 message: {e}")
            return False

# Usage
interface_engine = EMRInterfaceEngine(host='0.0.0.0', port=2575)
# interface_engine.start_listener()  # Run in production
```

### Method 3: Database Integration (Direct)

**⚠️ WARNING**: Direct database integration bypasses EMR business logic and should only be used with explicit approval and proper safeguards.

```python
import psycopg2
from typing import Dict, List

class DirectDatabaseConnector:
    """Direct database connector (use with caution)"""
    
    def __init__(self, db_config):
        self.db_config = db_config
        self.conn = None
    
    def connect(self):
        """Connect to EMR database"""
        self.conn = psycopg2.connect(
            host=self.db_config['host'],
            port=self.db_config['port'],
            database=self.db_config['database'],
            user=self.db_config['user'],
            password=self.db_config['password']
        )
        print("✅ Connected to EMR database")
    
    def get_patient_by_mrn(self, mrn: str) -> Dict:
        """Retrieve patient by MRN (READ ONLY)"""
        cursor = self.conn.cursor()
        
        # Example query - adjust for your EMR schema
        query = """
            SELECT patient_id, first_name, last_name, birth_date, gender
            FROM patients
            WHERE medical_record_number = %s
        """
        
        cursor.execute(query, (mrn,))
        row = cursor.fetchone()
        
        if row:
            return {
                'patient_id': row[0],
                'first_name': row[1],
                'last_name': row[2],
                'birth_date': row[3],
                'gender': row[4]
            }
        return None
    
    def disconnect(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            print("✅ Database connection closed")

# Usage (only with proper authorization)
# db_connector = DirectDatabaseConnector({
#     'host': 'emr-db.hospital.org',
#     'port': 5432,
#     'database': 'emr_prod',
#     'user': 'readonly_user',
#     'password': os.getenv('EMR_DB_PASSWORD')
# })
```

## Complete Integration Pipeline

### Bi-directional Integration Setup

```python
from src.aimedres.security.phi_scrubber import PHIScrubber
from src.aimedres.integration.ehr import FHIRIntegrationEngine

class CompleteEMRIntegration:
    """Complete bi-directional EMR/EHR integration"""
    
    def __init__(self, emr_type, config):
        self.emr_type = emr_type
        self.config = config
        
        # Initialize components
        self.phi_scrubber = PHIScrubber(aggressive=True)
        self.fhir_engine = FHIRIntegrationEngine(
            base_url=config['fhir_base_url'],
            auth_token=config['auth_token'],
            version="R4"
        )
        
        # Initialize EMR-specific connector
        self.connector = self._initialize_connector()
    
    def _initialize_connector(self):
        """Initialize EMR-specific connector"""
        if self.emr_type == 'epic':
            return EpicFHIRConnector(
                base_url=self.config['fhir_base_url'],
                client_id=self.config['client_id'],
                client_secret=self.config['client_secret']
            )
        elif self.emr_type == 'cerner':
            return CernerFHIRConnector(
                base_url=self.config['fhir_base_url'],
                api_key=self.config['api_key']
            )
        else:
            raise ValueError(f"Unsupported EMR type: {self.emr_type}")
    
    def ingest_patient_data(self, patient_id: str) -> Dict:
        """
        Complete patient data ingestion from EMR
        
        Steps:
        1. Retrieve from EMR via FHIR
        2. Validate and de-identify
        3. Store in AiMedRes
        4. Queue for processing
        """
        print(f"📥 Ingesting patient {patient_id} from {self.emr_type}...")
        
        # Step 1: Retrieve from EMR
        fhir_patient = self.connector.get_patient(patient_id)
        fhir_observations = self.connector.get_observations(patient_id)
        
        # Step 2: Convert to internal format
        patient_data = self.fhir_engine.fhir_to_internal(fhir_patient)
        
        # Step 3: PHI validation
        validation_result = self.phi_scrubber.detect_phi(str(patient_data))
        if validation_result.has_phi:
            print(f"⚠️  PHI detected: {validation_result.phi_types_found}")
            patient_data = self.phi_scrubber.sanitize(str(patient_data))
        
        # Step 4: Store and process
        internal_id = self._store_patient(patient_data)
        
        print(f"✅ Patient ingested: {internal_id}")
        return {
            'internal_id': internal_id,
            'external_id': patient_id,
            'status': 'success'
        }
    
    def send_results_to_emr(self, patient_id: str, assessment_result: Dict) -> bool:
        """
        Send AI assessment results back to EMR
        
        Steps:
        1. Format as FHIR DiagnosticReport
        2. Validate FHIR structure
        3. Send to EMR
        4. Verify receipt
        """
        print(f"📤 Sending results for patient {patient_id} to {self.emr_type}...")
        
        # Step 1: Create FHIR DiagnosticReport
        diagnostic_report = self._create_diagnostic_report(
            patient_id,
            assessment_result
        )
        
        # Step 2: Validate FHIR structure
        if not self._validate_fhir_resource(diagnostic_report):
            print("❌ Invalid FHIR resource")
            return False
        
        # Step 3: Send to EMR
        try:
            response = self.connector.create_diagnostic_report(
                diagnostic_report.to_fhir_json()
            )
            
            print(f"✅ Results sent to EMR: {response['id']}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to send results: {e}")
            return False
    
    def _create_diagnostic_report(self, patient_id, assessment):
        """Create FHIR DiagnosticReport from assessment"""
        from src.aimedres.integration.ehr import FHIRDiagnosticReport
        from datetime import datetime
        
        return FHIRDiagnosticReport(
            id=f"aimedres-{assessment['id']}",
            status="final",
            category=[{
                "coding": [{
                    "system": "http://terminology.hl7.org/CodeSystem/v2-0074",
                    "code": "LAB",
                    "display": "Laboratory"
                }]
            }],
            code={
                "coding": [{
                    "system": "http://loinc.org",
                    "code": "55188-7",
                    "display": "AI Risk Assessment"
                }],
                "text": f"AiMedRes {assessment['type']} Assessment"
            },
            subject={"reference": f"Patient/{patient_id}"},
            effective_datetime=datetime.now().isoformat(),
            issued=datetime.now().isoformat(),
            performer=[{
                "reference": "Organization/aimedres",
                "display": "AiMedRes AI Platform"
            }],
            result=[],
            conclusion=assessment['conclusion']
        )
    
    def _validate_fhir_resource(self, resource) -> bool:
        """Validate FHIR resource structure"""
        # Basic validation
        return hasattr(resource, 'to_fhir_json')
    
    def _store_patient(self, patient_data) -> str:
        """Store patient in internal database"""
        # Implement your storage logic
        return f"INTERNAL_{hash(str(patient_data))}"

# Usage example
integration = CompleteEMRIntegration(
    emr_type='epic',
    config={
        'fhir_base_url': os.getenv('EPIC_FHIR_URL'),
        'client_id': os.getenv('EPIC_CLIENT_ID'),
        'client_secret': os.getenv('EPIC_CLIENT_SECRET')
    }
)

# Ingest patient
result = integration.ingest_patient_data('12345')

# Process and send results
assessment = {
    'id': 'assess-001',
    'type': 'Alzheimer Risk',
    'conclusion': 'High risk detected. Recommend further evaluation.'
}
integration.send_results_to_emr('12345', assessment)
```

## Configuration by EMR System

### Epic Configuration

```bash
# .env configuration for Epic
EPIC_ENABLED=true
EPIC_FHIR_URL=https://fhir.epic.com/interconnect-fhir-oauth/api/FHIR/R4
EPIC_CLIENT_ID=your_client_id
EPIC_CLIENT_SECRET=your_client_secret
EPIC_TENANT_ID=your_tenant_id

# Epic-specific settings
EPIC_SANDBOX_MODE=false
EPIC_API_VERSION=R4
```

### Cerner Configuration

```bash
# .env configuration for Cerner
CERNER_ENABLED=true
CERNER_FHIR_URL=https://fhir.cerner.com/r4/<tenant_id>
CERNER_API_KEY=your_api_key
CERNER_TENANT_ID=your_tenant_id

# Cerner-specific settings
CERNER_SANDBOX_MODE=false
CERNER_API_VERSION=R4
```

### Allscripts Configuration

```bash
# .env configuration for Allscripts
ALLSCRIPTS_ENABLED=true
ALLSCRIPTS_HL7_HOST=hl7.allscripts.hospital.org
ALLSCRIPTS_HL7_PORT=2575
ALLSCRIPTS_INTERFACE_TYPE=HL7v2

# Allscripts-specific settings
ALLSCRIPTS_MESSAGE_ENCODING=UTF-8
ALLSCRIPTS_ACK_TIMEOUT=30
```

## Testing EMR Integration

### Integration Test Suite

```python
import unittest

class TestEMRIntegration(unittest.TestCase):
    """Test suite for EMR integration"""
    
    def setUp(self):
        """Set up test environment"""
        self.integration = CompleteEMRIntegration(
            emr_type='epic',
            config={
                'fhir_base_url': 'https://fhir.epic.com/test',
                'client_id': 'test_client',
                'client_secret': 'test_secret'
            }
        )
    
    def test_patient_ingestion(self):
        """Test patient data ingestion"""
        result = self.integration.ingest_patient_data('TEST-001')
        self.assertEqual(result['status'], 'success')
        self.assertIsNotNone(result['internal_id'])
    
    def test_results_reporting(self):
        """Test results reporting to EMR"""
        assessment = {
            'id': 'test-assess-001',
            'type': 'Test Assessment',
            'conclusion': 'Test conclusion'
        }
        
        success = self.integration.send_results_to_emr('TEST-001', assessment)
        self.assertTrue(success)
    
    def test_phi_handling(self):
        """Test PHI scrubbing in pipeline"""
        test_data = "Patient John Smith, DOB 03/15/1965"
        result = self.integration.phi_scrubber.detect_phi(test_data)
        self.assertTrue(result.has_phi)

if __name__ == '__main__':
    unittest.main()
```

### Manual Testing Checklist

- [ ] Connection to EMR established successfully
- [ ] Patient data retrieval working
- [ ] PHI scrubbing functioning correctly
- [ ] Data transformation accurate
- [ ] Results successfully sent to EMR
- [ ] Error handling working properly
- [ ] Logging capturing all transactions
- [ ] Performance meets requirements (< 2s per transaction)
- [ ] Concurrent requests handled properly
- [ ] Failover/retry logic tested

## Monitoring and Maintenance

### Integration Monitoring Dashboard

```python
class EMRIntegrationMonitor:
    """Monitor EMR integration health and performance"""
    
    def __init__(self):
        self.metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'average_response_time': 0,
            'last_check': None
        }
    
    def record_request(self, success: bool, response_time: float):
        """Record request metrics"""
        self.metrics['total_requests'] += 1
        
        if success:
            self.metrics['successful_requests'] += 1
        else:
            self.metrics['failed_requests'] += 1
        
        # Update average response time
        n = self.metrics['total_requests']
        current_avg = self.metrics['average_response_time']
        self.metrics['average_response_time'] = (
            (current_avg * (n - 1) + response_time) / n
        )
    
    def get_health_status(self) -> Dict:
        """Get integration health status"""
        total = self.metrics['total_requests']
        success_rate = (
            self.metrics['successful_requests'] / total * 100
            if total > 0 else 0
        )
        
        status = 'healthy' if success_rate >= 95 else 'degraded'
        if success_rate < 80:
            status = 'critical'
        
        return {
            'status': status,
            'success_rate': f"{success_rate:.2f}%",
            'average_response_time': f"{self.metrics['average_response_time']:.2f}s",
            'total_requests': total
        }

# Usage
monitor = EMRIntegrationMonitor()

# Record each request
import time
start = time.time()
success = integration.ingest_patient_data('12345')
elapsed = time.time() - start
monitor.record_request(success, elapsed)

# Check health
health = monitor.get_health_status()
print(f"Integration Status: {health['status']}")
```

## Troubleshooting

### Common Issues and Solutions

#### Issue: Authentication Failures

**Solution**: Verify credentials and token expiration

```python
def diagnose_auth_issue(connector):
    """Diagnose authentication issues"""
    print("🔍 Diagnosing authentication...")
    
    # Check credentials
    if not connector.access_token:
        print("❌ No access token - authenticate first")
        connector.authenticate()
    
    # Test token validity
    try:
        connector.get_patient('test')
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            print("❌ Token expired - re-authenticating")
            connector.authenticate()
        else:
            print(f"❌ Other auth error: {e}")
```

#### Issue: FHIR Resource Validation Errors

**Solution**: Validate against FHIR schema

```python
def validate_fhir_strict(resource_data):
    """Strict FHIR validation"""
    from jsonschema import validate, ValidationError
    
    # Load FHIR schema (simplified example)
    schema = {
        "type": "object",
        "required": ["resourceType", "id"],
        "properties": {
            "resourceType": {"type": "string"},
            "id": {"type": "string"}
        }
    }
    
    try:
        validate(instance=resource_data, schema=schema)
        print("✅ FHIR resource valid")
        return True
    except ValidationError as e:
        print(f"❌ FHIR validation error: {e}")
        return False
```

#### Issue: Performance Degradation

**Solution**: Implement caching and batch processing

```python
from functools import lru_cache
import time

class OptimizedEMRConnector:
    """EMR connector with performance optimizations"""
    
    @lru_cache(maxsize=100)
    def get_patient_cached(self, patient_id):
        """Cached patient retrieval"""
        return self.get_patient(patient_id)
    
    def batch_get_patients(self, patient_ids):
        """Batch patient retrieval"""
        patients = []
        for patient_id in patient_ids:
            try:
                patient = self.get_patient_cached(patient_id)
                patients.append(patient)
            except Exception as e:
                print(f"❌ Failed to get patient {patient_id}: {e}")
        
        return patients
```

## Security Considerations

### EMR Integration Security Checklist

- [ ] All connections use TLS 1.2+
- [ ] Credentials stored securely (environment variables, secrets manager)
- [ ] PHI scrubbing enabled and tested
- [ ] Audit logging for all EMR interactions
- [ ] Network isolated (DMZ or private network)
- [ ] Rate limiting configured
- [ ] Timeout values appropriate
- [ ] Error messages don't leak sensitive data
- [ ] Regular security audits scheduled
- [ ] Incident response plan documented

## Support and Resources

### EMR Vendor Documentation

- **Epic**: [fhir.epic.com](https://fhir.epic.com/)
- **Cerner**: [fhir.cerner.com](https://fhir.cerner.com/)
- **Allscripts**: Contact Allscripts support

### AiMedRes Resources

- EHR Integration Module: `src/aimedres/integration/ehr.py`
- FHIR Implementation: `src/aimedres/integration/ehr.py`
- Test Suite: `tests/integration/`

## References

- HL7 FHIR: [HL7.org/FHIR](https://www.hl7.org/fhir/)
- Epic on FHIR: [fhir.epic.com](https://fhir.epic.com/)
- Cerner FHIR: [fhir.cerner.com](https://fhir.cerner.com/)
