# Healthcare Standards & Interoperability Guide

## Overview

This guide covers the healthcare data standards (HL7, FHIR, DICOM) supported by AiMedRes and provides configuration instructions for implementing interoperable data flows.

## Supported Standards

1. **FHIR (Fast Healthcare Interoperability Resources)** - R4
2. **HL7 v2.x** - Message-based integration
3. **DICOM** - Medical imaging integration (limited support)

## FHIR R4 Integration

### Overview

AiMedRes provides FHIR R4 compliant interfaces through the EHR integration module located at `src/aimedres/integration/ehr.py`.

### Supported FHIR Resources

| Resource | Support Level | Purpose |
|----------|--------------|---------|
| Patient | ✅ Full | Patient demographics and identifiers |
| Observation | ✅ Full | Clinical observations and measurements |
| DiagnosticReport | ✅ Full | AI assessment results |
| Condition | ✅ Full | Diagnoses and health conditions |
| MedicationStatement | ✅ Full | Medication history |
| Procedure | ⚠️ Partial | Medical procedures |
| Encounter | ⚠️ Partial | Patient encounters |
| AllergyIntolerance | ⚠️ Partial | Allergy information |

### FHIR Configuration

#### 1. Initialize FHIR Integration

```python
from src.aimedres.integration.ehr import (
    FHIRIntegrationEngine,
    FHIRPatient,
    FHIRObservation,
    FHIRDiagnosticReport
)

# Initialize FHIR engine
fhir_engine = FHIRIntegrationEngine(
    base_url="https://fhir.hospital.org",
    auth_token=os.getenv('FHIR_AUTH_TOKEN'),
    version="R4"
)
```

#### 2. Patient Data Ingestion

```python
# Retrieve patient data from FHIR server
patient_fhir = fhir_engine.get_patient(patient_id="12345")

# Convert to internal format
patient_data = fhir_engine.fhir_to_internal(patient_fhir)

# Process through AiMedRes
risk_assessment = process_patient_risk(patient_data)
```

#### 3. Results Reporting via FHIR

```python
from datetime import datetime

# Create FHIR DiagnosticReport for AI assessment
diagnostic_report = FHIRDiagnosticReport(
    id=f"aimedres-{assessment_id}",
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
        "text": "AiMedRes Alzheimer's Disease Risk Assessment"
    },
    subject={
        "reference": f"Patient/{patient_id}"
    },
    effective_datetime=datetime.now().isoformat(),
    issued=datetime.now().isoformat(),
    performer=[{
        "reference": "Organization/aimedres",
        "display": "AiMedRes AI Platform"
    }],
    result=[{
        "reference": f"Observation/{observation_id}",
        "display": "Risk Score"
    }],
    conclusion=f"Risk Level: {risk_level}, Confidence: {confidence:.2%}"
)

# Convert to FHIR JSON
fhir_json = diagnostic_report.to_fhir_json()

# Send to FHIR server
fhir_engine.create_resource("DiagnosticReport", fhir_json)
```

### FHIR API Endpoints

#### Patient Ingest Endpoint

```python
from flask import Flask, request, jsonify
from src.aimedres.security.phi_scrubber import enforce_phi_free_ingestion

app = Flask(__name__)

@app.route('/fhir/R4/Patient', methods=['POST'])
def create_patient():
    """FHIR-compliant patient creation endpoint"""
    try:
        fhir_patient = request.get_json()
        
        # Validate FHIR format
        if not fhir_patient.get('resourceType') == 'Patient':
            return jsonify({
                'error': 'Invalid resource type'
            }), 400
        
        # Convert and process
        patient = fhir_engine.fhir_to_internal(fhir_patient)
        
        # PHI validation
        enforce_phi_free_ingestion(patient, field_name="patient_data")
        
        # Store and process
        patient_id = store_patient(patient)
        
        return jsonify({
            'resourceType': 'Patient',
            'id': patient_id,
            'status': 'created'
        }), 201
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/fhir/R4/Patient/<patient_id>', methods=['GET'])
def get_patient(patient_id):
    """Retrieve patient in FHIR format"""
    try:
        patient = retrieve_patient(patient_id)
        fhir_patient = fhir_engine.internal_to_fhir(patient)
        return jsonify(fhir_patient.to_fhir_json())
    except Exception as e:
        return jsonify({'error': str(e)}), 404
```

#### Results Reporting Endpoint

```python
@app.route('/fhir/R4/DiagnosticReport', methods=['POST'])
def create_diagnostic_report():
    """Create FHIR DiagnosticReport for AI results"""
    try:
        report_data = request.get_json()
        
        # Process AI assessment
        assessment = run_ai_assessment(report_data)
        
        # Create FHIR DiagnosticReport
        fhir_report = create_fhir_diagnostic_report(assessment)
        
        # Store and return
        report_id = store_report(fhir_report)
        
        return jsonify({
            'resourceType': 'DiagnosticReport',
            'id': report_id,
            'status': 'final'
        }), 201
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400
```

### FHIR Validation

```python
def validate_fhir_resource(resource_type, resource_data):
    """Validate FHIR resource against R4 specification"""
    required_fields = {
        'Patient': ['resourceType', 'identifier'],
        'Observation': ['resourceType', 'status', 'code', 'subject'],
        'DiagnosticReport': ['resourceType', 'status', 'code', 'subject']
    }
    
    if resource_type not in required_fields:
        return False, f"Unsupported resource type: {resource_type}"
    
    for field in required_fields[resource_type]:
        if field not in resource_data:
            return False, f"Missing required field: {field}"
    
    return True, "Valid"
```

## HL7 v2.x Integration

### Overview

HL7 v2.x messages are supported for legacy system integration. Common message types include ADT (Admission, Discharge, Transfer) and ORU (Observation Result).

### Supported Message Types

| Message Type | Description | Support |
|--------------|-------------|---------|
| ADT^A01 | Patient Admission | ✅ |
| ADT^A08 | Patient Update | ✅ |
| ORU^R01 | Observation Results | ✅ |
| ORM^O01 | Order Messages | ⚠️ Partial |

### HL7 Message Parser

```python
import hl7
from typing import Dict, Any

class HL7MessageParser:
    """Parse HL7 v2.x messages"""
    
    def __init__(self):
        self.message_handlers = {
            'ADT': self.handle_adt,
            'ORU': self.handle_oru,
        }
    
    def parse_message(self, hl7_message: str) -> Dict[str, Any]:
        """Parse HL7 message and extract data"""
        try:
            # Parse HL7 message
            message = hl7.parse(hl7_message)
            
            # Extract message type
            msg_type = str(message.segment('MSH')[9]).split('^')[0]
            
            # Route to appropriate handler
            if msg_type in self.message_handlers:
                return self.message_handlers[msg_type](message)
            else:
                raise ValueError(f"Unsupported message type: {msg_type}")
                
        except Exception as e:
            raise ValueError(f"HL7 parsing error: {e}")
    
    def handle_adt(self, message) -> Dict[str, Any]:
        """Handle ADT (Admission/Discharge/Transfer) messages"""
        pid = message.segment('PID')
        pv1 = message.segment('PV1')
        
        return {
            'message_type': 'ADT',
            'patient_id': str(pid[3]),
            'patient_name': str(pid[5]),
            'birth_date': str(pid[7]),
            'gender': str(pid[8]),
            'visit_number': str(pv1[19]) if pv1 else None,
            'admission_date': str(pv1[44]) if pv1 else None,
        }
    
    def handle_oru(self, message) -> Dict[str, Any]:
        """Handle ORU (Observation Result) messages"""
        pid = message.segment('PID')
        obr = message.segment('OBR')
        
        # Extract all OBX segments (observations)
        observations = []
        for segment in message.segments('OBX'):
            observations.append({
                'type': str(segment[3]),
                'value': str(segment[5]),
                'units': str(segment[6]),
                'reference_range': str(segment[7]),
                'status': str(segment[11])
            })
        
        return {
            'message_type': 'ORU',
            'patient_id': str(pid[3]),
            'order_number': str(obr[3]) if obr else None,
            'observations': observations,
        }

# Usage example
parser = HL7MessageParser()

hl7_msg = """MSH|^~\\&|SENDING_APP|SENDING_FAC|RECEIVING_APP|RECEIVING_FAC|20240115120000||ADT^A01|MSG00001|P|2.5
PID|1||123456||DOE^JOHN||19650315|M
PV1|1|I|WARD^ROOM^BED||||||||||||||||12345"""

parsed_data = parser.parse_message(hl7_msg)
print(parsed_data)
```

### HL7 Message Generator

```python
def generate_oru_message(patient_id: str, observations: list) -> str:
    """Generate HL7 ORU^R01 message for AI results"""
    from datetime import datetime
    
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    
    # MSH segment
    msh = f"MSH|^~\\&|AIMEDRES|AIMEDRES_SYS|EHR|HOSPITAL|{timestamp}||ORU^R01|{timestamp}|P|2.5"
    
    # PID segment
    pid = f"PID|1||{patient_id}||||||||||||||||||||||||||"
    
    # OBR segment
    obr = f"OBR|1||{timestamp}|55188-7^AI Risk Assessment^LN"
    
    # OBX segments for each observation
    obx_segments = []
    for idx, obs in enumerate(observations, start=1):
        obx = (f"OBX|{idx}|NM|{obs['code']}^{obs['name']}^LN||"
               f"{obs['value']}|{obs['units']}|{obs.get('reference', '')}||"
               f"F|||{timestamp}")
        obx_segments.append(obx)
    
    # Combine all segments
    message = '\r'.join([msh, pid, obr] + obx_segments)
    
    return message

# Example usage
observations = [
    {
        'code': 'RISK-001',
        'name': 'Alzheimer Risk Score',
        'value': '0.75',
        'units': 'probability',
        'reference': '0.0-1.0'
    }
]

hl7_message = generate_oru_message('12345', observations)
print(hl7_message)
```

### HL7 Interface Configuration

```python
import socket
import threading

class HL7MLLPServer:
    """MLLP (Minimal Lower Layer Protocol) server for HL7 messages"""
    
    def __init__(self, host='0.0.0.0', port=2575):
        self.host = host
        self.port = port
        self.parser = HL7MessageParser()
        self.running = False
    
    def start(self):
        """Start HL7 MLLP server"""
        self.running = True
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind((self.host, self.port))
        server_socket.listen(5)
        
        print(f"✅ HL7 MLLP server listening on {self.host}:{self.port}")
        
        while self.running:
            try:
                client_socket, address = server_socket.accept()
                thread = threading.Thread(
                    target=self.handle_client,
                    args=(client_socket,)
                )
                thread.start()
            except Exception as e:
                print(f"❌ Server error: {e}")
    
    def handle_client(self, client_socket):
        """Handle HL7 client connection"""
        try:
            # Read message with MLLP framing
            data = b''
            while True:
                chunk = client_socket.recv(1024)
                if not chunk:
                    break
                data += chunk
                
                # Check for end of message (0x1C followed by 0x0D)
                if b'\x1c\r' in data:
                    break
            
            # Remove MLLP framing (0x0B start, 0x1C end, 0x0D terminator)
            if data.startswith(b'\x0b') and data.endswith(b'\x1c\r'):
                hl7_message = data[1:-2].decode('utf-8')
                
                # Parse message
                parsed = self.parser.parse_message(hl7_message)
                
                # Process data
                self.process_hl7_data(parsed)
                
                # Send ACK
                ack = self.generate_ack(hl7_message)
                client_socket.sendall(b'\x0b' + ack.encode('utf-8') + b'\x1c\r')
                
                print(f"✅ Processed HL7 message: {parsed['message_type']}")
            
        except Exception as e:
            print(f"❌ Client handling error: {e}")
        
        finally:
            client_socket.close()
    
    def generate_ack(self, hl7_message: str) -> str:
        """Generate HL7 ACK message"""
        lines = hl7_message.split('\r')
        msh = lines[0].split('|')
        
        msg_control_id = msh[10] if len(msh) > 10 else 'UNKNOWN'
        timestamp = msh[7] if len(msh) > 7 else ''
        
        ack = (f"MSH|^~\\&|AIMEDRES|AIMEDRES_SYS|{msh[3]}|{msh[4]}|"
               f"{timestamp}||ACK|{msg_control_id}|P|2.5\r"
               f"MSA|AA|{msg_control_id}")
        
        return ack
    
    def process_hl7_data(self, data: Dict[str, Any]):
        """Process parsed HL7 data"""
        # Implement your data processing logic here
        pass
    
    def stop(self):
        """Stop the server"""
        self.running = False

# Usage
# server = HL7MLLPServer(host='0.0.0.0', port=2575)
# server.start()
```

## DICOM Integration (Limited Support)

### Overview

AiMedRes provides basic DICOM support for medical imaging metadata. Full image processing requires additional configuration.

### DICOM Metadata Extraction

```python
import pydicom
from typing import Dict, Any

class DICOMProcessor:
    """Basic DICOM metadata processor"""
    
    def extract_metadata(self, dicom_file: str) -> Dict[str, Any]:
        """Extract metadata from DICOM file"""
        try:
            ds = pydicom.dcmread(dicom_file)
            
            metadata = {
                'patient_id': str(ds.PatientID) if 'PatientID' in ds else None,
                'patient_name': str(ds.PatientName) if 'PatientName' in ds else None,
                'study_date': str(ds.StudyDate) if 'StudyDate' in ds else None,
                'modality': str(ds.Modality) if 'Modality' in ds else None,
                'study_description': str(ds.StudyDescription) if 'StudyDescription' in ds else None,
                'series_description': str(ds.SeriesDescription) if 'SeriesDescription' in ds else None,
                'manufacturer': str(ds.Manufacturer) if 'Manufacturer' in ds else None,
                'body_part': str(ds.BodyPartExamined) if 'BodyPartExamined' in ds else None,
            }
            
            return metadata
            
        except Exception as e:
            raise ValueError(f"DICOM processing error: {e}")
    
    def anonymize_dicom(self, dicom_file: str, output_file: str):
        """Remove PHI from DICOM file"""
        try:
            ds = pydicom.dcmread(dicom_file)
            
            # Remove PHI fields
            phi_tags = [
                'PatientName', 'PatientID', 'PatientBirthDate',
                'PatientSex', 'PatientAddress', 'PatientTelephoneNumbers',
                'InstitutionName', 'InstitutionAddress',
                'ReferringPhysicianName', 'PhysiciansOfRecord'
            ]
            
            for tag in phi_tags:
                if tag in ds:
                    delattr(ds, tag)
            
            # Save anonymized DICOM
            ds.save_as(output_file)
            
            print(f"✅ DICOM anonymized: {output_file}")
            
        except Exception as e:
            raise ValueError(f"DICOM anonymization error: {e}")

# Usage
dicom_processor = DICOMProcessor()
metadata = dicom_processor.extract_metadata('brain_mri.dcm')
dicom_processor.anonymize_dicom('brain_mri.dcm', 'brain_mri_anon.dcm')
```

## Key Data Flows

### 1. Patient Ingest Flow

```
External EHR → FHIR API → PHI Scrubber → Internal DB → Processing
```

**Configuration:**

```python
def patient_ingest_pipeline(fhir_patient):
    """Complete patient ingestion pipeline"""
    
    # Step 1: Receive FHIR patient
    print("📥 Receiving FHIR patient data...")
    
    # Step 2: Convert to internal format
    patient_data = fhir_engine.fhir_to_internal(fhir_patient)
    
    # Step 3: PHI validation
    enforce_phi_free_ingestion(patient_data, field_name="patient_data")
    
    # Step 4: Store in database
    patient_id = store_patient(patient_data)
    
    # Step 5: Queue for processing
    queue_for_assessment(patient_id)
    
    print(f"✅ Patient ingested: {patient_id}")
    return patient_id
```

### 2. Results Reporting Flow

```
AI Assessment → FHIR DiagnosticReport → External EHR
```

**Configuration:**

```python
def results_reporting_pipeline(assessment_result):
    """Complete results reporting pipeline"""
    
    # Step 1: Create FHIR DiagnosticReport
    print("📤 Generating FHIR DiagnosticReport...")
    fhir_report = create_fhir_diagnostic_report(assessment_result)
    
    # Step 2: Send to external EHR
    response = fhir_engine.create_resource(
        'DiagnosticReport',
        fhir_report.to_fhir_json()
    )
    
    # Step 3: Log transaction
    log_transfer(
        method='FHIR_API',
        direction='outbound',
        resource_type='DiagnosticReport',
        resource_id=response['id'],
        success=True
    )
    
    print(f"✅ Results reported: {response['id']}")
    return response['id']
```

### 3. Audit Flow

```
All Operations → Audit Logger → Secure Audit DB → FHIR AuditEvent
```

**Configuration:**

```python
from src.aimedres.security.monitoring import SecurityMonitor

security_monitor = SecurityMonitor({
    'audit_enabled': True,
    'audit_log_path': '/var/log/aimedres/audit.log'
})

def audit_data_access(user, resource_type, resource_id, action):
    """Audit all data access"""
    security_monitor.log_access(
        user_id=user,
        resource=f"{resource_type}/{resource_id}",
        action=action,
        timestamp=datetime.now()
    )
```

## Environment Configuration

Add to `.env` file:

```bash
# FHIR Configuration
FHIR_ENABLED=true
FHIR_BASE_URL=https://fhir.hospital.org
FHIR_AUTH_TOKEN=your_fhir_token
FHIR_VERSION=R4

# HL7 Configuration
HL7_ENABLED=true
HL7_MLLP_HOST=0.0.0.0
HL7_MLLP_PORT=2575

# DICOM Configuration
DICOM_ENABLED=false
DICOM_STORAGE_PATH=/data/dicom

# Audit Configuration
AUDIT_ENABLED=true
AUDIT_LOG_PATH=/var/log/aimedres/audit.log
```

## Testing and Validation

### FHIR Validation Script

```python
def test_fhir_integration():
    """Test FHIR integration"""
    print("Testing FHIR Integration...")
    
    # Test patient creation
    test_patient = {
        'resourceType': 'Patient',
        'identifier': [{'value': 'TEST-001'}],
        'name': [{'family': 'Test', 'given': ['Patient']}],
        'gender': 'male',
        'birthDate': '1970-01-01'
    }
    
    try:
        result = fhir_engine.create_resource('Patient', test_patient)
        print(f"✅ FHIR patient creation: {result['id']}")
        return True
    except Exception as e:
        print(f"❌ FHIR test failed: {e}")
        return False
```

## Compliance Checklist

- [ ] FHIR R4 endpoints implemented
- [ ] HL7 v2.x message handling configured
- [ ] PHI scrubber integrated at all ingestion points
- [ ] FHIR resource validation enabled
- [ ] Audit logging for all data flows
- [ ] Error handling and retry logic implemented
- [ ] Data integrity verification
- [ ] Performance testing completed
- [ ] Integration with external systems tested
- [ ] Documentation updated

## References

- FHIR R4 Specification: [HL7.org/FHIR](https://www.hl7.org/fhir/)
- HL7 v2.x Documentation: [HL7.org/v2](https://www.hl7.org/implement/standards/product_brief.cfm?product_id=185)
- DICOM Standard: [dicomstandard.org](https://www.dicomstandard.org/)
