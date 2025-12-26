#!/usr/bin/env python3
"""
EHR Integration Module for Clinical Decision Support System

Provides FHIR-compliant interfaces for integrating with Electronic Health Record systems:
- FHIR R4 data model compliance
- HL7 message processing
- Real-time data ingestion
- Bi-directional synchronization
- Secure data exchange protocols

This module enables seamless integration with existing healthcare infrastructure.
"""

import hashlib
import json
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Import existing security components
from secure_medical_processor import SecureMedicalDataProcessor

logger = logging.getLogger("EHRIntegration")


@dataclass
class FHIRPatient:
    """FHIR R4 Patient resource representation"""

    id: str
    identifier: List[Dict[str, str]]
    active: bool
    name: List[Dict[str, Any]]
    gender: str
    birth_date: str
    address: List[Dict[str, Any]]
    telecom: List[Dict[str, str]]
    extension: List[Dict[str, Any]] = None

    def to_fhir_json(self) -> Dict[str, Any]:
        """Convert to FHIR JSON format"""
        return {
            "resourceType": "Patient",
            "id": self.id,
            "identifier": self.identifier,
            "active": self.active,
            "name": self.name,
            "gender": self.gender,
            "birthDate": self.birth_date,
            "address": self.address,
            "telecom": self.telecom,
            "extension": self.extension or [],
        }


@dataclass
class FHIRObservation:
    """FHIR R4 Observation resource representation"""

    id: str
    status: str
    category: List[Dict[str, Any]]
    code: Dict[str, Any]
    subject: Dict[str, str]
    effective_datetime: str
    value: Union[Dict[str, Any], str, float]
    performer: List[Dict[str, str]]
    interpretation: List[Dict[str, Any]] = None
    reference_range: List[Dict[str, Any]] = None

    def to_fhir_json(self) -> Dict[str, Any]:
        """Convert to FHIR JSON format"""
        return {
            "resourceType": "Observation",
            "id": self.id,
            "status": self.status,
            "category": self.category,
            "code": self.code,
            "subject": self.subject,
            "effectiveDateTime": self.effective_datetime,
            "valueQuantity": self.value if isinstance(self.value, dict) else {"value": self.value},
            "performer": self.performer,
            "interpretation": self.interpretation or [],
            "referenceRange": self.reference_range or [],
        }


@dataclass
class FHIRDiagnosticReport:
    """FHIR R4 DiagnosticReport resource for AI assessments"""

    id: str
    status: str
    category: List[Dict[str, Any]]
    code: Dict[str, Any]
    subject: Dict[str, str]
    effective_datetime: str
    issued: str
    performer: List[Dict[str, str]]
    result: List[Dict[str, str]]
    conclusion: str
    conclusion_code: List[Dict[str, Any]]

    def to_fhir_json(self) -> Dict[str, Any]:
        """Convert to FHIR JSON format"""
        return {
            "resourceType": "DiagnosticReport",
            "id": self.id,
            "status": self.status,
            "category": self.category,
            "code": self.code,
            "subject": self.subject,
            "effectiveDateTime": self.effective_datetime,
            "issued": self.issued,
            "performer": self.performer,
            "result": self.result,
            "conclusion": self.conclusion,
            "conclusionCode": self.conclusion_code,
        }


class FHIRConverter:
    """
    Converts between internal data formats and FHIR R4 resources.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.organization_id = config.get("organization_id", "duetmind-adaptive")

    def patient_data_to_fhir(self, patient_data: Dict[str, Any]) -> FHIRPatient:
        """Convert internal patient data to FHIR Patient resource"""

        patient_id = patient_data.get("patient_id", str(uuid.uuid4()))

        # Convert gender
        gender_map = {0: "female", 1: "male", "F": "female", "M": "male"}
        gender = gender_map.get(patient_data.get("M/F", "unknown"), "unknown")

        # Calculate birth date from age
        birth_date = self._calculate_birth_date(patient_data.get("Age"))

        return FHIRPatient(
            id=patient_id,
            identifier=[
                {
                    "use": "usual",
                    "system": f"http://{self.organization_id}/patient-id",
                    "value": patient_id,
                }
            ],
            active=True,
            name=[
                {"use": "official", "family": f"Patient_{patient_id[-6:]}", "given": ["Anonymous"]}
            ],
            gender=gender,
            birth_date=birth_date,
            address=[{"use": "home", "type": "both", "country": "US"}],
            telecom=[
                {
                    "system": "email",
                    "value": f"patient.{patient_id}@{self.organization_id}.com",
                    "use": "home",
                }
            ],
        )

    def assessment_to_diagnostic_report(
        self, assessment: Dict[str, Any], patient_id: str
    ) -> FHIRDiagnosticReport:
        """Convert risk assessment to FHIR DiagnosticReport"""

        report_id = str(uuid.uuid4())
        condition = assessment.get("condition", "unknown")

        # Map condition to LOINC codes
        loinc_codes = {
            "alzheimer": {"code": "72133-2", "display": "Alzheimer disease assessment"},
            "cardiovascular": {"code": "72136-5", "display": "Cardiovascular risk assessment"},
            "diabetes": {"code": "33747-0", "display": "Diabetes risk assessment"},
            "stroke": {"code": "72104-3", "display": "Stroke risk assessment"},
        }

        condition_code = loinc_codes.get(
            condition, {"code": "418799008", "display": "Medical risk assessment"}
        )

        return FHIRDiagnosticReport(
            id=report_id,
            status="final",
            category=[
                {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/v2-0074",
                            "code": "LAB",
                            "display": "Laboratory",
                        }
                    ]
                }
            ],
            code={
                "coding": [
                    {
                        "system": "http://loinc.org",
                        "code": condition_code["code"],
                        "display": condition_code["display"],
                    }
                ]
            },
            subject={"reference": f"Patient/{patient_id}"},
            effective_datetime=datetime.now().isoformat(),
            issued=datetime.now().isoformat(),
            performer=[
                {
                    "reference": f"Organization/{self.organization_id}",
                    "display": "DuetMind Adaptive AI System",
                }
            ],
            result=[],  # Will be populated with observations
            conclusion=self._format_conclusion(assessment),
            conclusion_code=[
                {
                    "coding": [
                        {
                            "system": "http://snomed.info/sct",
                            "code": self._get_risk_level_code(assessment.get("risk_level", "LOW")),
                            "display": f"{assessment.get('risk_level', 'LOW')} risk",
                        }
                    ]
                }
            ],
        )

    def create_risk_score_observation(
        self, assessment: Dict[str, Any], patient_id: str
    ) -> FHIRObservation:
        """Create FHIR Observation for risk score"""

        obs_id = str(uuid.uuid4())
        condition = assessment.get("condition", "unknown")

        return FHIRObservation(
            id=obs_id,
            status="final",
            category=[
                {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                            "code": "survey",
                            "display": "Survey",
                        }
                    ]
                }
            ],
            code={
                "coding": [
                    {
                        "system": "http://loinc.org",
                        "code": "72133-2",
                        "display": f"{condition.title()} risk score",
                    }
                ]
            },
            subject={"reference": f"Patient/{patient_id}"},
            effective_datetime=datetime.now().isoformat(),
            value={
                "value": assessment.get("risk_score", 0.0),
                "unit": "probability",
                "system": "http://unitsofmeasure.org",
                "code": "1",
            },
            performer=[
                {
                    "reference": f"Organization/{self.organization_id}",
                    "display": "DuetMind Adaptive AI System",
                }
            ],
            interpretation=[
                {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation",
                            "code": self._get_interpretation_code(
                                assessment.get("risk_level", "LOW")
                            ),
                            "display": assessment.get("risk_level", "LOW"),
                        }
                    ]
                }
            ],
            reference_range=[
                {
                    "low": {"value": 0.0, "unit": "probability"},
                    "high": {"value": 1.0, "unit": "probability"},
                    "text": "Normal range: 0.0 to 1.0",
                }
            ],
        )

    def _calculate_birth_date(self, age: Optional[int]) -> str:
        """Calculate birth date from age"""
        if age is None:
            return "1950-01-01"  # Default

        birth_year = datetime.now().year - age
        return f"{birth_year}-01-01"

    def _format_conclusion(self, assessment: Dict[str, Any]) -> str:
        """Format assessment conclusion"""
        condition = assessment.get("condition", "unknown").title()
        risk_level = assessment.get("risk_level", "LOW")
        risk_score = assessment.get("risk_score", 0.0)
        confidence = assessment.get("confidence", 0.0)

        return (
            f"{condition} risk assessment: {risk_level} risk level "
            f"(score: {risk_score:.3f}, confidence: {confidence:.3f}). "
            f"Recommended interventions: {', '.join(assessment.get('interventions', [])[:3])}"
        )

    def _get_risk_level_code(self, risk_level: str) -> str:
        """Get SNOMED code for risk level"""
        codes = {
            "HIGH": "75540009",  # High risk
            "MEDIUM": "371879000",  # Medium risk
            "LOW": "62482003",  # Low risk
            "MINIMAL": "62482003",  # Low risk
        }
        return codes.get(risk_level, "62482003")

    def _get_interpretation_code(self, risk_level: str) -> str:
        """Get interpretation code for risk level"""
        codes = {
            "HIGH": "H",  # High
            "MEDIUM": "N",  # Normal
            "LOW": "L",  # Low
            "MINIMAL": "L",  # Low
        }
        return codes.get(risk_level, "N")


class HL7MessageProcessor:
    """
    Processes HL7 messages for EHR integration.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.message_types = ["ADT", "ORU", "ORM"]  # Admit/Discharge, Results, Orders

    def parse_hl7_message(self, message: str) -> Dict[str, Any]:
        """Parse HL7 message into structured data"""

        lines = message.strip().split("\n")
        if not lines:
            raise ValueError("Empty HL7 message")

        # Parse MSH (Message Header) segment
        msh_segment = lines[0]
        if not msh_segment.startswith("MSH"):
            raise ValueError("Invalid HL7 message: Missing MSH segment")

        fields = msh_segment.split("|")
        message_type = fields[8] if len(fields) > 8 else "Unknown"

        parsed_data = {
            "message_type": message_type,
            "timestamp": fields[6] if len(fields) > 6 else datetime.now().isoformat(),
            "sending_application": fields[2] if len(fields) > 2 else "Unknown",
            "receiving_application": fields[4] if len(fields) > 4 else "DuetMind",
            "segments": {},
        }

        # Parse remaining segments
        for line in lines:
            if len(line) < 3:
                continue

            segment_type = line[:3]
            segment_data = self._parse_segment(line)

            if segment_type not in parsed_data["segments"]:
                parsed_data["segments"][segment_type] = []
            parsed_data["segments"][segment_type].append(segment_data)

        return parsed_data

    def create_ack_message(self, original_message: Dict[str, Any], status: str = "AA") -> str:
        """Create HL7 ACK (Acknowledgment) message"""

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        control_id = str(uuid.uuid4())[:8]

        msh_segment = (
            f"MSH|^~\\&|DuetMind|DUETMIND|"
            f"{original_message.get('sending_application', 'EHR')}|"
            f"EHR|{timestamp}||ACK|{control_id}|P|2.5"
        )

        msa_segment = f"MSA|{status}|{control_id}|Message received and processed"

        return f"{msh_segment}\n{msa_segment}"

    def extract_patient_data(self, hl7_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract patient data from parsed HL7 message"""

        patient_data = {}

        # Extract from PID (Patient Identification) segment
        pid_segments = hl7_data.get("segments", {}).get("PID", [])
        if pid_segments:
            pid = pid_segments[0]
            patient_data.update(
                {
                    "patient_id": pid.get("patient_id", str(uuid.uuid4())),
                    "family_name": pid.get("family_name", "Unknown"),
                    "given_name": pid.get("given_name", "Unknown"),
                    "birth_date": pid.get("birth_date"),
                    "gender": pid.get("gender"),
                    "address": pid.get("address"),
                }
            )

        # Extract observations from OBX segments
        obx_segments = hl7_data.get("segments", {}).get("OBX", [])
        observations = {}

        for obx in obx_segments:
            obs_id = obx.get("observation_id")
            obs_value = obx.get("observation_value")
            if obs_id and obs_value:
                observations[obs_id] = obs_value

        patient_data["observations"] = observations

        return patient_data

    def _parse_segment(self, segment: str) -> Dict[str, Any]:
        """Parse individual HL7 segment"""

        fields = segment.split("|")
        segment_type = fields[0]

        if segment_type == "PID":
            return self._parse_pid_segment(fields)
        elif segment_type == "OBX":
            return self._parse_obx_segment(fields)
        elif segment_type == "MSH":
            return self._parse_msh_segment(fields)
        else:
            return {"type": segment_type, "fields": fields}

    def _parse_pid_segment(self, fields: List[str]) -> Dict[str, Any]:
        """Parse PID (Patient Identification) segment"""
        return {
            "patient_id": fields[3] if len(fields) > 3 else None,
            "family_name": fields[5].split("^")[0] if len(fields) > 5 else None,
            "given_name": fields[5].split("^")[1] if len(fields) > 5 and "^" in fields[5] else None,
            "birth_date": fields[7] if len(fields) > 7 else None,
            "gender": fields[8] if len(fields) > 8 else None,
            "address": fields[11] if len(fields) > 11 else None,
        }

    def _parse_obx_segment(self, fields: List[str]) -> Dict[str, Any]:
        """Parse OBX (Observation) segment"""
        return {
            "observation_id": fields[3] if len(fields) > 3 else None,
            "observation_value": fields[5] if len(fields) > 5 else None,
            "units": fields[6] if len(fields) > 6 else None,
            "reference_range": fields[7] if len(fields) > 7 else None,
            "abnormal_flags": fields[8] if len(fields) > 8 else None,
        }

    def _parse_msh_segment(self, fields: List[str]) -> Dict[str, Any]:
        """Parse MSH (Message Header) segment"""
        return {
            "sending_application": fields[2] if len(fields) > 2 else None,
            "receiving_application": fields[4] if len(fields) > 4 else None,
            "timestamp": fields[6] if len(fields) > 6 else None,
            "message_type": fields[8] if len(fields) > 8 else None,
            "message_control_id": fields[9] if len(fields) > 9 else None,
        }


class EHRConnector:
    """
    Main EHR integration connector that handles bi-directional data exchange.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.fhir_converter = FHIRConverter(config)
        self.hl7_processor = HL7MessageProcessor(config)
        self.medical_processor = SecureMedicalDataProcessor(config)
        self.message_queue = []
        self.sync_log = []

    def ingest_patient_data(
        self, data: Union[str, Dict[str, Any]], format_type: str = "json"
    ) -> Dict[str, Any]:
        """
        Ingest patient data from EHR in various formats.

        Args:
            data: Patient data (HL7 message, FHIR JSON, or dict)
            format_type: 'hl7' or 'fhir' or 'json'

        Returns:
            Standardized patient data dict
        """

        try:
            if format_type == "hl7":
                hl7_data = self.hl7_processor.parse_hl7_message(data)
                patient_data = self.hl7_processor.extract_patient_data(hl7_data)

            elif format_type == "fhir":
                patient_data = self._extract_from_fhir(data)

            else:  # json
                patient_data = data if isinstance(data, dict) else json.loads(data)

            # Standardize data format
            standardized_data = self._standardize_patient_data(patient_data)

            # Log ingestion
            self._log_data_ingestion(standardized_data, format_type)

            return standardized_data

        except Exception as e:
            logger.error(f"Data ingestion failed: {e}")
            raise

    def export_assessment_results(
        self, assessment_results: Dict[str, Any], patient_id: str, format_type: str = "fhir"
    ) -> Union[str, Dict[str, Any]]:
        """
        Export assessment results to EHR-compatible format.

        Args:
            assessment_results: Risk assessment results
            patient_id: Patient identifier
            format_type: 'fhir' or 'hl7'

        Returns:
            Formatted results for EHR integration
        """

        try:
            if format_type == "fhir":
                return self._export_to_fhir(assessment_results, patient_id)
            elif format_type == "hl7":
                return self._export_to_hl7(assessment_results, patient_id)
            else:
                raise ValueError(f"Unsupported export format: {format_type}")

        except Exception as e:
            logger.error(f"Assessment export failed: {e}")
            raise

    def sync_with_ehr(self, ehr_endpoint: str, patient_id: str) -> Dict[str, Any]:
        """
        Perform bi-directional sync with EHR system.

        Args:
            ehr_endpoint: EHR system endpoint URL
            patient_id: Patient to sync

        Returns:
            Sync status and results
        """

        sync_result = {
            "patient_id": patient_id,
            "sync_timestamp": datetime.now().isoformat(),
            "status": "pending",
            "operations": [],
            "errors": [],
        }

        try:
            # 1. Fetch latest data from EHR (simulated)
            latest_data = self._fetch_from_ehr(ehr_endpoint, patient_id)
            sync_result["operations"].append("data_fetch")

            # 2. Check for updates needed
            updates_needed = self._check_for_updates(patient_id, latest_data)

            # 3. Send assessment results if needed
            if updates_needed:
                export_result = self._send_to_ehr(ehr_endpoint, patient_id)
                sync_result["operations"].append("data_export")

            sync_result["status"] = "completed"

        except Exception as e:
            sync_result["status"] = "failed"
            sync_result["errors"].append(str(e))
            logger.error(f"EHR sync failed: {e}")

        # Log sync operation
        self.sync_log.append(sync_result)

        return sync_result

    def _extract_from_fhir(self, fhir_data: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Extract patient data from FHIR resource"""

        if isinstance(fhir_data, str):
            fhir_data = json.loads(fhir_data)

        patient_data = {}

        if fhir_data.get("resourceType") == "Patient":
            patient_data["patient_id"] = fhir_data.get("id")
            patient_data["gender"] = fhir_data.get("gender")

            # Extract name
            names = fhir_data.get("name", [])
            if names:
                name = names[0]
                patient_data["family_name"] = name.get("family")
                patient_data["given_name"] = " ".join(name.get("given", []))

            # Calculate age from birth date
            birth_date = fhir_data.get("birthDate")
            if birth_date:
                birth_year = int(birth_date[:4])
                patient_data["Age"] = datetime.now().year - birth_year

        return patient_data

    def _standardize_patient_data(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Standardize patient data to internal format"""

        standardized = {}

        # Patient ID
        standardized["patient_id"] = patient_data.get("patient_id", str(uuid.uuid4()))

        # Age
        if "Age" in patient_data:
            standardized["Age"] = patient_data["Age"]
        elif "birth_date" in patient_data:
            birth_year = int(patient_data["birth_date"][:4])
            standardized["Age"] = datetime.now().year - birth_year

        # Gender (convert to internal format)
        gender = patient_data.get("gender", "").lower()
        if gender in ["f", "female", "0"]:
            standardized["M/F"] = 0
        elif gender in ["m", "male", "1"]:
            standardized["M/F"] = 1

        # Copy observations directly
        observations = patient_data.get("observations", {})
        standardized.update(observations)

        # Copy other relevant fields
        for field in ["MMSE", "CDR", "EDUC", "nWBV", "eTIV", "ASF", "SES"]:
            if field in patient_data:
                standardized[field] = patient_data[field]

        return standardized

    def _export_to_fhir(
        self, assessment_results: Dict[str, Any], patient_id: str
    ) -> Dict[str, Any]:
        """Export assessment results as FHIR resources"""

        bundle = {
            "resourceType": "Bundle",
            "id": str(uuid.uuid4()),
            "type": "collection",
            "timestamp": datetime.now().isoformat(),
            "entry": [],
        }

        # Create diagnostic report for each assessment
        for condition, assessment in assessment_results.items():
            if hasattr(assessment, "__dict__"):
                assessment_dict = assessment.__dict__
            else:
                assessment_dict = assessment

            # Create diagnostic report
            diagnostic_report = self.fhir_converter.assessment_to_diagnostic_report(
                assessment_dict, patient_id
            )

            bundle["entry"].append({"resource": diagnostic_report.to_fhir_json()})

            # Create risk score observation
            risk_obs = self.fhir_converter.create_risk_score_observation(
                assessment_dict, patient_id
            )

            bundle["entry"].append({"resource": risk_obs.to_fhir_json()})

        return bundle

    def _export_to_hl7(self, assessment_results: Dict[str, Any], patient_id: str) -> str:
        """Export assessment results as HL7 ORU message"""

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        control_id = str(uuid.uuid4())[:8]

        # MSH segment
        msh = (
            f"MSH|^~\\&|DuetMind|DUETMIND|EHR|HOSPITAL|{timestamp}||" f"ORU^R01|{control_id}|P|2.5"
        )

        # PID segment
        pid = f"PID|||{patient_id}||Patient^Anonymous||19700101|U"

        # OBR segment (observation request)
        obr = f"OBR|1||{control_id}|^Risk Assessment^L|||{timestamp}"

        segments = [msh, pid, obr]

        # OBX segments for each assessment
        obx_sequence = 1
        for condition, assessment in assessment_results.items():
            if hasattr(assessment, "__dict__"):
                assessment_dict = assessment.__dict__
            else:
                assessment_dict = assessment

            risk_score = assessment_dict.get("risk_score", 0.0)
            risk_level = assessment_dict.get("risk_level", "LOW")

            obx = (
                f"OBX|{obx_sequence}|NM|^{condition.title()} Risk Score^L||"
                f"{risk_score}|probability||{risk_level}|||F"
            )
            segments.append(obx)
            obx_sequence += 1

        return "\n".join(segments)

    def _log_data_ingestion(self, patient_data: Dict[str, Any], format_type: str):
        """Log data ingestion for audit purposes"""

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "operation": "data_ingestion",
            "format_type": format_type,
            "patient_id": patient_data.get("patient_id", "unknown"),
            "data_hash": hashlib.sha256(
                json.dumps(patient_data, sort_keys=True).encode()
            ).hexdigest()[:16],
            "fields_count": len(patient_data),
        }

        logger.info(f"Data ingestion: {log_entry}")

    def _fetch_from_ehr(self, ehr_endpoint: str, patient_id: str) -> Dict[str, Any]:
        """Fetch latest data from EHR (simulated)"""

        # This would be replaced with actual EHR API calls
        logger.info(f"Fetching data from EHR {ehr_endpoint} for patient {patient_id}")

        # Simulated response
        return {"patient_id": patient_id, "last_updated": datetime.now().isoformat(), "data": {}}

    def _check_for_updates(self, patient_id: str, latest_data: Dict[str, Any]) -> bool:
        """Check if updates need to be sent to EHR"""

        # Simple check - in practice, would compare timestamps and data hashes
        return True

    def _send_to_ehr(self, ehr_endpoint: str, patient_id: str) -> Dict[str, Any]:
        """Send assessment results to EHR (simulated)"""

        logger.info(f"Sending data to EHR {ehr_endpoint} for patient {patient_id}")

        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "patient_id": patient_id,
        }


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Example configuration
    config = {"organization_id": "duetmind-adaptive", "master_password": "test_password"}

    # Create EHR connector
    ehr_connector = EHRConnector(config)

    # Example HL7 message
    hl7_message = r"""MSH|^~\&|EHR|HOSPITAL|DuetMind|DUETMIND|20241201120000||ADT^A01|12345|P|2.5
PID|||PATIENT_001||Doe^Jane||19450515|F||||||||||
OBX|1|NM|MMSE^Mini Mental State Exam^L||22|score||||F
OBX|2|NM|CDR^Clinical Dementia Rating^L||0.5|rating||||F"""

    # Test HL7 ingestion
    patient_data = ehr_connector.ingest_patient_data(hl7_message, "hl7")
    print("=== EHR Integration Test ===")
    print(f"Ingested patient data: {patient_data}")

    # Example FHIR patient resource
    fhir_patient = {
        "resourceType": "Patient",
        "id": "PATIENT_002",
        "gender": "female",
        "birthDate": "1945-05-15",
        "name": [{"family": "Smith", "given": ["Mary"]}],
    }

    # Test FHIR ingestion
    fhir_data = ehr_connector.ingest_patient_data(fhir_patient, "fhir")
    print(f"FHIR patient data: {fhir_data}")

    # Mock assessment results
    assessment_results = {
        "alzheimer": {
            "risk_score": 0.65,
            "risk_level": "MEDIUM",
            "condition": "alzheimer",
            "confidence": 0.8,
            "interventions": ["cognitive_training", "monitoring"],
        }
    }

    # Test FHIR export
    fhir_bundle = ehr_connector.export_assessment_results(assessment_results, "PATIENT_001", "fhir")
    print(f"FHIR export completed: {len(fhir_bundle['entry'])} resources")

    # Test HL7 export
    hl7_result = ehr_connector.export_assessment_results(assessment_results, "PATIENT_001", "hl7")
    print(f"HL7 export: {len(hl7_result.split('|'))} segments")
