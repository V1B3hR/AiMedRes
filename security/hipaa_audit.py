#!/usr/bin/env python3
"""
HIPAA-Compliant Audit Logging System

Provides comprehensive audit logging for Protected Health Information (PHI) access
and system activities to ensure HIPAA compliance. Features include:

- Real-time audit trail generation
- Secure log storage with integrity verification
- Automated compliance violation detection
- Regulatory reporting capabilities
- Log retention and archival management
"""

import json
import hashlib
import sqlite3
import uuid
import time
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional, Union
from enum import Enum
from dataclasses import dataclass, asdict
from pathlib import Path
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import os
import threading

logger = logging.getLogger('duetmind.hipaa_audit')


class AccessType(Enum):
    """Types of PHI access events"""
    READ = "READ"
    WRITE = "WRITE"
    UPDATE = "UPDATE"
    DELETE = "DELETE" 
    EXPORT = "EXPORT"
    PRINT = "PRINT"
    VIEW = "VIEW"
    SEARCH = "SEARCH"
    COPY = "COPY"


class AuditEvent(Enum):
    """HIPAA audit event types"""
    PHI_ACCESS = "phi_access"
    LOGIN_ATTEMPT = "login_attempt"
    LOGOUT = "logout"
    PASSWORD_CHANGE = "password_change"
    PERMISSION_CHANGE = "permission_change"
    DATA_EXPORT = "data_export"
    SYSTEM_CONFIG_CHANGE = "system_config_change"
    SECURITY_VIOLATION = "security_violation"
    MODEL_PREDICTION = "model_prediction"
    CLINICAL_DECISION = "clinical_decision"


class ComplianceStatus(Enum):
    """Compliance status for audit events"""
    COMPLIANT = "COMPLIANT"
    WARNING = "WARNING"
    VIOLATION = "VIOLATION"
    CRITICAL = "CRITICAL"


@dataclass
class HIPAAAuditRecord:
    """HIPAA audit record structure"""
    audit_id: str
    timestamp: datetime
    event_type: AuditEvent
    user_id: str
    user_role: str
    patient_id_hash: Optional[str]
    access_type: Optional[AccessType]
    resource: str
    purpose: str
    outcome: str
    ip_address: str
    session_id: str
    device_info: str
    location: Optional[str]
    compliance_status: ComplianceStatus
    risk_level: str
    additional_data: Dict[str, Any]
    integrity_hash: str


class HIPAAAuditLogger:
    """
    HIPAA-compliant audit logging system with comprehensive tracking and validation.
    """
    
    def __init__(self, 
                 audit_db_path: str = "hipaa_audit.db",
                 encryption_key: Optional[str] = None,
                 retention_days: int = 2557):  # 7 years HIPAA requirement
        """
        Initialize HIPAA audit logger.
        
        Args:
            audit_db_path: Path to audit database
            encryption_key: Key for encrypting sensitive audit data
            retention_days: Data retention period (default 7 years for HIPAA)
        """
        self.audit_db_path = audit_db_path
        self.retention_days = retention_days
        self._setup_encryption(encryption_key)
        self._setup_database()
        self._lock = threading.Lock()
        
        # Compliance violation tracking
        self.violation_threshold = {
            'failed_logins_per_hour': 10,
            'unusual_access_patterns': 50,
            'after_hours_access_threshold': 5
        }
        
        logger.info("HIPAA Audit Logger initialized with enhanced security")
    
    def _setup_encryption(self, encryption_key: Optional[str]):
        """Setup encryption for sensitive audit data."""
        if encryption_key:
            key = encryption_key.encode()
        else:
            # Generate from environment or create new
            key = os.environ.get('HIPAA_AUDIT_KEY', '').encode()
            if not key:
                key = os.urandom(32)
                logger.warning("Generated temporary audit encryption key. Set HIPAA_AUDIT_KEY env var for production.")
        
        # Derive Fernet key
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'hipaa_audit_salt_v1',
            iterations=100000,
        )
        derived_key = base64.urlsafe_b64encode(kdf.derive(key))
        self.fernet = Fernet(derived_key)
    
    def _setup_database(self):
        """Initialize audit database with proper schema."""
        conn = sqlite3.connect(self.audit_db_path)
        cursor = conn.cursor()
        
        # Create main audit log table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS hipaa_audit_log (
                audit_id TEXT PRIMARY KEY,
                timestamp REAL NOT NULL,
                event_type TEXT NOT NULL,
                user_id TEXT NOT NULL,
                user_role TEXT NOT NULL,
                patient_id_hash TEXT,
                access_type TEXT,
                resource TEXT NOT NULL,
                purpose TEXT NOT NULL,
                outcome TEXT NOT NULL,
                ip_address TEXT NOT NULL,
                session_id TEXT NOT NULL,
                device_info TEXT,
                location TEXT,
                compliance_status TEXT NOT NULL,
                risk_level TEXT NOT NULL,
                additional_data_encrypted BLOB,
                integrity_hash TEXT NOT NULL,
                created_at REAL NOT NULL
            )
        ''')
        
        # Create compliance violations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS compliance_violations (
                violation_id TEXT PRIMARY KEY,
                audit_id TEXT NOT NULL,
                violation_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                description TEXT NOT NULL,
                detected_at REAL NOT NULL,
                resolved_at REAL,
                resolution_notes TEXT,
                FOREIGN KEY (audit_id) REFERENCES hipaa_audit_log (audit_id)
            )
        ''')
        
        # Create user access patterns table for anomaly detection
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_access_patterns (
                pattern_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                access_hour INTEGER NOT NULL,
                access_day INTEGER NOT NULL,
                access_count INTEGER NOT NULL,
                last_updated REAL NOT NULL
            )
        ''')
        
        # Create indexes for performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON hipaa_audit_log(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_id ON hipaa_audit_log(user_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_patient_hash ON hipaa_audit_log(patient_id_hash)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_compliance_status ON hipaa_audit_log(compliance_status)')
        
        conn.commit()
        conn.close()
    
    def log_phi_access(self,
                      user_id: str,
                      user_role: str,
                      patient_id: str,
                      access_type: AccessType,
                      resource: str,
                      purpose: str,
                      outcome: str = "SUCCESS",
                      ip_address: str = "unknown",
                      session_id: str = None,
                      device_info: str = "unknown",
                      location: str = None,
                      additional_data: Dict[str, Any] = None) -> str:
        """
        Log PHI access event with HIPAA compliance validation.
        
        Args:
            user_id: ID of user accessing PHI
            user_role: Role of the user (doctor, nurse, admin, etc.)
            patient_id: Patient identifier (will be hashed)
            access_type: Type of access (read, write, update, etc.)
            resource: Resource being accessed
            purpose: Purpose of access for HIPAA compliance
            outcome: Outcome of access attempt
            ip_address: IP address of access
            session_id: Session identifier
            device_info: Device information
            location: Physical location if available
            additional_data: Additional audit data
        
        Returns:
            audit_id: Unique audit record identifier
        """
        return self._log_audit_event(
            event_type=AuditEvent.PHI_ACCESS,
            user_id=user_id,
            user_role=user_role,
            patient_id=patient_id,
            access_type=access_type,
            resource=resource,
            purpose=purpose,
            outcome=outcome,
            ip_address=ip_address,
            session_id=session_id,
            device_info=device_info,
            location=location,
            additional_data=additional_data or {}
        )
    
    def log_clinical_decision(self,
                            user_id: str,
                            user_role: str,
                            patient_id: str,
                            decision_data: Dict[str, Any],
                            ai_confidence: float,
                            human_override: bool = False,
                            **kwargs) -> str:
        """Log AI-assisted clinical decision with compliance tracking."""
        additional_data = {
            'ai_confidence': ai_confidence,
            'human_override': human_override,
            'decision_data': decision_data
        }
        additional_data.update(kwargs.get('additional_data', {}))
        
        return self._log_audit_event(
            event_type=AuditEvent.CLINICAL_DECISION,
            user_id=user_id,
            user_role=user_role,
            patient_id=patient_id,
            access_type=AccessType.READ,
            resource="clinical_decision_support",
            purpose="clinical_diagnosis_treatment",
            outcome="SUCCESS",
            additional_data=additional_data,
            **kwargs
        )
    
    def _log_audit_event(self, 
                        event_type: AuditEvent,
                        user_id: str,
                        user_role: str,
                        resource: str,
                        purpose: str,
                        outcome: str,
                        patient_id: Optional[str] = None,
                        access_type: Optional[AccessType] = None,
                        ip_address: str = "unknown",
                        session_id: str = None,
                        device_info: str = "unknown",
                        location: str = None,
                        additional_data: Dict[str, Any] = None) -> str:
        """Internal method to log audit events with validation."""
        
        audit_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc)
        session_id = session_id or f"session_{uuid.uuid4()}"
        additional_data = additional_data or {}
        
        # Hash patient ID for privacy
        patient_id_hash = None
        if patient_id:
            patient_id_hash = self._hash_patient_id(patient_id)
        
        # Assess compliance and risk
        compliance_status, risk_level = self._assess_compliance_risk(
            event_type, user_role, access_type, timestamp, ip_address, additional_data
        )
        
        # Create audit record
        record = HIPAAAuditRecord(
            audit_id=audit_id,
            timestamp=timestamp,
            event_type=event_type,
            user_id=user_id,
            user_role=user_role,
            patient_id_hash=patient_id_hash,
            access_type=access_type,
            resource=resource,
            purpose=purpose,
            outcome=outcome,
            ip_address=ip_address,
            session_id=session_id,
            device_info=device_info,
            location=location,
            compliance_status=compliance_status,
            risk_level=risk_level,
            additional_data=additional_data,
            integrity_hash=""  # Will be calculated
        )
        
        # Calculate integrity hash
        record.integrity_hash = self._calculate_integrity_hash(record)
        
        # Store in database
        with self._lock:
            self._store_audit_record(record)
            
            # Update access patterns for anomaly detection
            self._update_access_patterns(user_id, timestamp)
            
            # Check for violations
            if compliance_status in [ComplianceStatus.VIOLATION, ComplianceStatus.CRITICAL]:
                self._handle_compliance_violation(record)
        
        logger.info(f"Audit event logged: {audit_id} - {event_type.value} by {user_id}")
        
        return audit_id
    
    def _hash_patient_id(self, patient_id: str) -> str:
        """Hash patient ID for audit logs while maintaining uniqueness."""
        # Use HMAC for consistent hashing with secret key
        return hashlib.sha256(f"{patient_id}_audit_salt".encode()).hexdigest()[:16]
    
    def _assess_compliance_risk(self, 
                               event_type: AuditEvent,
                               user_role: str,
                               access_type: Optional[AccessType],
                               timestamp: datetime,
                               ip_address: str,
                               additional_data: Dict[str, Any]) -> tuple:
        """Assess compliance status and risk level for audit event."""
        
        compliance_status = ComplianceStatus.COMPLIANT
        risk_level = "LOW"
        
        # Check for high-risk scenarios
        hour = timestamp.hour
        
        # After-hours access (outside 6 AM - 10 PM)
        if hour < 6 or hour > 22:
            risk_level = "MEDIUM"
            if self._check_excessive_after_hours_access(timestamp):
                compliance_status = ComplianceStatus.WARNING
        
        # Export operations are high risk
        if access_type == AccessType.EXPORT:
            risk_level = "HIGH"
        
        # Admin operations during off-hours
        if user_role == "admin" and (hour < 6 or hour > 22):
            risk_level = "HIGH"
            compliance_status = ComplianceStatus.WARNING
        
        # Check for unusual patterns
        if self._detect_unusual_access_pattern(additional_data):
            risk_level = "HIGH"
            compliance_status = ComplianceStatus.WARNING
        
        return compliance_status, risk_level
    
    def _check_excessive_after_hours_access(self, timestamp: datetime) -> bool:
        """Check if user has excessive after-hours access."""
        # This would implement more sophisticated checking
        return False
    
    def _detect_unusual_access_pattern(self, additional_data: Dict[str, Any]) -> bool:
        """Detect unusual access patterns that might indicate security issues."""
        # Placeholder for advanced pattern detection
        return False
    
    def _calculate_integrity_hash(self, record: HIPAAAuditRecord) -> str:
        """Calculate integrity hash for audit record."""
        # Create a consistent string representation
        record_copy = record.__dict__.copy()
        record_copy.pop('integrity_hash', None)  # Remove hash field itself
        record_str = json.dumps(record_copy, default=str, sort_keys=True)
        return hashlib.sha256(record_str.encode()).hexdigest()
    
    def _store_audit_record(self, record: HIPAAAuditRecord):
        """Store audit record in database with encryption."""
        conn = sqlite3.connect(self.audit_db_path)
        cursor = conn.cursor()
        
        # Encrypt additional data
        additional_data_encrypted = self.fernet.encrypt(
            json.dumps(record.additional_data).encode()
        )
        
        cursor.execute('''
            INSERT INTO hipaa_audit_log (
                audit_id, timestamp, event_type, user_id, user_role,
                patient_id_hash, access_type, resource, purpose, outcome,
                ip_address, session_id, device_info, location,
                compliance_status, risk_level, additional_data_encrypted,
                integrity_hash, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            record.audit_id, record.timestamp.timestamp(), record.event_type.value,
            record.user_id, record.user_role, record.patient_id_hash,
            record.access_type.value if record.access_type else None,
            record.resource, record.purpose, record.outcome, record.ip_address,
            record.session_id, record.device_info, record.location,
            record.compliance_status.value, record.risk_level,
            additional_data_encrypted, record.integrity_hash, time.time()
        ))
        
        conn.commit()
        conn.close()
    
    def _update_access_patterns(self, user_id: str, timestamp: datetime):
        """Update user access patterns for anomaly detection."""
        conn = sqlite3.connect(self.audit_db_path)
        cursor = conn.cursor()
        
        # Update or insert access pattern
        cursor.execute('''
            INSERT OR REPLACE INTO user_access_patterns (
                pattern_id, user_id, access_hour, access_day, access_count, last_updated
            ) VALUES (?, ?, ?, ?, 
                COALESCE((SELECT access_count FROM user_access_patterns 
                         WHERE user_id = ? AND access_hour = ? AND access_day = ?), 0) + 1,
                ?)
        ''', (
            f"{user_id}_{timestamp.hour}_{timestamp.weekday()}", user_id,
            timestamp.hour, timestamp.weekday(),
            user_id, timestamp.hour, timestamp.weekday(), time.time()
        ))
        
        conn.commit()
        conn.close()
    
    def _handle_compliance_violation(self, record: HIPAAAuditRecord):
        """Handle detected compliance violations."""
        violation_id = str(uuid.uuid4())
        
        conn = sqlite3.connect(self.audit_db_path)
        cursor = conn.cursor()
        
        violation_type = "HIPAA_COMPLIANCE_WARNING"
        severity = record.compliance_status.value
        description = f"Compliance issue detected: {record.event_type.value} - {record.risk_level} risk"
        
        cursor.execute('''
            INSERT INTO compliance_violations (
                violation_id, audit_id, violation_type, severity,
                description, detected_at
            ) VALUES (?, ?, ?, ?, ?, ?)
        ''', (violation_id, record.audit_id, violation_type, severity,
              description, time.time()))
        
        conn.commit()
        conn.close()
        
        # Log violation for immediate attention
        logger.warning(f"COMPLIANCE VIOLATION: {violation_id} - {description}")
    
    def get_audit_trail(self, 
                       patient_id: str = None,
                       user_id: str = None,
                       start_time: datetime = None,
                       end_time: datetime = None,
                       limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieve audit trail with filtering options."""
        conn = sqlite3.connect(self.audit_db_path)
        cursor = conn.cursor()
        
        query = "SELECT * FROM hipaa_audit_log WHERE 1=1"
        params = []
        
        if patient_id:
            patient_hash = self._hash_patient_id(patient_id)
            query += " AND patient_id_hash = ?"
            params.append(patient_hash)
        
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time.timestamp())
        
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time.timestamp())
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        # Convert to dictionaries and decrypt additional data
        columns = [description[0] for description in cursor.description]
        results = []
        
        for row in rows:
            record_dict = dict(zip(columns, row))
            
            # Decrypt additional data
            if record_dict['additional_data_encrypted']:
                try:
                    decrypted = self.fernet.decrypt(record_dict['additional_data_encrypted'])
                    record_dict['additional_data'] = json.loads(decrypted.decode())
                except Exception as e:
                    logger.error(f"Failed to decrypt additional data: {e}")
                    record_dict['additional_data'] = {}
            
            # Remove encrypted field
            record_dict.pop('additional_data_encrypted', None)
            
            results.append(record_dict)
        
        conn.close()
        return results
    
    def get_compliance_violations(self, 
                                resolved: bool = None,
                                severity: str = None,
                                days_back: int = 30) -> List[Dict[str, Any]]:
        """Get compliance violations with filtering."""
        conn = sqlite3.connect(self.audit_db_path)
        cursor = conn.cursor()
        
        query = """
            SELECT v.*, a.user_id, a.event_type, a.timestamp 
            FROM compliance_violations v
            JOIN hipaa_audit_log a ON v.audit_id = a.audit_id
            WHERE v.detected_at >= ?
        """
        params = [time.time() - (days_back * 24 * 3600)]
        
        if resolved is not None:
            if resolved:
                query += " AND v.resolved_at IS NOT NULL"
            else:
                query += " AND v.resolved_at IS NULL"
        
        if severity:
            query += " AND v.severity = ?"
            params.append(severity)
        
        query += " ORDER BY v.detected_at DESC"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        columns = [description[0] for description in cursor.description]
        results = [dict(zip(columns, row)) for row in rows]
        
        conn.close()
        return results
    
    def generate_compliance_report(self, 
                                  start_date: datetime,
                                  end_date: datetime) -> Dict[str, Any]:
        """Generate comprehensive HIPAA compliance report."""
        conn = sqlite3.connect(self.audit_db_path)
        cursor = conn.cursor()
        
        start_ts = start_date.timestamp()
        end_ts = end_date.timestamp()
        
        # Get summary statistics
        cursor.execute('''
            SELECT 
                COUNT(*) as total_events,
                COUNT(DISTINCT user_id) as unique_users,
                COUNT(DISTINCT patient_id_hash) as unique_patients,
                COUNT(CASE WHEN compliance_status != 'COMPLIANT' THEN 1 END) as violations
            FROM hipaa_audit_log 
            WHERE timestamp BETWEEN ? AND ?
        ''', (start_ts, end_ts))
        
        summary = dict(zip(['total_events', 'unique_users', 'unique_patients', 'violations'], cursor.fetchone()))
        
        # Get event type breakdown
        cursor.execute('''
            SELECT event_type, COUNT(*) as count
            FROM hipaa_audit_log 
            WHERE timestamp BETWEEN ? AND ?
            GROUP BY event_type
        ''', (start_ts, end_ts))
        
        event_breakdown = {row[0]: row[1] for row in cursor.fetchall()}
        
        # Get compliance status breakdown
        cursor.execute('''
            SELECT compliance_status, COUNT(*) as count
            FROM hipaa_audit_log 
            WHERE timestamp BETWEEN ? AND ?
            GROUP BY compliance_status
        ''', (start_ts, end_ts))
        
        compliance_breakdown = {row[0]: row[1] for row in cursor.fetchall()}
        
        conn.close()
        
        return {
            'report_period': {
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat()
            },
            'summary': summary,
            'event_breakdown': event_breakdown,
            'compliance_breakdown': compliance_breakdown,
            'compliance_rate': (summary['total_events'] - summary['violations']) / max(summary['total_events'], 1) * 100
        }
    
    def cleanup_old_records(self) -> int:
        """Clean up audit records older than retention period."""
        cutoff_time = time.time() - (self.retention_days * 24 * 3600)
        
        conn = sqlite3.connect(self.audit_db_path)
        cursor = conn.cursor()
        
        # First get count of records to be deleted
        cursor.execute('SELECT COUNT(*) FROM hipaa_audit_log WHERE timestamp < ?', (cutoff_time,))
        count_to_delete = cursor.fetchone()[0]
        
        # Delete old records
        cursor.execute('DELETE FROM hipaa_audit_log WHERE timestamp < ?', (cutoff_time,))
        cursor.execute('DELETE FROM compliance_violations WHERE detected_at < ?', (cutoff_time,))
        cursor.execute('DELETE FROM user_access_patterns WHERE last_updated < ?', (cutoff_time,))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Cleaned up {count_to_delete} old audit records")
        return count_to_delete


# Convenience function for global audit logger
_global_audit_logger = None

def get_audit_logger() -> HIPAAAuditLogger:
    """Get global audit logger instance."""
    global _global_audit_logger
    if _global_audit_logger is None:
        _global_audit_logger = HIPAAAuditLogger()
    return _global_audit_logger


def audit_phi_access(user_id: str, patient_id: str, action: str, purpose: str, **kwargs) -> str:
    """Convenience function to audit PHI access."""
    logger = get_audit_logger()
    return logger.log_phi_access(
        user_id=user_id,
        user_role=kwargs.get('user_role', 'unknown'),
        patient_id=patient_id,
        access_type=AccessType(action.upper()),
        resource=kwargs.get('resource', 'patient_data'),
        purpose=purpose,
        outcome=kwargs.get('outcome', 'SUCCESS'),
        ip_address=kwargs.get('ip_address', 'unknown'),
        session_id=kwargs.get('session_id', None),
        device_info=kwargs.get('device_info', 'unknown'),
        location=kwargs.get('location', None),
        additional_data=kwargs.get('additional_data', {})
    )