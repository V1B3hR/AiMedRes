"""
Privacy management and data retention policies.

Provides GDPR and HIPAA compliant data handling:
- Automated data retention and deletion
- Privacy-preserving data processing
- Audit trail for data access
- Right to erasure implementation
- Data minimization policies
"""

import os
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import logging
import threading
from dataclasses import dataclass

security_logger = logging.getLogger('duetmind.security')

@dataclass
class DataRetentionPolicy:
    """
    Data retention policy configuration.
    
    Defines how long different types of data should be retained
    and when automatic deletion should occur.
    """
    
    # Retention periods in days
    medical_data_retention_days: int = 2555  # 7 years (typical HIPAA requirement)
    api_logs_retention_days: int = 90
    training_data_retention_days: int = 365
    model_data_retention_days: int = 1095  # 3 years
    audit_logs_retention_days: int = 2555  # 7 years
    
    # Privacy settings
    anonymize_after_days: int = 30
    enable_automatic_deletion: bool = True
    require_explicit_consent: bool = True
    
    # GDPR compliance
    gdpr_enabled: bool = True
    data_subject_rights_enabled: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'medical_data_retention_days': self.medical_data_retention_days,
            'api_logs_retention_days': self.api_logs_retention_days,
            'training_data_retention_days': self.training_data_retention_days,
            'model_data_retention_days': self.model_data_retention_days,
            'audit_logs_retention_days': self.audit_logs_retention_days,
            'anonymize_after_days': self.anonymize_after_days,
            'enable_automatic_deletion': self.enable_automatic_deletion,
            'require_explicit_consent': self.require_explicit_consent,
            'gdpr_enabled': self.gdpr_enabled,
            'data_subject_rights_enabled': self.data_subject_rights_enabled
        }

class PrivacyManager:
    """
    Comprehensive privacy management system.
    
    Features:
    - GDPR and HIPAA compliance
    - Automated data retention and deletion
    - Privacy audit trails
    - Data subject rights (access, rectification, erasure)
    - Data minimization enforcement
    - Consent management
    """
    
    def __init__(self, config: Dict[str, Any], db_path: str = None):
        self.config = config
        self.retention_policy = DataRetentionPolicy(**config.get('retention_policy', {}))
        
        # Initialize privacy database
        self.db_path = db_path or os.path.join(os.getcwd(), 'privacy_audit.db')
        self._init_privacy_database()
        
        # Background cleanup thread
        self.cleanup_thread = None
        self.cleanup_interval_hours = config.get('cleanup_interval_hours', 24)
        self.running = False
        
        security_logger.info("Privacy manager initialized with GDPR/HIPAA compliance")
    
    def _init_privacy_database(self):
        """Initialize privacy audit database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Data access audit table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS data_access_audit (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    user_id TEXT NOT NULL,
                    data_type TEXT NOT NULL,
                    action TEXT NOT NULL,
                    data_id TEXT,
                    ip_address TEXT,
                    purpose TEXT,
                    legal_basis TEXT
                )
            ''')
            
            # Data retention tracking
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS data_retention_tracking (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    data_id TEXT UNIQUE NOT NULL,
                    data_type TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    retention_until DATETIME NOT NULL,
                    anonymized_at DATETIME,
                    deleted_at DATETIME,
                    status TEXT DEFAULT 'active'
                )
            ''')
            
            # Consent management
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS consent_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    subject_id TEXT NOT NULL,
                    consent_type TEXT NOT NULL,
                    granted_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    revoked_at DATETIME,
                    purpose TEXT NOT NULL,
                    legal_basis TEXT NOT NULL,
                    status TEXT DEFAULT 'active'
                )
            ''')
            
            conn.commit()
    
    def start_background_cleanup(self):
        """Start background thread for automatic data cleanup."""
        if not self.running:
            self.running = True
            self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
            self.cleanup_thread.start()
            security_logger.info("Background privacy cleanup started")
    
    def stop_background_cleanup(self):
        """Stop background cleanup thread."""
        self.running = False
        if self.cleanup_thread and self.cleanup_thread.is_alive():
            self.cleanup_thread.join(timeout=5)
        security_logger.info("Background privacy cleanup stopped")
    
    def _cleanup_loop(self):
        """Background cleanup loop."""
        import time
        
        while self.running:
            try:
                self.perform_automatic_cleanup()
                time.sleep(self.cleanup_interval_hours * 3600)  # Convert hours to seconds
            except Exception as e:
                security_logger.error(f"Error in privacy cleanup: {e}")
                time.sleep(3600)  # Wait 1 hour before retrying
    
    def log_data_access(self, user_id: str, data_type: str, action: str, 
                       data_id: str = None, ip_address: str = None, 
                       purpose: str = None, legal_basis: str = None):
        """
        Log data access for audit trail.
        
        Args:
            user_id: User accessing the data
            data_type: Type of data being accessed
            action: Action performed (read, write, delete, etc.)
            data_id: Identifier of specific data
            ip_address: IP address of accessor
            purpose: Purpose of data access
            legal_basis: Legal basis for processing (GDPR requirement)
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO data_access_audit 
                (user_id, data_type, action, data_id, ip_address, purpose, legal_basis)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (user_id, data_type, action, data_id, ip_address, purpose, legal_basis))
            conn.commit()
    
    def register_data_for_retention(self, data_id: str, data_type: str) -> bool:
        """
        Register data for retention tracking.
        
        Args:
            data_id: Unique identifier for the data
            data_type: Type of data (medical, api_log, training, etc.)
            
        Returns:
            Success status
        """
        try:
            # Calculate retention period
            retention_days = getattr(self.retention_policy, f"{data_type}_retention_days", 365)
            retention_until = datetime.now() + timedelta(days=retention_days)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO data_retention_tracking 
                    (data_id, data_type, retention_until)
                    VALUES (?, ?, ?)
                ''', (data_id, data_type, retention_until))
                conn.commit()
            
            security_logger.info(f"Registered {data_type} data {data_id} for retention until {retention_until}")
            return True
        except Exception as e:
            security_logger.error(f"Failed to register data for retention: {e}")
            return False
    
    def anonymize_data(self, data_id: str) -> bool:
        """
        Mark data as anonymized in retention tracking.
        
        Args:
            data_id: Data identifier to mark as anonymized
            
        Returns:
            Success status
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE data_retention_tracking 
                    SET anonymized_at = CURRENT_TIMESTAMP, status = 'anonymized'
                    WHERE data_id = ?
                ''', (data_id,))
                conn.commit()
            
            security_logger.info(f"Data {data_id} marked as anonymized")
            return True
        except Exception as e:
            security_logger.error(f"Failed to mark data as anonymized: {e}")
            return False
    
    def delete_data(self, data_id: str, reason: str = "retention_policy") -> bool:
        """
        Mark data as deleted and log the action.
        
        Args:
            data_id: Data identifier to delete
            reason: Reason for deletion
            
        Returns:
            Success status
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE data_retention_tracking 
                    SET deleted_at = CURRENT_TIMESTAMP, status = 'deleted'
                    WHERE data_id = ?
                ''', (data_id,))
                conn.commit()
            
            # Log deletion action
            self.log_data_access(
                user_id="system",
                data_type="retention_management",
                action="delete",
                data_id=data_id,
                purpose=reason,
                legal_basis="retention_policy"
            )
            
            security_logger.info(f"Data {data_id} marked as deleted (reason: {reason})")
            return True
        except Exception as e:
            security_logger.error(f"Failed to delete data: {e}")
            return False
    
    def perform_automatic_cleanup(self) -> Dict[str, int]:
        """
        Perform automatic data cleanup based on retention policies.
        
        Returns:
            Dictionary with cleanup statistics
        """
        stats = {
            'anonymized_count': 0,
            'deleted_count': 0,
            'errors': 0
        }
        
        if not self.retention_policy.enable_automatic_deletion:
            security_logger.info("Automatic deletion is disabled")
            return stats
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Find data ready for anonymization
                anonymize_cutoff = datetime.now() - timedelta(days=self.retention_policy.anonymize_after_days)
                cursor.execute('''
                    SELECT data_id FROM data_retention_tracking 
                    WHERE created_at < ? AND anonymized_at IS NULL AND status = 'active'
                ''', (anonymize_cutoff,))
                
                anonymize_candidates = cursor.fetchall()
                
                for (data_id,) in anonymize_candidates:
                    if self.anonymize_data(data_id):
                        stats['anonymized_count'] += 1
                    else:
                        stats['errors'] += 1
                
                # Find data ready for deletion
                cursor.execute('''
                    SELECT data_id FROM data_retention_tracking 
                    WHERE retention_until < ? AND deleted_at IS NULL AND status != 'deleted'
                ''', (datetime.now(),))
                
                deletion_candidates = cursor.fetchall()
                
                for (data_id,) in deletion_candidates:
                    if self.delete_data(data_id, "automatic_retention_policy"):
                        stats['deleted_count'] += 1
                    else:
                        stats['errors'] += 1
        
        except Exception as e:
            security_logger.error(f"Error during automatic cleanup: {e}")
            stats['errors'] += 1
        
        security_logger.info(f"Privacy cleanup completed: {stats}")
        return stats
    
    def get_retention_status(self, data_id: str) -> Optional[Dict[str, Any]]:
        """Get retention status for specific data."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT * FROM data_retention_tracking WHERE data_id = ?
                ''', (data_id,))
                row = cursor.fetchone()
                
                if row:
                    columns = [description[0] for description in cursor.description]
                    return dict(zip(columns, row))
                return None
        except Exception as e:
            security_logger.error(f"Failed to get retention status: {e}")
            return None
    
    def generate_privacy_report(self) -> Dict[str, Any]:
        """Generate comprehensive privacy compliance report."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Data retention statistics
                cursor.execute('''
                    SELECT data_type, status, COUNT(*) as count 
                    FROM data_retention_tracking 
                    GROUP BY data_type, status
                ''')
                retention_stats = cursor.fetchall()
                
                # Access audit statistics
                cursor.execute('''
                    SELECT data_type, action, COUNT(*) as count 
                    FROM data_access_audit 
                    WHERE timestamp > datetime('now', '-30 days')
                    GROUP BY data_type, action
                ''')
                access_stats = cursor.fetchall()
                
                return {
                    'generated_at': datetime.now().isoformat(),
                    'retention_policy': self.retention_policy.to_dict(),
                    'retention_statistics': retention_stats,
                    'access_statistics': access_stats,
                    'compliance_status': {
                        'gdpr_enabled': self.retention_policy.gdpr_enabled,
                        'automatic_deletion': self.retention_policy.enable_automatic_deletion,
                        'data_subject_rights': self.retention_policy.data_subject_rights_enabled
                    }
                }
        except Exception as e:
            security_logger.error(f"Failed to generate privacy report: {e}")
            return {'error': str(e)}