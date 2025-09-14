"""
Advanced Safety Monitoring for DuetMind Adaptive.

This module extends the existing SecurityMonitor with advanced safety features:
- Safety domains (system, data, model, interaction, clinical)
- Pluggable safety checks interface
- Structured safety event persistence
- Correlation tracking and event chain analysis
"""

import time
import threading
import sqlite3
import uuid
from datetime import datetime, timedelta
from collections import defaultdict, deque
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
from abc import ABC, abstractmethod
import logging
import json

safety_logger = logging.getLogger('duetmind.safety')


class SafetyDomain(Enum):
    """Safety monitoring domains."""
    SYSTEM = "system"  # Performance, resource usage, technical issues
    DATA = "data"  # Data quality, schema validation, completeness
    MODEL = "model"  # Confidence drift, calibration, accuracy
    INTERACTION = "interaction"  # Conversation rules, user interaction patterns
    CLINICAL = "clinical"  # Guideline adherence, medical accuracy


@dataclass
class SafetyFinding:
    """Represents a safety check finding."""
    domain: SafetyDomain
    check_name: str
    severity: str  # 'info', 'warning', 'critical', 'emergency'
    message: str
    value: Optional[float] = None
    threshold: Optional[float] = None
    correlation_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.metadata is None:
            self.metadata = {}


class ISafetyCheck(ABC):
    """Interface for pluggable safety checks."""
    
    @property
    @abstractmethod
    def domain(self) -> SafetyDomain:
        """Return the safety domain this check monitors."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this safety check."""
        pass
    
    @abstractmethod
    def run(self, context: Dict[str, Any], correlation_id: Optional[str] = None) -> List[SafetyFinding]:
        """
        Run the safety check and return findings.
        
        Args:
            context: Context data for the check (metrics, inputs, etc.)
            correlation_id: Optional correlation ID for tracing
            
        Returns:
            List of SafetyFinding objects
        """
        pass


class SafetyMonitor:
    """
    Advanced safety monitoring system for DuetMind Adaptive.
    
    Features:
    - Multiple safety domains with pluggable checks
    - Structured safety event persistence
    - Correlation tracking for event chains
    - Integration with existing SecurityMonitor
    - Graduated intervention recommendations
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.monitoring_enabled = config.get('safety_monitoring_enabled', True)
        
        # Safety checks registry
        self.safety_checks: Dict[SafetyDomain, List[ISafetyCheck]] = defaultdict(list)
        
        # Event storage
        self.safety_db = config.get('safety_db_path', 'safety_events.db')
        self._init_safety_db()
        
        # Event correlation
        self.correlation_chains = defaultdict(list)
        self.active_correlations = {}
        
        # Monitoring state
        self.monitor_thread = None
        self.running = False
        
        # Alert callbacks
        self.alert_callbacks: List[Callable] = []
        
        # Integration with existing SecurityMonitor
        self.security_monitor = None
        
        safety_logger.info("Advanced Safety Monitor initialized")
    
    def _init_safety_db(self):
        """Initialize safety events database."""
        with sqlite3.connect(self.safety_db) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS safety_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    domain TEXT NOT NULL,
                    check_name TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    message TEXT NOT NULL,
                    value REAL,
                    threshold REAL,
                    correlation_id TEXT,
                    metadata_json TEXT,
                    event_data TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS event_correlations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    correlation_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    metadata_json TEXT
                )
            """)
            
            # Create indexes for performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_safety_timestamp ON safety_events(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_safety_domain ON safety_events(domain)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_safety_correlation ON safety_events(correlation_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_correlation_id ON event_correlations(correlation_id)")
            
            conn.commit()
    
    def register_safety_check(self, safety_check: ISafetyCheck):
        """Register a safety check for a specific domain."""
        self.safety_checks[safety_check.domain].append(safety_check)
        safety_logger.info(f"Registered safety check '{safety_check.name}' for domain {safety_check.domain.value}")
    
    def unregister_safety_check(self, domain: SafetyDomain, check_name: str):
        """Unregister a safety check."""
        self.safety_checks[domain] = [
            check for check in self.safety_checks[domain] 
            if check.name != check_name
        ]
        safety_logger.info(f"Unregistered safety check '{check_name}' from domain {domain.value}")
    
    def create_correlation_id(self, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Create a new correlation ID for event tracking."""
        correlation_id = str(uuid.uuid4())
        
        with sqlite3.connect(self.safety_db) as conn:
            conn.execute("""
                INSERT INTO event_correlations (correlation_id, created_at, status, metadata_json)
                VALUES (?, ?, ?, ?)
            """, (
                correlation_id,
                datetime.now().isoformat(),
                'active',
                json.dumps(metadata or {})
            ))
            conn.commit()
        
        self.active_correlations[correlation_id] = {
            'created_at': datetime.now(),
            'metadata': metadata or {},
            'event_count': 0
        }
        
        return correlation_id
    
    def run_safety_checks(self, domain: Optional[SafetyDomain] = None, 
                         context: Optional[Dict[str, Any]] = None,
                         correlation_id: Optional[str] = None) -> List[SafetyFinding]:
        """
        Run safety checks for specified domain(s).
        
        Args:
            domain: Specific domain to check, or None for all domains
            context: Context data for checks
            correlation_id: Optional correlation ID for event tracking
            
        Returns:
            List of all safety findings
        """
        if not self.monitoring_enabled:
            return []
        
        if context is None:
            context = {}
        
        domains_to_check = [domain] if domain else list(SafetyDomain)
        all_findings = []
        
        for check_domain in domains_to_check:
            checks = self.safety_checks.get(check_domain, [])
            
            for safety_check in checks:
                try:
                    findings = safety_check.run(context, correlation_id)
                    all_findings.extend(findings)
                    
                    # Store findings in database
                    for finding in findings:
                        self._store_safety_finding(finding)
                        
                        # Update correlation tracking
                        if finding.correlation_id and finding.correlation_id in self.active_correlations:
                            self.active_correlations[finding.correlation_id]['event_count'] += 1
                            self.correlation_chains[finding.correlation_id].append(finding)
                    
                except Exception as e:
                    safety_logger.error(f"Error running safety check {safety_check.name}: {e}")
                    # Create error finding
                    error_finding = SafetyFinding(
                        domain=check_domain,
                        check_name=safety_check.name,
                        severity='warning',
                        message=f"Safety check failed: {str(e)}",
                        correlation_id=correlation_id,
                        metadata={'error': str(e)}
                    )
                    all_findings.append(error_finding)
                    self._store_safety_finding(error_finding)
        
        # Check for alert conditions
        self._evaluate_findings_for_alerts(all_findings)
        
        return all_findings
    
    def _store_safety_finding(self, finding: SafetyFinding):
        """Store a safety finding in the database."""
        with sqlite3.connect(self.safety_db) as conn:
            conn.execute("""
                INSERT INTO safety_events (
                    timestamp, domain, check_name, severity, message,
                    value, threshold, correlation_id, metadata_json, event_data
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                finding.timestamp.isoformat(),
                finding.domain.value,
                finding.check_name,
                finding.severity,
                finding.message,
                finding.value,
                finding.threshold,
                finding.correlation_id,
                json.dumps(finding.metadata or {}),
                json.dumps(asdict(finding), default=str)
            ))
            conn.commit()
    
    def _evaluate_findings_for_alerts(self, findings: List[SafetyFinding]):
        """Evaluate findings and trigger alerts if necessary."""
        if not findings:
            return
        
        # Count findings by severity
        severity_counts = defaultdict(int)
        critical_findings = []
        emergency_findings = []
        
        for finding in findings:
            severity_counts[finding.severity] += 1
            if finding.severity == 'critical':
                critical_findings.append(finding)
            elif finding.severity == 'emergency':
                emergency_findings.append(finding)
        
        # Trigger alerts based on severity
        if emergency_findings:
            self._trigger_safety_alert('EMERGENCY', emergency_findings, severity_counts)
        elif critical_findings:
            self._trigger_safety_alert('CRITICAL', critical_findings, severity_counts)
        elif severity_counts['warning'] >= 5:  # Multiple warnings
            self._trigger_safety_alert('WARNING', findings[-5:], severity_counts)
    
    def _trigger_safety_alert(self, level: str, findings: List[SafetyFinding], 
                             severity_counts: Dict[str, int]):
        """Trigger a safety alert."""
        alert_data = {
            'level': level,
            'timestamp': datetime.now().isoformat(),
            'findings_count': len(findings),
            'severity_breakdown': dict(severity_counts),
            'sample_findings': [
                {
                    'domain': f.domain.value,
                    'check': f.check_name,
                    'severity': f.severity,
                    'message': f.message,
                    'correlation_id': f.correlation_id
                }
                for f in findings[:3]  # Include first 3 findings
            ]
        }
        
        safety_logger.warning(f"SAFETY ALERT [{level}]: {json.dumps(alert_data, indent=2)}")
        
        # Call registered alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert_data)
            except Exception as e:
                safety_logger.error(f"Error in safety alert callback: {e}")
    
    def get_safety_findings(self, hours: int = 24, domain: Optional[SafetyDomain] = None,
                           correlation_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve safety findings from the database."""
        cutoff_time = (datetime.now() - timedelta(hours=hours)).isoformat()
        
        query = """
            SELECT timestamp, domain, check_name, severity, message,
                   value, threshold, correlation_id, metadata_json
            FROM safety_events 
            WHERE timestamp > ?
        """
        params = [cutoff_time]
        
        if domain:
            query += " AND domain = ?"
            params.append(domain.value)
        
        if correlation_id:
            query += " AND correlation_id = ?"
            params.append(correlation_id)
        
        query += " ORDER BY timestamp DESC"
        
        with sqlite3.connect(self.safety_db) as conn:
            cursor = conn.execute(query, params)
            
            findings = []
            for row in cursor.fetchall():
                findings.append({
                    'timestamp': row[0],
                    'domain': row[1],
                    'check_name': row[2],
                    'severity': row[3],
                    'message': row[4],
                    'value': row[5],
                    'threshold': row[6],
                    'correlation_id': row[7],
                    'metadata': json.loads(row[8] or '{}')
                })
            
            return findings
    
    def get_correlation_chain(self, correlation_id: str) -> List[SafetyFinding]:
        """Get all findings for a specific correlation ID."""
        return self.correlation_chains.get(correlation_id, [])
    
    def get_safety_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get safety monitoring summary."""
        cutoff_time = (datetime.now() - timedelta(hours=hours)).isoformat()
        
        with sqlite3.connect(self.safety_db) as conn:
            # Overall statistics
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_events,
                    COUNT(DISTINCT domain) as domains_active,
                    COUNT(DISTINCT correlation_id) as correlation_chains
                FROM safety_events 
                WHERE timestamp > ?
            """, (cutoff_time,))
            
            stats = cursor.fetchone()
            
            # Events by domain
            cursor = conn.execute("""
                SELECT domain, COUNT(*) as count
                FROM safety_events 
                WHERE timestamp > ?
                GROUP BY domain
            """, (cutoff_time,))
            
            domain_stats = {row[0]: row[1] for row in cursor.fetchall()}
            
            # Events by severity
            cursor = conn.execute("""
                SELECT severity, COUNT(*) as count
                FROM safety_events 
                WHERE timestamp > ?
                GROUP BY severity
            """, (cutoff_time,))
            
            severity_stats = {row[0]: row[1] for row in cursor.fetchall()}
        
        return {
            'monitoring_enabled': self.monitoring_enabled,
            'registered_checks': {
                domain.value: len(checks) 
                for domain, checks in self.safety_checks.items()
            },
            'summary_period_hours': hours,
            'total_events': stats[0] or 0,
            'active_domains': stats[1] or 0,
            'correlation_chains': stats[2] or 0,
            'events_by_domain': domain_stats,
            'events_by_severity': severity_stats,
            'overall_status': self._determine_safety_status(severity_stats),
            'generated_at': datetime.now().isoformat()
        }
    
    def _determine_safety_status(self, severity_stats: Dict[str, int]) -> str:
        """Determine overall safety status based on recent events."""
        emergency_count = severity_stats.get('emergency', 0)
        critical_count = severity_stats.get('critical', 0)
        warning_count = severity_stats.get('warning', 0)
        
        if emergency_count > 0:
            return 'EMERGENCY'
        elif critical_count > 0:
            return 'CRITICAL'
        elif warning_count > 10:  # Too many warnings
            return 'WARNING'
        else:
            return 'SAFE'
    
    def register_alert_callback(self, callback: Callable):
        """Register callback for safety alerts."""
        self.alert_callbacks.append(callback)
    
    def start_monitoring(self, interval_seconds: int = 60):
        """Start continuous safety monitoring."""
        if not self.monitoring_enabled:
            safety_logger.info("Safety monitoring is disabled")
            return
        
        if not self.running:
            self.running = True
            self.monitor_thread = threading.Thread(target=self._monitoring_loop, args=(interval_seconds,), daemon=True)
            self.monitor_thread.start()
            safety_logger.info(f"Safety monitoring started with {interval_seconds}s interval")
    
    def stop_monitoring(self):
        """Stop safety monitoring."""
        self.running = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        safety_logger.info("Safety monitoring stopped")
    
    def _monitoring_loop(self, interval_seconds: int):
        """Main monitoring loop."""
        while self.running:
            try:
                # Run all safety checks
                findings = self.run_safety_checks()
                
                if findings:
                    safety_logger.info(f"Safety check completed: {len(findings)} findings")
                
                # Clean up old correlations
                self._cleanup_old_correlations()
                
                time.sleep(interval_seconds)
                
            except Exception as e:
                safety_logger.error(f"Error in safety monitoring loop: {e}")
                time.sleep(60)  # Wait longer on error
    
    def _cleanup_old_correlations(self):
        """Clean up old correlation chains to prevent memory leaks."""
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        # Remove old correlation chains from memory
        expired_correlations = [
            corr_id for corr_id, data in self.active_correlations.items()
            if data['created_at'] < cutoff_time
        ]
        
        for corr_id in expired_correlations:
            del self.active_correlations[corr_id]
            if corr_id in self.correlation_chains:
                del self.correlation_chains[corr_id]
        
        if expired_correlations:
            safety_logger.info(f"Cleaned up {len(expired_correlations)} old correlation chains")