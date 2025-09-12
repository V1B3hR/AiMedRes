#!/usr/bin/env python3
"""
SQLite adapter for audit event chain to ensure compatibility.
"""

from .event_chain import AuditEventChain
from sqlalchemy import create_engine, text
import logging

logger = logging.getLogger(__name__)


class SQLiteAuditEventChain(AuditEventChain):
    """
    SQLite-compatible version of AuditEventChain.
    """
    
    def _create_tables(self):
        """Create SQLite-compatible audit event tables."""
        try:
            with self.engine.connect() as conn:
                # Create audit events table with SQLite-compatible schema
                conn.execute(text("""
                CREATE TABLE IF NOT EXISTS audit_events (
                    id TEXT PRIMARY KEY,
                    event_type VARCHAR(100) NOT NULL,
                    entity_type VARCHAR(100) NOT NULL,
                    entity_id VARCHAR(200) NOT NULL,
                    user_id VARCHAR(100),
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    event_data TEXT,
                    metadata_json TEXT,
                    previous_hash VARCHAR(64),
                    current_hash VARCHAR(64) NOT NULL,
                    chain_index INTEGER NOT NULL,
                    verification_status VARCHAR(20) DEFAULT 'unverified'
                )
                """))
                
                # Create indexes separately
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_audit_events_timestamp ON audit_events(timestamp)"))
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_audit_events_entity ON audit_events(entity_type, entity_id)"))
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_audit_events_hash ON audit_events(current_hash)"))
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_audit_events_chain ON audit_events(chain_index)"))
                
                # Create audit event summary table
                conn.execute(text("""
                CREATE TABLE IF NOT EXISTS audit_chain_summary (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    last_chain_index INTEGER NOT NULL DEFAULT 0,
                    last_hash VARCHAR(64),
                    total_events INTEGER NOT NULL DEFAULT 0,
                    last_verification DATETIME,
                    chain_status VARCHAR(20) DEFAULT 'valid'
                )
                """))
                
                # Initialize summary if empty - check first
                result = conn.execute(text("SELECT COUNT(*) FROM audit_chain_summary")).fetchone()
                if result[0] == 0:
                    conn.execute(text("INSERT INTO audit_chain_summary (last_chain_index, total_events) VALUES (0, 0)"))
                
                conn.commit()
                logger.info("SQLite audit event tables created/verified")
                
        except Exception as e:
            logger.error(f"Error creating SQLite audit tables: {e}")
            raise
    
    def log_event(self, event_type: str, entity_type: str, entity_id: str,
                  event_data: dict, user_id: str = None, metadata: dict = None) -> str:
        """SQLite-compatible event logging."""
        import uuid
        import json
        from datetime import datetime, timezone
        
        try:
            with self.engine.connect() as conn:
                # Get previous chain info
                previous_info = conn.execute(text("""
                SELECT last_chain_index, last_hash, total_events
                FROM audit_chain_summary LIMIT 1
                """)).fetchone()
                
                if previous_info:
                    previous_index, previous_hash, total_events = previous_info
                    new_index = previous_index + 1
                else:
                    previous_index, previous_hash, total_events = 0, None, 0
                    new_index = 1
                
                # Generate event ID
                event_id = str(uuid.uuid4())
                
                # Prepare event data for hashing
                timestamp = datetime.now(timezone.utc)
                event_for_hash = {
                    'event_type': event_type,
                    'entity_type': entity_type,
                    'entity_id': entity_id,
                    'timestamp': timestamp,
                    'event_data': event_data
                }
                
                # Calculate hash
                current_hash = self._calculate_hash(event_for_hash, previous_hash)
                
                # Insert event
                conn.execute(text("""
                INSERT INTO audit_events 
                (id, event_type, entity_type, entity_id, user_id, timestamp, 
                 event_data, metadata_json, previous_hash, current_hash, chain_index)
                VALUES 
                (:id, :event_type, :entity_type, :entity_id, :user_id, :timestamp,
                 :event_data, :metadata, :previous_hash, :current_hash, :chain_index)
                """), {
                    'id': event_id,
                    'event_type': event_type,
                    'entity_type': entity_type,
                    'entity_id': entity_id,
                    'user_id': user_id,
                    'timestamp': timestamp.isoformat(),
                    'event_data': json.dumps(event_data),
                    'metadata': json.dumps(metadata or {}),
                    'previous_hash': previous_hash,
                    'current_hash': current_hash,
                    'chain_index': new_index
                })
                
                # Update chain summary
                conn.execute(text("""
                UPDATE audit_chain_summary 
                SET last_chain_index = :chain_index,
                    last_hash = :current_hash,
                    total_events = :total_events
                """), {
                    'chain_index': new_index,
                    'current_hash': current_hash,
                    'total_events': total_events + 1
                })
                
                conn.commit()
                
                logger.info(f"Logged audit event {event_id}: {event_type} for {entity_type}/{entity_id}")
                return event_id
                
        except Exception as e:
            logger.error(f"Error logging audit event: {e}")
            raise