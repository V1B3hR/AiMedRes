#!/usr/bin/env python3
"""
Audit Event System with Hash Chaining for DuetMind Adaptive MLOps.
Provides immutable audit trail for model operations and data changes.
"""

import hashlib
import json
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
from sqlalchemy import create_engine, text, MetaData, Table, Column, String, DateTime, Text, Integer
from sqlalchemy.dialects.postgresql import UUID, JSONB
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AuditEventChain:
    """
    Blockchain-inspired audit event system with hash chaining for integrity verification.
    """
    
    def __init__(self, db_connection_string: str):
        """
        Initialize the audit event chain.
        
        Args:
            db_connection_string: PostgreSQL connection string
        """
        self.engine = create_engine(db_connection_string)
        self.metadata = MetaData()
        self._create_tables()
        logger.info("Initialized AuditEventChain")
    
    def _create_tables(self):
        """Create audit event tables if they don't exist."""
        try:
            with self.engine.connect() as conn:
                # Create audit events table
                conn.execute(text("""
                CREATE TABLE IF NOT EXISTS audit_events (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    event_type VARCHAR(100) NOT NULL,
                    entity_type VARCHAR(100) NOT NULL,
                    entity_id VARCHAR(200) NOT NULL,
                    user_id VARCHAR(100),
                    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    event_data JSONB,
                    metadata_json JSONB,
                    previous_hash VARCHAR(64),
                    current_hash VARCHAR(64) NOT NULL,
                    chain_index INTEGER NOT NULL,
                    verification_status VARCHAR(20) DEFAULT 'unverified'
                );
                
                CREATE INDEX IF NOT EXISTS idx_audit_events_timestamp ON audit_events(timestamp);
                CREATE INDEX IF NOT EXISTS idx_audit_events_entity ON audit_events(entity_type, entity_id);
                CREATE INDEX IF NOT EXISTS idx_audit_events_hash ON audit_events(current_hash);
                CREATE INDEX IF NOT EXISTS idx_audit_events_chain ON audit_events(chain_index);
                """))
                
                # Create audit event summary table for chain integrity
                conn.execute(text("""
                CREATE TABLE IF NOT EXISTS audit_chain_summary (
                    id SERIAL PRIMARY KEY,
                    last_chain_index INTEGER NOT NULL DEFAULT 0,
                    last_hash VARCHAR(64),
                    total_events INTEGER NOT NULL DEFAULT 0,
                    last_verification TIMESTAMP WITH TIME ZONE,
                    chain_status VARCHAR(20) DEFAULT 'valid'
                );
                
                -- Initialize summary if empty
                INSERT INTO audit_chain_summary (last_chain_index, total_events)
                SELECT 0, 0
                WHERE NOT EXISTS (SELECT 1 FROM audit_chain_summary);
                """))
                
                conn.commit()
                logger.info("Audit event tables created/verified")
                
        except Exception as e:
            logger.error(f"Error creating audit tables: {e}")
            raise
    
    def _calculate_hash(self, event_data: Dict[str, Any], previous_hash: Optional[str]) -> str:
        """
        Calculate hash for an event using SHA-256.
        
        Args:
            event_data: Event data to hash
            previous_hash: Hash of the previous event in chain
            
        Returns:
            SHA-256 hash string
        """
        # Create deterministic string representation
        hash_input = {
            'event_type': event_data['event_type'],
            'entity_type': event_data['entity_type'],
            'entity_id': event_data['entity_id'],
            'timestamp': event_data['timestamp'].isoformat() if isinstance(event_data['timestamp'], datetime) else str(event_data['timestamp']),
            'event_data': event_data.get('event_data', {}),
            'previous_hash': previous_hash or ''
        }
        
        # Sort keys for deterministic hashing
        hash_string = json.dumps(hash_input, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(hash_string.encode('utf-8')).hexdigest()
    
    def log_event(self, event_type: str, entity_type: str, entity_id: str,
                  event_data: Dict[str, Any], user_id: Optional[str] = None,
                  metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Log an audit event to the chain.
        
        Args:
            event_type: Type of event (e.g., 'model_trained', 'data_ingested', 'drift_detected')
            entity_type: Type of entity (e.g., 'model', 'dataset', 'experiment')
            entity_id: Unique identifier for the entity
            event_data: Event-specific data
            user_id: User who triggered the event
            metadata: Additional metadata
            
        Returns:
            Event ID (UUID string)
        """
        try:
            with self.engine.connect() as conn:
                # Get previous chain info
                previous_info = conn.execute(text("""
                SELECT last_chain_index, last_hash, total_events
                FROM audit_chain_summary
                ORDER BY id DESC LIMIT 1
                """)).fetchone()
                
                if previous_info:
                    previous_index, previous_hash, total_events = previous_info
                    new_index = previous_index + 1
                else:
                    previous_index, previous_hash, total_events = 0, None, 0
                    new_index = 1
                
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
                result = conn.execute(text("""
                INSERT INTO audit_events 
                (event_type, entity_type, entity_id, user_id, timestamp, 
                 event_data, metadata_json, previous_hash, current_hash, chain_index)
                VALUES 
                (:event_type, :entity_type, :entity_id, :user_id, :timestamp,
                 :event_data, :metadata, :previous_hash, :current_hash, :chain_index)
                RETURNING id
                """), {
                    'event_type': event_type,
                    'entity_type': entity_type,
                    'entity_id': entity_id,
                    'user_id': user_id,
                    'timestamp': timestamp,
                    'event_data': json.dumps(event_data),
                    'metadata': json.dumps(metadata or {}),
                    'previous_hash': previous_hash,
                    'current_hash': current_hash,
                    'chain_index': new_index
                })
                
                event_id = str(result.fetchone()[0])
                
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
    
    def verify_chain_integrity(self, start_index: int = 1, end_index: Optional[int] = None) -> Dict[str, Any]:
        """
        Verify the integrity of the audit chain.
        
        Args:
            start_index: Starting chain index to verify
            end_index: Ending chain index (None for all)
            
        Returns:
            Verification results
        """
        try:
            with self.engine.connect() as conn:
                # Build query
                query = """
                SELECT id, event_type, entity_type, entity_id, timestamp, 
                       event_data, previous_hash, current_hash, chain_index
                FROM audit_events 
                WHERE chain_index >= :start_index
                """
                params = {'start_index': start_index}
                
                if end_index:
                    query += " AND chain_index <= :end_index"
                    params['end_index'] = end_index
                
                query += " ORDER BY chain_index"
                
                events = conn.execute(text(query), params).fetchall()
                
                verification_results = {
                    'total_events_checked': len(events),
                    'valid_events': 0,
                    'invalid_events': [],
                    'chain_valid': True,
                    'verification_timestamp': datetime.now(timezone.utc).isoformat()
                }
                
                previous_hash = None
                if start_index > 1:
                    # Get the hash of the event before start_index
                    prev_event = conn.execute(text("""
                    SELECT current_hash FROM audit_events 
                    WHERE chain_index = :prev_index
                    """), {'prev_index': start_index - 1}).fetchone()
                    
                    if prev_event:
                        previous_hash = prev_event[0]
                
                # Verify each event in sequence
                for event in events:
                    event_id, event_type, entity_type, entity_id, timestamp, event_data, stored_prev_hash, stored_current_hash, chain_index = event
                    
                    # Reconstruct event data for hashing
                    event_for_hash = {
                        'event_type': event_type,
                        'entity_type': entity_type,
                        'entity_id': entity_id,
                        'timestamp': timestamp,
                        'event_data': json.loads(event_data) if event_data else {}
                    }
                    
                    # Calculate expected hash
                    expected_hash = self._calculate_hash(event_for_hash, previous_hash)
                    
                    # Verify hash integrity
                    if expected_hash != stored_current_hash:
                        verification_results['invalid_events'].append({
                            'event_id': str(event_id),
                            'chain_index': chain_index,
                            'expected_hash': expected_hash,
                            'stored_hash': stored_current_hash,
                            'issue': 'hash_mismatch'
                        })
                        verification_results['chain_valid'] = False
                    elif stored_prev_hash != previous_hash:
                        verification_results['invalid_events'].append({
                            'event_id': str(event_id),
                            'chain_index': chain_index,
                            'expected_prev_hash': previous_hash,
                            'stored_prev_hash': stored_prev_hash,
                            'issue': 'chain_break'
                        })
                        verification_results['chain_valid'] = False
                    else:
                        verification_results['valid_events'] += 1
                    
                    # Update for next iteration
                    previous_hash = stored_current_hash
                
                # Update verification status in database
                if verification_results['chain_valid']:
                    conn.execute(text("""
                    UPDATE audit_chain_summary 
                    SET last_verification = NOW(), chain_status = 'valid'
                    """))
                else:
                    conn.execute(text("""
                    UPDATE audit_chain_summary 
                    SET last_verification = NOW(), chain_status = 'invalid'
                    """))
                
                conn.commit()
                
                logger.info(f"Chain verification completed: {verification_results['valid_events']}/{verification_results['total_events_checked']} events valid")
                return verification_results
                
        except Exception as e:
            logger.error(f"Error verifying chain integrity: {e}")
            return {
                'error': str(e),
                'chain_valid': False,
                'verification_timestamp': datetime.now(timezone.utc).isoformat()
            }
    
    def get_entity_audit_trail(self, entity_type: str, entity_id: str, 
                              limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get audit trail for a specific entity.
        
        Args:
            entity_type: Type of entity
            entity_id: Entity identifier
            limit: Maximum number of events to return
            
        Returns:
            List of audit events for the entity
        """
        try:
            with self.engine.connect() as conn:
                events = conn.execute(text("""
                SELECT id, event_type, timestamp, event_data, metadata_json, 
                       current_hash, chain_index, user_id
                FROM audit_events
                WHERE entity_type = :entity_type AND entity_id = :entity_id
                ORDER BY timestamp DESC
                LIMIT :limit
                """), {
                    'entity_type': entity_type,
                    'entity_id': entity_id,
                    'limit': limit
                }).fetchall()
                
                trail = []
                for event in events:
                    trail.append({
                        'id': str(event[0]),
                        'event_type': event[1],
                        'timestamp': event[2].isoformat() if event[2] else None,
                        'event_data': json.loads(event[3]) if event[3] else {},
                        'metadata': json.loads(event[4]) if event[4] else {},
                        'hash': event[5],
                        'chain_index': event[6],
                        'user_id': event[7]
                    })
                
                return trail
                
        except Exception as e:
            logger.error(f"Error getting audit trail: {e}")
            return []
    
    def get_chain_summary(self) -> Dict[str, Any]:
        """
        Get summary information about the audit chain.
        
        Returns:
            Chain summary statistics
        """
        try:
            with self.engine.connect() as conn:
                summary = conn.execute(text("""
                SELECT last_chain_index, last_hash, total_events, 
                       last_verification, chain_status
                FROM audit_chain_summary
                ORDER BY id DESC LIMIT 1
                """)).fetchone()
                
                if summary:
                    return {
                        'last_chain_index': summary[0],
                        'last_hash': summary[1],
                        'total_events': summary[2],
                        'last_verification': summary[3].isoformat() if summary[3] else None,
                        'chain_status': summary[4]
                    }
                else:
                    return {
                        'last_chain_index': 0,
                        'total_events': 0,
                        'chain_status': 'empty'
                    }
                    
        except Exception as e:
            logger.error(f"Error getting chain summary: {e}")
            return {'error': str(e)}


def demo_audit_system():
    """
    Demonstrate the audit event system functionality.
    """
    logger.info("Starting audit event system demonstration...")
    
    # Initialize with local SQLite for demo (in production, use PostgreSQL)
    audit_chain = AuditEventChain("sqlite:///audit_demo.db")
    
    try:
        # Log some sample events
        events = [
            {
                'event_type': 'model_trained',
                'entity_type': 'model',
                'entity_id': 'alzheimer_classifier_v1.0',
                'event_data': {
                    'accuracy': 0.94,
                    'training_duration': 120,
                    'dataset_size': 2149
                },
                'user_id': 'ml_engineer_1'
            },
            {
                'event_type': 'drift_detected',
                'entity_type': 'dataset',
                'entity_id': 'alzheimer_features_v2',
                'event_data': {
                    'drift_score': 0.15,
                    'affected_features': ['age', 'mmse_score'],
                    'detection_method': 'evidently'
                },
                'user_id': 'system'
            },
            {
                'event_type': 'model_promoted',
                'entity_type': 'model',
                'entity_id': 'alzheimer_classifier_v1.0',
                'event_data': {
                    'promotion_reason': 'accuracy_threshold_met',
                    'previous_version': 'alzheimer_classifier_v0.9',
                    'environment': 'production'
                },
                'user_id': 'deployment_system'
            }
        ]
        
        event_ids = []
        for event in events:
            event_id = audit_chain.log_event(**event)
            event_ids.append(event_id)
            
        logger.info(f"Logged {len(event_ids)} audit events")
        
        # Verify chain integrity
        verification_results = audit_chain.verify_chain_integrity()
        logger.info(f"Chain verification: {verification_results['chain_valid']}")
        logger.info(f"Valid events: {verification_results['valid_events']}/{verification_results['total_events_checked']}")
        
        # Get audit trail for model
        trail = audit_chain.get_entity_audit_trail('model', 'alzheimer_classifier_v1.0')
        logger.info(f"Model audit trail contains {len(trail)} events")
        
        # Get chain summary
        summary = audit_chain.get_chain_summary()
        logger.info(f"Chain summary: {summary['total_events']} total events, status: {summary['chain_status']}")
        
        print("\n=== Audit Event System Demo Results ===")
        print(f"Events logged: {len(event_ids)}")
        print(f"Chain integrity: {'VALID' if verification_results['chain_valid'] else 'INVALID'}")
        print(f"Total events in chain: {summary['total_events']}")
        print(f"Model audit trail events: {len(trail)}")
        
        if trail:
            print("\nLatest model events:")
            for event in trail[:3]:  # Show latest 3 events
                print(f"  - {event['event_type']} at {event['timestamp']} by {event['user_id']}")
        
    except Exception as e:
        logger.error(f"Error in audit system demo: {e}")


if __name__ == "__main__":
    demo_audit_system()