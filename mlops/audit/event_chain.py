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
import uuid
import traceback

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
        with self.engine.begin() as conn:
            # Create audit events table (SQLite compatible)
            conn.execute(text("""
            CREATE TABLE IF NOT EXISTS audit_events (
                id TEXT PRIMARY KEY,
                event_type TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                user_id TEXT,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                event_data TEXT,
                metadata_json TEXT,
                previous_hash TEXT,
                current_hash TEXT NOT NULL,
                chain_index INTEGER NOT NULL,
                verification_status TEXT DEFAULT 'unverified'
            )
            """))

            # Create indexes separately
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_audit_events_timestamp ON audit_events(timestamp)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_audit_events_entity ON audit_events(entity_type, entity_id)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_audit_events_hash ON audit_events(current_hash)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_audit_events_chain ON audit_events(chain_index)"))

            # Create audit event summary table for chain integrity (SQLite compatible)
            conn.execute(text("""
            CREATE TABLE IF NOT EXISTS audit_chain_summary (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                last_chain_index INTEGER NOT NULL DEFAULT 0,
                last_hash TEXT,
                total_events INTEGER NOT NULL DEFAULT 0,
                last_verification TEXT,
                chain_status TEXT DEFAULT 'valid'
            )
            """))

            # Initialize summary if empty
            conn.execute(text("""
            INSERT OR IGNORE INTO audit_chain_summary (id, last_chain_index, total_events)
            VALUES (1, 0, 0)
            """))

            logger.info("Audit event tables created/verified")

     except Exception as e:
            logger.error(f"Error creating audit tables: {e}")
            raise

def _deep_sort(obj):
    """Recursively sort lists and dicts for deterministic serialization."""
    if isinstance(obj, dict):
        return {k: _deep_sort(obj[k]) for k in sorted(obj)}
    if isinstance(obj, list):
        return sorted(_deep_sort(x) for x in obj)
    return obj

def _calculate_hash(
    self,
    event_data: Dict[str, Any],
    previous_hash: Optional[str],
    *,
    hash_algo: str = 'sha256',
    version: int = 1
) -> str:
    """
    Calculate hash for an event using a cryptographic hash function.
    
    Args:
        event_data: Event data to hash.
        previous_hash: Hash of the previous event in chain.
        hash_algo: Hash algorithm (default 'sha256').
        version: Hash schema version.

    Returns:
        Hash string.
    """
    # Validate presence of required fields
    required_fields = ['event_type', 'entity_type', 'entity_id', 'timestamp']
    for field in required_fields:
        if field not in event_data:
            raise ValueError(f"Missing required field in event_data: {field}")

    # Normalize and deeply sort any nested structures
    normalized_event_data = _deep_sort(event_data.get('event_data', {}))
    timestamp = event_data['timestamp']
    if isinstance(timestamp, datetime):
        timestamp_str = timestamp.isoformat()
    else:
        timestamp_str = str(timestamp)

    # Prepare deterministic hash input
    hash_input = {
        'version': version,
        'event_type': event_data['event_type'],
        'entity_type': event_data['entity_type'],
        'entity_id': event_data['entity_id'],
        'timestamp': timestamp_str,
        'event_data': normalized_event_data,
        'previous_hash': previous_hash or ''
    }

    # Canonical JSON serialization
    hash_string = json.dumps(
        hash_input,
        sort_keys=True,
        separators=(',', ':'),
        ensure_ascii=False
    )

    # Select hash algorithm
    try:
        hasher = getattr(hashlib, hash_algo)()
    except AttributeError:
        raise ValueError(f"Unsupported hash algorithm: {hash_algo}")

    hasher.update(hash_string.encode('utf-8'))
    return hasher.hexdigest()
    
logger = logging.getLogger(__name__)

def _deep_sort(obj):
    """Recursively sort lists and dicts for deterministic serialization."""
    if isinstance(obj, dict):
        return {k: _deep_sort(obj[k]) for k in sorted(obj)}
    if isinstance(obj, list):
        return sorted(_deep_sort(x) for x in obj)
    return obj

def log_event(
    self,
    event_type: str,
    entity_type: str,
    entity_id: str,
    event_data: Dict[str, Any],
    user_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Log an audit event to the chain.
    Returns: Event ID (UUID string)
    """
    try:
        # Input validation
        for field, value in [
            ("event_type", event_type),
            ("entity_type", entity_type),
            ("entity_id", entity_id),
            ("event_data", event_data),
        ]:
            if not value:
                raise ValueError(f"Missing required field: {field}")

        with self.engine.begin() as conn:
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

            # Prepare event data for hashing and storage
            timestamp = datetime.now(timezone.utc)
            timestamp_str = timestamp.isoformat(timespec="microseconds")

            event_for_hash = {
                'event_type': event_type,
                'entity_type': entity_type,
                'entity_id': entity_id,
                'timestamp': timestamp_str,
                'event_data': _deep_sort(event_data),
            }

            event_id = str(uuid.uuid4())
            current_hash = self._calculate_hash(event_for_hash, previous_hash)

            # Insert event (deterministic, deeply sorted JSON)
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
                'timestamp': timestamp_str,
                'event_data': json.dumps(_deep_sort(event_data), sort_keys=True, separators=(',', ':')),
                'metadata': json.dumps(_deep_sort(metadata or {}), sort_keys=True, separators=(',', ':')),
                'previous_hash': previous_hash or '',
                'current_hash': current_hash,
                'chain_index': new_index
            })

            # Update chain summary
            conn.execute(text("""
                UPDATE audit_chain_summary 
                SET last_chain_index = :chain_index,
                    last_hash = :current_hash,
                    total_events = :total_events
                WHERE id = 1
            """), {
                'chain_index': new_index,
                'current_hash': current_hash,
                'total_events': total_events + 1
            })

            logger.info(f"Logged audit event {event_id}: {event_type} for {entity_type}/{entity_id}")
            return event_id

    except Exception as e:
        logger.error(f"Error logging audit event: {e}", exc_info=True)
        raise
    
    def verify_chain_integrity(self, start_index: int = 1, end_index: Optional[int] = None) -> Dict[str, Any]:
    """
    Verify the integrity of the audit chain.
    Args:
        start_index: Starting chain index to verify
        end_index: Ending chain index (None for all)
    Returns:
        Verification results as a dict.
    """
    from sqlalchemy.exc import SQLAlchemyError
    import traceback

    try:
        with self.engine.begin() as conn:
            # Build and execute query for audit events in range
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

            verification_time = datetime.now(timezone.utc).isoformat()
            verification_results = {
                'total_events_checked': len(events),
                'checked_range': {
                    'start_index': start_index,
                    'end_index': end_index if end_index is not None else (events[-1][8] if events else None)
                },
                'valid_events': 0,
                'invalid_events': [],
                'chain_valid': True,
                'verification_timestamp': verification_time
            }

            # Get previous hash before start_index
            previous_hash = None
            if start_index > 1:
                prev_event = conn.execute(text("""
                    SELECT current_hash FROM audit_events 
                    WHERE chain_index = :prev_index
                """), {'prev_index': start_index - 1}).fetchone()
                previous_hash = prev_event[0] if prev_event else None

            # Iterate and verify each event
            for event in events:
                (event_id, event_type, entity_type, entity_id, timestamp,
                 event_data_json, stored_prev_hash, stored_current_hash, chain_index) = event

                # Parse event_data and timestamp
                try:
                    event_data = json.loads(event_data_json) if event_data_json else {}
                except Exception:
                    event_data = event_data_json or {}

                # Use consistent timestamp string for hash
                if isinstance(timestamp, str):
                    timestamp_str = timestamp
                else:
                    # If timestamp is datetime, convert to isoformat
                    timestamp_str = timestamp.isoformat() if hasattr(timestamp, "isoformat") else str(timestamp)

                # Deterministic event_data
                event_for_hash = {
                    'event_type': event_type,
                    'entity_type': entity_type,
                    'entity_id': entity_id,
                    'timestamp': timestamp_str,
                    'event_data': self._deep_sort(event_data)
                }

                expected_hash = self._calculate_hash(event_for_hash, previous_hash)

                # Hash mismatch
                if expected_hash != stored_current_hash:
                    verification_results['invalid_events'].append({
                        'event_id': str(event_id),
                        'chain_index': chain_index,
                        'expected_hash': expected_hash,
                        'stored_hash': stored_current_hash,
                        'issue': 'hash_mismatch'
                    })
                    verification_results['chain_valid'] = False
                # Chain break (previous hash mismatch)
                elif stored_prev_hash != (previous_hash or ''):
                    verification_results['invalid_events'].append({
                        'event_id': str(event_id),
                        'chain_index': chain_index,
                        'expected_prev_hash': previous_hash or '',
                        'stored_prev_hash': stored_prev_hash,
                        'issue': 'chain_break'
                    })
                    verification_results['chain_valid'] = False
                else:
                    verification_results['valid_events'] += 1

                previous_hash = stored_current_hash

            # Update summary table with verification time and status
            conn.execute(
                text("""
                    UPDATE audit_chain_summary 
                    SET last_verification = :verification_time, chain_status = :status
                    WHERE id = 1
                """),
                {
                    'verification_time': verification_time,
                    'status': 'valid' if verification_results['chain_valid'] else 'invalid'
                }
            )

            logger.info(
                f"Chain verification completed: {verification_results['valid_events']}/"
                f"{verification_results['total_events_checked']} events valid, "
                f"chain_valid={verification_results['chain_valid']}"
            )
            return verification_results

    except SQLAlchemyError as db_exc:
        logger.error(f"Database error during chain integrity verification: {db_exc}", exc_info=True)
        return {
            'error': str(db_exc),
            'chain_valid': False,
            'verification_timestamp': datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"General error verifying chain integrity: {e}\n{traceback.format_exc()}")
        return {
            'error': str(e),
            'chain_valid': False,
            'verification_timestamp': datetime.now(timezone.utc).isoformat()
        }

def _safe_json_loads(raw: Optional[str], fallback: Any = None) -> Any:
    if not raw:
        return fallback if fallback is not None else {}
    try:
        return json.loads(raw)
    except Exception as e:
        logger.warning(f"Failed to decode JSON: {e}; raw={raw!r}")
        return fallback if fallback is not None else {}

def get_entity_audit_trail(
    self,
    entity_type: str,
    entity_id: str,
    limit: int = 100,
    event_type: Optional[str] = None  # extensible filter
) -> List[Dict[str, Any]]:
    """
    Get audit trail for a specific entity.

    Args:
        entity_type: Type of entity.
        entity_id: Entity identifier.
        limit: Maximum number of events (clamped to 1000).
        event_type: (Optional) Filter by event type.

    Returns:
        List of dicts:
        [
            {
                'id': str,
                'event_type': str,
                'timestamp': str (ISO8601, UTC),
                'event_data': dict,
                'metadata': dict,
                'hash': str,
                'chain_index': int,
                'user_id': str
            },
            ...
        ]
    """
    # Validate and sanitize input
    if not entity_type or not entity_id:
        logger.error("entity_type and entity_id must be provided.")
        return []
    if limit <= 0:
        logger.warning(f"Requested limit {limit} is invalid, using default 100.")
        limit = 100
    limit = min(limit, 1000)

    try:
        with self.engine.connect() as conn:
            query = """
                SELECT id, event_type, timestamp, event_data, metadata_json, 
                       current_hash, chain_index, user_id
                FROM audit_events
                WHERE entity_type = :entity_type AND entity_id = :entity_id
            """
            params = {
                'entity_type': entity_type,
                'entity_id': entity_id,
                'limit': limit
            }
            if event_type:
                query += " AND event_type = :event_type"
                params['event_type'] = event_type
            query += " ORDER BY timestamp DESC LIMIT :limit"

            events = conn.execute(text(query), params).fetchall()

            trail = []
            for event in events:
                # Defensive conversion for timestamp
                ts = event[2]
                if hasattr(ts, 'isoformat'):
                    timestamp_str = ts.isoformat()
                else:
                    timestamp_str = str(ts) if ts else None

                trail.append({
                    'id': str(event[0]),
                    'event_type': event[1],
                    'timestamp': timestamp_str,
                    'event_data': _safe_json_loads(event[3]),
                    'metadata': _safe_json_loads(event[4]),
                    'hash': event[5],
                    'chain_index': event[6],
                    'user_id': event[7]
                })

            return trail

    except Exception as e:
        logger.error(f"Error getting audit trail: {e}", exc_info=True)
        # Optionally, return a structured error instead of empty list
        return []
    
    #!/usr/bin/env python3
"""
Audit Event System with Hash Chaining for DuetMind Adaptive MLOps.
Provides immutable audit trail for model operations and data changes.
"""
# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _deep_sort(obj):
    """Recursively sort lists and dicts for deterministic serialization."""
    if isinstance(obj, dict):
        return {k: _deep_sort(obj[k]) for k in sorted(obj)}
    if isinstance(obj, list):
        return sorted(_deep_sort(x) for x in obj)
    return obj

def _safe_json_loads(raw: Optional[str], fallback: Any = None) -> Any:
    if not raw:
        return fallback if fallback is not None else {}
    try:
        return json.loads(raw)
    except Exception as e:
        logger.warning(f"Failed to decode JSON: {e}; raw={raw!r}")
        return fallback if fallback is not None else {}

class AuditEventChain:
    """
    Blockchain-inspired audit event system with hash chaining for integrity verification.
    """

    def __init__(self, db_connection_string: str):
        """
        Initialize the audit event chain.

        Args:
            db_connection_string: Database connection string.
        """
        self.engine = create_engine(db_connection_string)
        self.metadata = MetaData()
        self._create_tables()
        logger.info("Initialized AuditEventChain")

    def _create_tables(self):
        """Create audit event tables if they don't exist."""
        try:
            with self.engine.begin() as conn:
                # Create audit events table (SQLite compatible)
                conn.execute(text("""
                CREATE TABLE IF NOT EXISTS audit_events (
                    id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    entity_id TEXT NOT NULL,
                    user_id TEXT,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    event_data TEXT,
                    metadata_json TEXT,
                    previous_hash TEXT,
                    current_hash TEXT NOT NULL,
                    chain_index INTEGER NOT NULL,
                    verification_status TEXT DEFAULT 'unverified'
                )
                """))

                # Create indexes
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_audit_events_timestamp ON audit_events(timestamp)"))
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_audit_events_entity ON audit_events(entity_type, entity_id)"))
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_audit_events_hash ON audit_events(current_hash)"))
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_audit_events_chain ON audit_events(chain_index)"))

                # Create audit event summary table
                conn.execute(text("""
                CREATE TABLE IF NOT EXISTS audit_chain_summary (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    last_chain_index INTEGER NOT NULL DEFAULT 0,
                    last_hash TEXT,
                    total_events INTEGER NOT NULL DEFAULT 0,
                    last_verification TEXT,
                    chain_status TEXT DEFAULT 'valid'
                )
                """))

                # Initialize summary if empty
                conn.execute(text("""
                INSERT OR IGNORE INTO audit_chain_summary (id, last_chain_index, total_events)
                VALUES (1, 0, 0)
                """))

                logger.info("Audit event tables created/verified")

        except Exception as e:
            logger.error(f"Error creating audit tables: {e}", exc_info=True)
            raise

    def _calculate_hash(self, event_data: Dict[str, Any], previous_hash: Optional[str], hash_algo: str = 'sha256', version: int = 1) -> str:
        """
        Calculate hash for an event using a cryptographic hash function.
        Args:
            event_data: Event data to hash.
            previous_hash: Hash of the previous event in chain.
            hash_algo: Hash algorithm (default 'sha256').
            version: Hash schema version.
        Returns:
            Hash string.
        """
        # Validate required fields
        for field in ['event_type', 'entity_type', 'entity_id', 'timestamp']:
            if field not in event_data:
                raise ValueError(f"Missing required field in event_data: {field}")

        # Normalize and deeply sort any nested structures
        normalized_event_data = _deep_sort(event_data.get('event_data', {}))
        timestamp = event_data['timestamp']
        timestamp_str = timestamp.isoformat() if hasattr(timestamp, "isoformat") else str(timestamp)

        # Prepare deterministic hash input
        hash_input = {
            'version': version,
            'event_type': event_data['event_type'],
            'entity_type': event_data['entity_type'],
            'entity_id': event_data['entity_id'],
            'timestamp': timestamp_str,
            'event_data': normalized_event_data,
            'previous_hash': previous_hash or ''
        }

        hash_string = json.dumps(
            hash_input,
            sort_keys=True,
            separators=(',', ':'),
            ensure_ascii=False
        )
        try:
            hasher = getattr(hashlib, hash_algo)()
        except AttributeError:
            raise ValueError(f"Unsupported hash algorithm: {hash_algo}")

        hasher.update(hash_string.encode('utf-8'))
        return hasher.hexdigest()

    def log_event(self, event_type: str, entity_type: str, entity_id: str,
                  event_data: Dict[str, Any], user_id: Optional[str] = None,
                  metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Log an audit event to the chain.
        Returns:
            Event ID (UUID string)
        """
        try:
            # Input validation
            for field, value in [
                ("event_type", event_type),
                ("entity_type", entity_type),
                ("entity_id", entity_id),
                ("event_data", event_data),
            ]:
                if not value:
                    raise ValueError(f"Missing required field: {field}")

            with self.engine.begin() as conn:
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

                timestamp = datetime.now(timezone.utc)
                timestamp_str = timestamp.isoformat(timespec="microseconds")

                event_for_hash = {
                    'event_type': event_type,
                    'entity_type': entity_type,
                    'entity_id': entity_id,
                    'timestamp': timestamp_str,
                    'event_data': _deep_sort(event_data),
                }

                event_id = str(uuid.uuid4())
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
                    'timestamp': timestamp_str,
                    'event_data': json.dumps(_deep_sort(event_data), sort_keys=True, separators=(',', ':')),
                    'metadata': json.dumps(_deep_sort(metadata or {}), sort_keys=True, separators=(',', ':')),
                    'previous_hash': previous_hash or '',
                    'current_hash': current_hash,
                    'chain_index': new_index
                })

                # Update chain summary
                conn.execute(text("""
                    UPDATE audit_chain_summary 
                    SET last_chain_index = :chain_index,
                        last_hash = :current_hash,
                        total_events = :total_events
                    WHERE id = 1
                """), {
                    'chain_index': new_index,
                    'current_hash': current_hash,
                    'total_events': total_events + 1
                })

                logger.info(f"Logged audit event {event_id}: {event_type} for {entity_type}/{entity_id}")
                return event_id

        except Exception as e:
            logger.error(f"Error logging audit event: {e}", exc_info=True)
            raise

    def verify_chain_integrity(self, start_index: int = 1, end_index: Optional[int] = None) -> Dict[str, Any]:
        """
        Verify the integrity of the audit chain.
        Returns:
            Verification results as a dict.
        """
        try:
            with self.engine.begin() as conn:
                # Build and execute query for audit events in range
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

                verification_time = datetime.now(timezone.utc).isoformat()
                verification_results = {
                    'total_events_checked': len(events),
                    'checked_range': {
                        'start_index': start_index,
                        'end_index': end_index if end_index is not None else (events[-1][8] if events else None)
                    },
                    'valid_events': 0,
                    'invalid_events': [],
                    'chain_valid': True,
                    'verification_timestamp': verification_time
                }

                # Get previous hash before start_index
                previous_hash = None
                if start_index > 1:
                    prev_event = conn.execute(text("""
                        SELECT current_hash FROM audit_events 
                        WHERE chain_index = :prev_index
                    """), {'prev_index': start_index - 1}).fetchone()
                    previous_hash = prev_event[0] if prev_event else None

                # Iterate and verify each event
                for event in events:
                    (event_id, event_type, entity_type, entity_id, timestamp,
                     event_data_json, stored_prev_hash, stored_current_hash, chain_index) = event

                    event_data = _safe_json_loads(event_data_json)
                    timestamp_str = timestamp if isinstance(timestamp, str) else (timestamp.isoformat() if hasattr(timestamp, "isoformat") else str(timestamp))

                    event_for_hash = {
                        'event_type': event_type,
                        'entity_type': entity_type,
                        'entity_id': entity_id,
                        'timestamp': timestamp_str,
                        'event_data': _deep_sort(event_data)
                    }

                    expected_hash = self._calculate_hash(event_for_hash, previous_hash)

                    if expected_hash != stored_current_hash:
                        verification_results['invalid_events'].append({
                            'event_id': str(event_id),
                            'chain_index': chain_index,
                            'expected_hash': expected_hash,
                            'stored_hash': stored_current_hash,
                            'issue': 'hash_mismatch'
                        })
                        verification_results['chain_valid'] = False
                    elif stored_prev_hash != (previous_hash or ''):
                        verification_results['invalid_events'].append({
                            'event_id': str(event_id),
                            'chain_index': chain_index,
                            'expected_prev_hash': previous_hash or '',
                            'stored_prev_hash': stored_prev_hash,
                            'issue': 'chain_break'
                        })
                        verification_results['chain_valid'] = False
                    else:
                        verification_results['valid_events'] += 1

                    previous_hash = stored_current_hash

                # Update summary table with verification time and status
                conn.execute(
                    text("""
                        UPDATE audit_chain_summary 
                        SET last_verification = :verification_time, chain_status = :status
                        WHERE id = 1
                    """),
                    {
                        'verification_time': verification_time,
                        'status': 'valid' if verification_results['chain_valid'] else 'invalid'
                    }
                )

                logger.info(
                    f"Chain verification completed: {verification_results['valid_events']}/"
                    f"{verification_results['total_events_checked']} events valid, "
                    f"chain_valid={verification_results['chain_valid']}"
                )
                return verification_results

        except Exception as e:
            logger.error(f"Error verifying chain integrity: {e}\n{traceback.format_exc()}")
            return {
                'error': str(e),
                'chain_valid': False,
                'verification_timestamp': datetime.now(timezone.utc).isoformat()
            }

    def get_entity_audit_trail(self, entity_type: str, entity_id: str, 
                               limit: int = 100, event_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get audit trail for a specific entity.
        Returns:
            List of audit events for the entity.
        """
        # Input validation and sanitization
        if not entity_type or not entity_id:
            logger.error("entity_type and entity_id must be provided.")
            return []
        if limit <= 0:
            logger.warning(f"Requested limit {limit} is invalid, using default 100.")
            limit = 100
        limit = min(limit, 1000)

        try:
            with self.engine.connect() as conn:
                query = """
                    SELECT id, event_type, timestamp, event_data, metadata_json, 
                           current_hash, chain_index, user_id
                    FROM audit_events
                    WHERE entity_type = :entity_type AND entity_id = :entity_id
                """
                params = {
                    'entity_type': entity_type,
                    'entity_id': entity_id,
                    'limit': limit
                }
                if event_type:
                    query += " AND event_type = :event_type"
                    params['event_type'] = event_type
                query += " ORDER BY timestamp DESC LIMIT :limit"

                events = conn.execute(text(query), params).fetchall()
                trail = []
                for event in events:
                    ts = event[2]
                    timestamp_str = ts.isoformat() if hasattr(ts, 'isoformat') else str(ts) if ts else None
                    trail.append({
                        'id': str(event[0]),
                        'event_type': event[1],
                        'timestamp': timestamp_str,
                        'event_data': _safe_json_loads(event[3]),
                        'metadata': _safe_json_loads(event[4]),
                        'hash': event[5],
                        'chain_index': event[6],
                        'user_id': event[7]
                    })
                return trail

        except Exception as e:
            logger.error(f"Error getting audit trail: {e}", exc_info=True)
            return []

    def get_chain_summary(self) -> Dict[str, Any]:
        """
        Get summary information about the audit chain.
        Returns:
            Chain summary statistics with robust error handling and additional metadata.
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
                    last_verification = summary[3]
                    if last_verification is not None:
                        if hasattr(last_verification, 'isoformat'):
                            last_verification = last_verification.isoformat()
                        else:
                            last_verification = str(last_verification)

                    return {
                        'last_chain_index': summary[0] if summary[0] is not None else 0,
                        'last_hash': summary[1] if summary[1] is not None else "",
                        'total_events': summary[2] if summary[2] is not None else 0,
                        'last_verification': last_verification,
                        'chain_status': summary[4] if summary[4] is not None else "unknown",
                        'summary_exists': True,
                    }
                else:
                    return {
                        'last_chain_index': 0,
                        'total_events': 0,
                        'chain_status': 'empty',
                        'summary_exists': False,
                    }
        except Exception as e:
            logger.error(f"Error getting chain summary: {e}", exc_info=True)
            return {
                'error': str(e),
                'summary_exists': False,
                'chain_status': 'error'
            }

def demo_audit_system():
    """
    Demonstrate the audit event system functionality.
    """
    logger.info("Starting audit event system demonstration...")

    # Initialize with local SQLite for demo (in production, use PostgreSQL)
    audit_chain = AuditEventChain("sqlite:///audit_demo.db")

    try:
        # Show chain summary before logging
        summary_before = audit_chain.get_chain_summary()
        logger.info(f"Initial chain summary: {summary_before}")

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
        logger.info(f"Chain verification: {verification_results.get('chain_valid', False)}")
        logger.info(f"Valid events: {verification_results.get('valid_events', 0)}/{verification_results.get('total_events_checked', 0)}")

        # Get audit trail for model
        trail = audit_chain.get_entity_audit_trail('model', 'alzheimer_classifier_v1.0')
        logger.info(f"Model audit trail contains {len(trail)} events")

        # Get chain summary after logging
        summary = audit_chain.get_chain_summary()
        logger.info(f"Final chain summary: {summary.get('total_events', 0)} total events, status: {summary.get('chain_status', 'unknown')}")

        print("\n=== Audit Event System Demo Results ===")
        print(f"Events logged: {len(event_ids)}")
        print(f"Chain integrity: {'VALID' if verification_results.get('chain_valid') else 'INVALID'}")
        print(f"Total events in chain: {summary.get('total_events', 0)}")
        print(f"Model audit trail events: {len(trail)}")

        if trail:
            print("\nLatest model events:")
            for event in trail[:3]:  # Show latest 3 events
                print(f"  - {event['event_type']} at {event['timestamp']} by {event.get('user_id', 'unknown')}")
        else:
            print("No audit trail events found for the model.")

    except Exception as e:
        logger.error(f"Error in audit system demo: {e}", exc_info=True)
        print(f"Demo failed due to error: {e}")

if __name__ == "__main__":
    demo_audit_system()
