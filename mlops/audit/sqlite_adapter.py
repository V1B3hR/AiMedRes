#!/usr/bin/env python3
"""
Single-cell / single-file implementation of a hash‑chained SQLite audit log.

Features:
- Append-only audit_events table with previous_hash + current_hash (blockchain-like chain)
- Summary table for O(1) retrieval of last hash/index
- SHA-256 hashing with canonical JSON serialization
- Chain integrity verification (full or partial range)
- Marks per-event verification_status on verification pass
- Resilient summary row handling
- Basic concurrency guard (SQLite busy timeout + optional threading demo)
"""

import json
import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

import hashlib
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

# -----------------------------------------------------------------------------
# Logging setup
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s"
)
logger = logging.getLogger("audit_chain")

# -----------------------------------------------------------------------------
# Parent Base Class
# -----------------------------------------------------------------------------
@dataclass
class AuditEvent:
    id: str
    event_type: str
    entity_type: str
    entity_id: str
    user_id: Optional[str]
    timestamp: str
    event_data: Dict[str, Any]
    metadata: Dict[str, Any]
    previous_hash: Optional[str]
    current_hash: str
    chain_index: int


class AuditEventChain:
    """
    Generic base class. Subclasses must:
      - provide self.engine (SQLAlchemy Engine)
      - implement/create tables (e.g. in __init__)
    """

    def __init__(self, engine: Engine):
        self.engine = engine
        self._create_tables()

    # --- Hashing ----------------------------------------------------------------
    def _canonical_json(self, obj: Any) -> str:
        return json.dumps(obj, sort_keys=True, separators=(",", ":"))

    def _calculate_hash(self, event_core: Dict[str, Any], previous_hash: Optional[str]) -> str:
        """
        event_core should include only the immutable, logically relevant event fields.
        """
        hasher = hashlib.sha256()
        # Use ISO timestamp (string) if datetime
        prepared = {}
        for k, v in event_core.items():
            if isinstance(v, datetime):
                prepared[k] = v.astimezone(timezone.utc).isoformat()
            else:
                prepared[k] = v
        payload = {
            "previous_hash": previous_hash,
            "event": prepared
        }
        hasher.update(self._canonical_json(payload).encode("utf-8"))
        return hasher.hexdigest()

    # --- Verification -----------------------------------------------------------
    def verify_chain_integrity(self, start_index: int = 1, end_index: Optional[int] = None) -> Dict[str, Any]:
        """
        Recompute hashes from start_index to (optional) end_index and compare stored values.
        Returns a dict with verification details.
        """
        start_time = time.time()
        with self.engine.connect() as conn:
            params = {"start_index": start_index}
            limit_clause = ""
            if end_index is not None:
                params["end_index"] = end_index
                index_filter = "chain_index BETWEEN :start_index AND :end_index"
            else:
                index_filter = "chain_index >= :start_index"

            rows = conn.execute(text(f"""
                SELECT id, event_type, entity_type, entity_id, user_id, timestamp,
                       event_data, metadata_json, previous_hash, current_hash, chain_index
                FROM audit_events
                WHERE {index_filter}
                ORDER BY chain_index ASC
            """), params).fetchall()

        total_checked = len(rows)
        if total_checked == 0:
            return {
                "chain_valid": True,
                "valid_events": 0,
                "total_events_checked": 0,
                "verification_timestamp": datetime.now(timezone.utc).isoformat(),
                "message": "No events in specified range."
            }

        chain_valid = True
        valid_events = 0
        previous_hash = None
        mismatches: List[Dict[str, Any]] = []

        # If start_index > 1 we need previous hash from prior event to properly verify chain link
        if start_index > 1:
            with self.engine.connect() as conn:
                prior = conn.execute(text("""
                    SELECT current_hash FROM audit_events
                    WHERE chain_index = :idx
                """), {"idx": start_index - 1}).fetchone()
            if prior:
                previous_hash = prior[0]

        for row in rows:
            (rid, event_type, entity_type, entity_id, user_id, ts,
             event_data_json, metadata_json, stored_prev, stored_current, cidx) = row

            # Deserialize
            try:
                event_data_obj = json.loads(event_data_json) if event_data_json else {}
            except json.JSONDecodeError:
                event_data_obj = {"_corrupt": event_data_json}
            try:
                metadata_obj = json.loads(metadata_json) if metadata_json else {}
            except json.JSONDecodeError:
                metadata_obj = {"_corrupt": metadata_json}

            # Reconstruct core for hashing
            ts_dt = datetime.fromisoformat(ts.replace("Z", "+00:00")) if "Z" in ts or "+" in ts else datetime.fromisoformat(ts)
            event_core = {
                "event_type": event_type,
                "entity_type": entity_type,
                "entity_id": entity_id,
                "timestamp": ts_dt,
                "event_data": event_data_obj
            }
            expected_hash = self._calculate_hash(event_core, previous_hash)

            link_ok = (stored_prev == previous_hash)
            hash_ok = (expected_hash == stored_current)

            if link_ok and hash_ok:
                valid_events += 1
            else:
                chain_valid = False
                mismatches.append({
                    "chain_index": cidx,
                    "event_id": rid,
                    "link_ok": link_ok,
                    "hash_ok": hash_ok,
                    "expected_hash": expected_hash,
                    "stored_current_hash": stored_current,
                    "expected_previous_hash": previous_hash,
                    "stored_previous_hash": stored_prev
                })

            previous_hash = stored_current  # advance

        duration = time.time() - start_time
        return {
            "chain_valid": chain_valid,
            "valid_events": valid_events,
            "total_events_checked": total_checked,
            "mismatches": mismatches,
            "verification_timestamp": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": round(duration, 6)
        }


# -----------------------------------------------------------------------------
# SQLite Implementation
# -----------------------------------------------------------------------------
class SQLiteAuditEventChain(AuditEventChain):
    """
    SQLite-specific implementation using two tables:
      - audit_events
      - audit_chain_summary (single row: id=1)
    """

    def _create_tables(self):
        with self.engine.begin() as conn:
            # audit_events
            conn.execute(text("""
            CREATE TABLE IF NOT EXISTS audit_events (
                id TEXT PRIMARY KEY,
                event_type TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                user_id TEXT,
                timestamp TEXT NOT NULL,
                event_data TEXT,
                metadata_json TEXT,
                previous_hash TEXT,
                current_hash TEXT NOT NULL,
                chain_index INTEGER NOT NULL,
                verification_status TEXT DEFAULT 'unverified'
            )
            """))
            # Add uniqueness on chain_index to detect races
            conn.execute(text("""
            CREATE UNIQUE INDEX IF NOT EXISTS ux_audit_events_chain_index
            ON audit_events(chain_index)
            """))
            conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_audit_events_timestamp
            ON audit_events(timestamp)
            """))
            conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_audit_events_entity
            ON audit_events(entity_type, entity_id)
            """))
            conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_audit_events_hash
            ON audit_events(current_hash)
            """))

            # Summary table
            conn.execute(text("""
            CREATE TABLE IF NOT EXISTS audit_chain_summary (
                id INTEGER PRIMARY KEY,
                last_chain_index INTEGER NOT NULL DEFAULT 0,
                last_hash TEXT,
                total_events INTEGER NOT NULL DEFAULT 0,
                last_verification TEXT,
                chain_status TEXT DEFAULT 'unknown'
            )
            """))
            # Ensure single row (id=1)
            row = conn.execute(text("SELECT COUNT(*) FROM audit_chain_summary")).fetchone()
            if row[0] == 0:
                conn.execute(text("""
                    INSERT INTO audit_chain_summary
                    (id, last_chain_index, total_events, chain_status)
                    VALUES (1, 0, 0, 'empty')
                """))

    def log_event(self,
                  event_type: str,
                  entity_type: str,
                  entity_id: str,
                  event_data: Dict[str, Any],
                  user_id: Optional[str] = None,
                  metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Insert a new audit event in a single transaction.
        """
        import uuid
        timestamp = datetime.now(timezone.utc)
        metadata = metadata or {}

        # Wrap in transaction
        with self.engine.begin() as conn:
            summary = conn.execute(text("""
                SELECT last_chain_index, last_hash, total_events
                FROM audit_chain_summary
                WHERE id = 1
            """)).fetchone()

            if summary:
                previous_index, previous_hash, total_events = summary
                new_index = previous_index + 1
            else:
                # Should not happen, but recover gracefully
                previous_index, previous_hash, total_events = 0, None, 0
                new_index = 1
                conn.execute(text("""
                    INSERT OR IGNORE INTO audit_chain_summary
                    (id, last_chain_index, total_events, chain_status)
                    VALUES (1, 0, 0, 'recovered')
                """))

            event_id = str(uuid.uuid4())
            event_core = {
                "event_type": event_type,
                "entity_type": entity_type,
                "entity_id": entity_id,
                "timestamp": timestamp,
                "event_data": event_data
            }
            current_hash = self._calculate_hash(event_core, previous_hash)

            conn.execute(text("""
                INSERT INTO audit_events
                (id, event_type, entity_type, entity_id, user_id, timestamp,
                 event_data, metadata_json, previous_hash, current_hash, chain_index)
                VALUES
                (:id, :event_type, :entity_type, :entity_id, :user_id, :timestamp,
                 :event_data, :metadata_json, :previous_hash, :current_hash, :chain_index)
            """), {
                "id": event_id,
                "event_type": event_type,
                "entity_type": entity_type,
                "entity_id": entity_id,
                "user_id": user_id,
                "timestamp": timestamp.isoformat(),
                "event_data": json.dumps(event_data, sort_keys=True),
                "metadata_json": json.dumps(metadata, sort_keys=True),
                "previous_hash": previous_hash,
                "current_hash": current_hash,
                "chain_index": new_index
            })

            # Update summary
            conn.execute(text("""
                UPDATE audit_chain_summary
                SET last_chain_index = :idx,
                    last_hash = :h,
                    total_events = :te,
                    chain_status = 'growing'
                WHERE id = 1
            """), {
                "idx": new_index,
                "h": current_hash,
                "te": total_events + 1
            })

        logger.info(f"Logged audit event {event_id} index={new_index} type={event_type}")
        return event_id

    def verify_chain_integrity(self, start_index: int = 1, end_index: Optional[int] = None) -> Dict[str, Any]:
        """
        Extend parent verification:
          - updates summary chain_status
          - sets per-event verification_status
        """
        result = super().verify_chain_integrity(start_index, end_index)

        with self.engine.begin() as conn:
            # Mark all events in range unverified first (optional)
            params = {"start_index": start_index}
            range_clause = "chain_index >= :start_index"
            if end_index is not None:
                params["end_index"] = end_index
                range_clause = "chain_index BETWEEN :start_index AND :end_index"

            # Update per-event statuses
            if result["chain_valid"]:
                conn.execute(text(f"""
                    UPDATE audit_events
                    SET verification_status = 'verified'
                    WHERE {range_clause}
                """), params)
            else:
                # Mark mismatches explicitly
                mismatch_indices = {m["chain_index"] for m in result.get("mismatches", [])}
                if mismatch_indices:
                    conn.execute(text(f"""
                        UPDATE audit_events
                        SET verification_status = 'verified'
                        WHERE {range_clause}
                          AND chain_index NOT IN ({",".join(str(i) for i in mismatch_indices)})
                    """), params)
                    conn.execute(text(f"""
                        UPDATE audit_events
                        SET verification_status = 'mismatch'
                        WHERE chain_index IN ({",".join(str(i) for i in mismatch_indices)})
                    """))
                else:
                    conn.execute(text(f"""
                        UPDATE audit_events
                        SET verification_status = 'verified'
                        WHERE {range_clause}
                    """), params)

            conn.execute(text("""
                UPDATE audit_chain_summary
                SET last_verification = :ts,
                    chain_status = :status
                WHERE id = 1
            """), {
                "ts": datetime.now(timezone.utc).isoformat(),
                "status": "valid" if result["chain_valid"] else "invalid"
            })

        return result


# -----------------------------------------------------------------------------
# Demonstration / Usage
# -----------------------------------------------------------------------------
def demo():
    # Use file-based DB for persistence; change to ":memory:" if ephemeral
    engine = create_engine(
        "sqlite:///audit_demo.db",
        connect_args={"timeout": 15},  # help with busy waits under concurrency
        future=True
    )
    chain = SQLiteAuditEventChain(engine)

    # Log some events
    chain.log_event(
        event_type="MODEL_REGISTERED",
        entity_type="Model",
        entity_id="model_v1",
        event_data={"accuracy": 0.91, "f1": 0.88},
        user_id="alice",
        metadata={"stage": "dev"}
    )
    chain.log_event(
        event_type="MODEL_PROMOTED",
        entity_type="Model",
        entity_id="model_v1",
        event_data={"from": "dev", "to": "staging"},
        user_id="release-bot",
        metadata={"ticket": "ML-123"}
    )
    chain.log_event(
        event_type="MODEL_DEPLOYED",
        entity_type="Model",
        entity_id="model_v1",
        event_data={"env": "prod", "latency_ms_p95": 37},
        user_id="release-bot",
        metadata={"canary": True}
    )

    verification = chain.verify_chain_integrity()
    print("\nInitial verification:")
    print(json.dumps(verification, indent=2))

    # Optional: simulate a tamper
    with engine.begin() as conn:
        conn.execute(text("""
            UPDATE audit_events
            SET event_data = '{"accuracy":0.99,"f1":0.88}'
            WHERE chain_index = 1
        """))

    tampered_verification = chain.verify_chain_integrity()
    print("\nAfter tamper verification:")
    print(json.dumps(tampered_verification, indent=2))

    # Concurrency demo (simple)
    def worker(n: int):
        chain.log_event(
            event_type="HEARTBEAT",
            entity_type="System",
            entity_id=f"node-{n}",
            event_data={"ts": datetime.now(timezone.utc).isoformat()},
            user_id="health-daemon"
        )

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
    for t in threads: t.start()
    for t in threads: t.join()

    post_concurrent = chain.verify_chain_integrity()
    print("\nPost-concurrency verification:")
    print(json.dumps(post_concurrent, indent=2))

    # Show summary
    with engine.connect() as conn:
        summary = conn.execute(text("SELECT * FROM audit_chain_summary WHERE id = 1")).fetchone()
        print("\nChain summary row:", dict(summary))


if __name__ == "__main__":
    demo()
