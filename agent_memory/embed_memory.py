"""
Centralized Agent Memory Store with Advanced Filtering and TensorFlow GPU Embedding

Features:
- Centralized filtering: All memory retrieval and association logic flows through a single, highly-configurable filter.
- Filtering fields: importance, certainty, memory type, safety, security, ethics.
- TensorFlow-based embedding with GPU support (for production, use a real TF model; here, a placeholder is provided).
- Extensible for integration with safety, security, and ethics modules.
- SQLAlchemy ORM, pgvector support, robust config, logging.

Requirements:
- PostgreSQL with pgvector extension
- TensorFlow (with GPU support enabled)
"""

import os
import uuid
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import tensorflow as tf
from sqlalchemy import (
    create_engine, text, func,
    String, Integer, Float, Text, JSON, TIMESTAMP, ForeignKey,
    Enum as PgEnum, CheckConstraint, UniqueConstraint
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

try:
    from pgvector.sqlalchemy import Vector as PGVector
    PGVECTOR_AVAILABLE = True
except Exception:
    PGVECTOR_AVAILABLE = False

# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("CentralizedAgentMemoryStore")

# ------------------------------------------------------------------------------
# Config
# ------------------------------------------------------------------------------
DEFAULT_MODEL_DIM = int(os.getenv("EMBEDDING_DIM", "384"))
DEFAULT_DB_URL = os.getenv("DATABASE_URL", "postgresql://user:secret@localhost:5432/agentmem")

# ------------------------------------------------------------------------------
# Enums and Data Classes
# ------------------------------------------------------------------------------

class MemoryType(str, Enum):
    reasoning = "reasoning"
    experience = "experience"
    knowledge = "knowledge"
    observation = "observation"
    fact = "fact"
    goal = "goal"
    plan = "plan"
    belief = "belief"
    instruction = "instruction"
    emotion = "emotion"
    hallucination = "hallucination"

class AssociationType(str, Enum):
    similar = "similar"
    causal = "causal"
    temporal = "temporal"
    related = "related"
    contradicts = "contradicts"
    explains = "explains"
    supports = "supports"
    depends_on = "depends_on"
    precedes = "precedes"
    follows = "follows"
    derived_from = "derived_from"
    generalizes = "generalizes"
    specializes = "specializes"
    summary_of = "summary_of"
    alternative = "alternative"
    correction = "correction"

@dataclass
class RetrievedMemory:
    id: int
    content: str
    memory_type: MemoryType
    importance_score: float
    certainty: float
    safety: float
    security: float
    ethics: float
    metadata: Dict[str, Any]
    created_at: datetime
    access_count: int
    last_accessed: Optional[datetime]
    similarity: Optional[float]

# ------------------------------------------------------------------------------
# SQLAlchemy Models
# ------------------------------------------------------------------------------
class Base(DeclarativeBase):
    pass

class AgentSession(Base):
    __tablename__ = "agent_sessions"
    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True)
    agent_name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    agent_version: Mapped[str] = mapped_column(String(50), nullable=False, default="1.0")
    session_metadata: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="active", index=True)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)
    ended_at: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP(timezone=True), nullable=True)

class AgentMemory(Base):
    __tablename__ = "agent_memory"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(UUID(as_uuid=False), ForeignKey("agent_sessions.id", ondelete="CASCADE"), index=True, nullable=False)
    memory_type: Mapped[MemoryType] = mapped_column(PgEnum(MemoryType, name="memory_type", create_type=False), nullable=False, index=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    importance_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.5)
    certainty: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    safety: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    security: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    ethics: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    metadata_json: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    expires_at: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)
    access_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_accessed: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP(timezone=True), nullable=True)
    if PGVECTOR_AVAILABLE:
        embedding: Mapped[Any] = mapped_column(PGVector(DEFAULT_MODEL_DIM), nullable=False)
    else:
        embedding: Mapped[str] = mapped_column(Text, nullable=False)
    __table_args__ = (
        CheckConstraint("importance_score >= 0.0 AND importance_score <= 1.0", name="importance_between_0_and_1"),
        CheckConstraint("certainty >= 0.0 AND certainty <= 1.0", name="certainty_between_0_and_1"),
        CheckConstraint("safety >= 0.0 AND safety <= 1.0", name="safety_between_0_and_1"),
        CheckConstraint("security >= 0.0 AND security <= 1.0", name="security_between_0_and_1"),
        CheckConstraint("ethics >= 0.0 AND ethics <= 1.0", name="ethics_between_0_and_1"),
    )

class MemoryAssociation(Base):
    __tablename__ = "memory_associations"
    source_memory_id: Mapped[int] = mapped_column(ForeignKey("agent_memory.id", ondelete="CASCADE"), primary_key=True)
    target_memory_id: Mapped[int] = mapped_column(ForeignKey("agent_memory.id", ondelete="CASCADE"), primary_key=True)
    association_type: Mapped[AssociationType] = mapped_column(
        PgEnum(AssociationType, name="association_type", create_type=False), primary_key=True
    )
    strength: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)
    __table_args__ = (
        CheckConstraint("strength >= 0.0 AND strength <= 1.0", name="association_strength_between_0_and_1"),
        UniqueConstraint("source_memory_id", "target_memory_id", "association_type", name="uq_memory_association"),
    )

# ------------------------------------------------------------------------------
# TensorFlow Embedding Utility (replace with production model as needed)
# ------------------------------------------------------------------------------
class TensorFlowEmbedder:
    def __init__(self, dim=DEFAULT_MODEL_DIM):
        self.dim = dim
        # For demo: randomly init a "model" kernel
        self.kernel = tf.Variable(tf.random.normal([dim, 256]), trainable=False)
        logger.info("Initialized TensorFlowEmbedder (placeholder).")

    def encode(self, text: str) -> np.ndarray:
        """Simulated embedding (replace with real TF model in production)."""
        x = tf.constant([float(ord(c)) for c in text[:256]])
        x = tf.pad(x, [[0, max(0, 256 - tf.shape(x)[0])]])
        emb = tf.linalg.matvec(self.kernel, tf.math.tanh(x))
        norm = tf.norm(emb)
        emb = emb / (norm + 1e-8)
        return emb.numpy()

# ------------------------------------------------------------------------------
# Centralized Filter Function
# ------------------------------------------------------------------------------
def centralized_memory_filter(
    memories: List[Dict[str, Any]],
    min_importance=0.5,
    min_certainty=0.8,
    allowed_types=None,
    min_safety=0.7,
    min_security=0.7,
    min_ethics=0.7
) -> List[Dict[str, Any]]:
    """Apply all filtering rules centrally."""
    return [
        m for m in memories
        if m['importance_score'] >= min_importance
        and m['certainty'] >= min_certainty
        and (allowed_types is None or m['memory_type'] in allowed_types)
        and m['safety'] >= min_safety
        and m['security'] >= min_security
        and m['ethics'] >= min_ethics
    ]

# ------------------------------------------------------------------------------
# Centralized Memory Store Class
# ------------------------------------------------------------------------------
class CentralizedAgentMemoryStore:
    def __init__(
        self,
        db_connection_string: Optional[str] = None,
        embedding_dim: int = DEFAULT_MODEL_DIM,
        echo_sql: bool = False,
        pool_size: int = 5,
        max_overflow: int = 10,
        tf_embedder: Optional[TensorFlowEmbedder] = None
    ):
        self.db_url = db_connection_string or DEFAULT_DB_URL
        self.engine = create_engine(
            self.db_url,
            echo=echo_sql,
            pool_size=pool_size,
            max_overflow=max_overflow,
            future=True,
        )
        self.SessionLocal = sessionmaker(bind=self.engine, expire_on_commit=False, autoflush=False, future=True)
        self.embedding_dim = embedding_dim
        self.embedder = tf_embedder or TensorFlowEmbedder(dim=embedding_dim)
        logger.info("CentralizedAgentMemoryStore initialized.")

    def store_memory(
        self,
        session_id: str,
        content: str,
        memory_type: MemoryType,
        importance: float,
        certainty: float,
        safety: float,
        security: float,
        ethics: float,
        metadata: Optional[Dict[str, Any]] = None,
        expires_hours: Optional[int] = None
    ) -> int:
        """Store a memory, enforcing all safety/importance/certainty/ethics constraints."""
        # Enforce constraints
        if not (0.0 <= importance <= 1.0):
            raise ValueError("importance must be 0..1")
        if not (0.0 <= certainty <= 1.0):
            raise ValueError("certainty must be 0..1")
        if not (0.0 <= safety <= 1.0):
            raise ValueError("safety must be 0..1")
        if not (0.0 <= security <= 1.0):
            raise ValueError("security must be 0..1")
        if not (0.0 <= ethics <= 1.0):
            raise ValueError("ethics must be 0..1")
        if not content or not content.strip():
            raise ValueError("content cannot be empty")

        # Safety: block storage if any core fields are below threshold
        if importance < 0.5 or certainty < 0.8 or safety < 0.7 or security < 0.7 or ethics < 0.7:
            raise ValueError("Memory does not meet minimum safety/importance/ethics thresholds for storage.")

        # Embedding
        embedding = self.embedder.encode(content)
        expires_at = None
        if expires_hours:
            expires_at = datetime.now(tz=timezone.utc) + timedelta(hours=expires_hours)
        mem_meta = metadata or {}
        try:
            with self.SessionLocal.begin() as db:
                mem = AgentMemory(
                    session_id=session_id,
                    memory_type=memory_type,
                    content=content,
                    importance_score=importance,
                    certainty=certainty,
                    safety=safety,
                    security=security,
                    ethics=ethics,
                    metadata_json=mem_meta,
                    expires_at=expires_at,
                    embedding=embedding.tolist() if PGVECTOR_AVAILABLE else str(embedding.tolist())
                )
                db.add(mem)
                db.flush()
                memory_id = int(mem.id)
            logger.info(f"Stored memory id={memory_id} [{memory_type.value}] with certainty={certainty:.2f}, safety={safety:.2f}")
            return memory_id
        except Exception as e:
            logger.exception("Failed to store memory")
            raise

    def retrieve_memories(
        self,
        session_id: str,
        query: str,
        limit: int = 5,
        min_importance: float = 0.5,
        min_certainty: float = 0.8,
        allowed_types: Optional[List[str]] = None,
        min_safety: float = 0.7,
        min_security: float = 0.7,
        min_ethics: float = 0.7
    ) -> List[RetrievedMemory]:
        """Centralized retrieval and filtering (semantic + rule-based)."""
        query_emb = self.embedder.encode(query)
        where = ["session_id = :session_id"]
        params = {"session_id": session_id}
        try:
            with self.SessionLocal.begin() as db:
                if PGVECTOR_AVAILABLE:
                    sql = f"""
                        SELECT id, content, memory_type, importance_score, certainty,
                               safety, security, ethics, metadata_json, created_at,
                               access_count, last_accessed,
                               (embedding <=> :query_embedding) AS distance
                        FROM agent_memory
                        WHERE {' AND '.join(where)}
                        ORDER BY embedding <=> :query_embedding, importance_score DESC, created_at DESC
                        LIMIT :limit
                    """
                    params["query_embedding"] = query_emb.tolist()
                    params["limit"] = limit * 5  # retrieve more, filter down
                    rows = db.execute(text(sql), params).fetchall()
                else:
                    sql = f"""
                        SELECT id, content, memory_type, importance_score, certainty,
                               safety, security, ethics, metadata_json, created_at,
                               access_count, last_accessed
                        FROM agent_memory
                        WHERE {' AND '.join(where)}
                        ORDER BY importance_score DESC, created_at DESC
                        LIMIT :limit
                    """
                    params["limit"] = limit * 5
                    rows = db.execute(text(sql), params).fetchall()
                # Format for centralized filter
                memories = []
                for r in rows:
                    mem = {
                        "id": int(r.id),
                        "content": r.content,
                        "memory_type": str(r.memory_type),
                        "importance_score": float(r.importance_score),
                        "certainty": float(r.certainty),
                        "safety": float(r.safety),
                        "security": float(r.security),
                        "ethics": float(r.ethics),
                        "metadata": r.metadata_json or {},
                        "created_at": r.created_at,
                        "access_count": int(r.access_count),
                        "last_accessed": r.last_accessed,
                        "similarity": 1.0 - float(getattr(r, "distance", 0.0)) if hasattr(r, "distance") else None,
                    }
                    memories.append(mem)
                # Centralized filter
                filtered = centralized_memory_filter(
                    memories,
                    min_importance=min_importance,
                    min_certainty=min_certainty,
                    allowed_types=allowed_types,
                    min_safety=min_safety,
                    min_security=min_security,
                    min_ethics=min_ethics
                )
                # Sort by similarity and importance
                filtered.sort(key=lambda m: (m.get('similarity', 0), m['importance_score']), reverse=True)
                top = filtered[:limit]
                return [
                    RetrievedMemory(
                        id=m["id"], content=m["content"], memory_type=MemoryType(m["memory_type"]),
                        importance_score=m["importance_score"], certainty=m["certainty"], safety=m["safety"],
                        security=m["security"], ethics=m["ethics"], metadata=m["metadata"], created_at=m["created_at"],
                        access_count=m["access_count"], last_accessed=m["last_accessed"], similarity=m["similarity"]
                    )
                    for m in top
                ]
        except Exception as e:
            logger.exception("Failed to retrieve memories")
            raise

    def create_memory_association(
        self,
        source_memory_id: int,
        target_memory_id: int,
        association_type: AssociationType,
        strength: float = 1.0
    ) -> None:
        """Associations are only allowed if both memories meet all safety/ethics/importance constraints."""
        with self.SessionLocal.begin() as db:
            s = db.execute(text("SELECT certainty, safety, security, ethics FROM agent_memory WHERE id=:id"),
                           {"id": source_memory_id}).fetchone()
            t = db.execute(text("SELECT certainty, safety, security, ethics FROM agent_memory WHERE id=:id"),
                           {"id": target_memory_id}).fetchone()
            if not s or not t:
                raise ValueError("Source or target memory not found")
            for field, minv in zip(
                ["certainty", "safety", "security", "ethics"],
                [0.8, 0.7, 0.7, 0.7]
            ):
                if float(s[field]) < minv or float(t[field]) < minv:
                    raise ValueError(f"Association blocked: {field} below safe threshold")
            # Proceed
            db.execute(
                text(
                    """
                    INSERT INTO memory_associations (source_memory_id, target_memory_id, association_type, strength)
                    VALUES (:source, :target, :type, :strength)
                    ON CONFLICT (source_memory_id, target_memory_id, association_type)
                    DO UPDATE SET strength = EXCLUDED.strength
                    """
                ),
                {"source": source_memory_id, "target": target_memory_id, "type": association_type.value, "strength": strength},
            )
            logger.info(f"Created association {association_type.value} between {source_memory_id} and {target_memory_id}")

# ------------------------------------------------------------------------------
# Example Usage
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("Starting centralized agent memory demo...")
    store = CentralizedAgentMemoryStore()
    # Create a session
    session_id = str(uuid.uuid4())
    with store.SessionLocal.begin() as db:
        db.add(AgentSession(id=session_id, agent_name="CentralAgent", agent_version="1.0", session_metadata={}, status="active"))
    # Store safe memory
    mem_id = store.store_memory(
        session_id=session_id,
        content="Patient diagnosed with hypertension and prescribed ACE inhibitors.",
        memory_type=MemoryType.knowledge,
        importance=0.9,
        certainty=0.95,
        safety=0.9,
        security=0.9,
        ethics=0.95,
        metadata={"source": "EHR"}
    )
    # Attempt unsafe memory (should fail)
    try:
        store.store_memory(
            session_id=session_id,
            content="This is a hallucinated and unsafe memory.",
            memory_type=MemoryType.hallucination,
            importance=0.2,
            certainty=0.2,
            safety=0.2,
            security=0.2,
            ethics=0.2,
            metadata={"source": "generated"}
        )
    except ValueError as e:
        logger.warning(f"Blocked unsafe memory: {e}")

    # Retrieve with full centralized filtering
    results = store.retrieve_memories(
        session_id=session_id,
        query="hypertension",
        limit=3,
        min_importance=0.5,
        min_certainty=0.8,
        allowed_types=["knowledge", "reasoning", "observation"],
        min_safety=0.7,
        min_security=0.7,
        min_ethics=0.7
    )
    for m in results:
        logger.info(f"Retrieved: {m.memory_type.value} | Certainty: {m.certainty:.2f} | Safety: {m.safety:.2f} | {m.content}")

    logger.info("Centralized agent memory demo completed.")
