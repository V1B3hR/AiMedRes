#!/usr/bin/env python3
"""
Agent Memory Store for DuetMind Adaptive (production-ready).

Key features:
- True semantic search using pgvector cosine distance
- SQLAlchemy ORM with typed models and transactional sessions
- Robust config (env + YAML), safe logging, and error handling
- Memory associations with enum-constrained types
- Expiration handling, access metrics, and importance thresholding

Requirements:
- PostgreSQL with pgvector extension
- See migrations/001_init.sql to set up schema and indexes
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

import yaml
from sentence_transformers import SentenceTransformer
from sqlalchemy import (
    JSON,
    TIMESTAMP,
    CheckConstraint,
    Enum as PgEnum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

try:
    # Optional but recommended: leverage SQLAlchemy pgvector integration
    # pip install pgvector
    from pgvector.sqlalchemy import Vector as PGVector
    PGVECTOR_AVAILABLE = True
except Exception:
    PGVECTOR_AVAILABLE = False


# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("AgentMemoryStore")


# ------------------------------------------------------------------------------
# Config
# ------------------------------------------------------------------------------
DEFAULT_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
DEFAULT_DB_URL = os.getenv("DATABASE_URL", "postgresql://duetmind:duetmind_secret@localhost:5432/duetmind")

# If you're using all-MiniLM-L6-v2, dim is 384. Change if you switch models.
# For production, keep a consistent model/dimension unless you version tables.
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))


def load_params(params_file: str = "params.yaml") -> dict:
    if os.path.exists(params_file):
        with open(params_file, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


# ------------------------------------------------------------------------------
# ORM Models
# ------------------------------------------------------------------------------
class Base(DeclarativeBase):
    pass


class MemoryType(str, Enum):
    reasoning = "reasoning"
    experience = "experience"
    knowledge = "knowledge"


class AssociationType(str, Enum):
    similar = "similar"
    causal = "causal"
    temporal = "temporal"
    related = "related"  # keep for compatibility with demo


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
    metadata_json: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    expires_at: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)
    access_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_accessed: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP(timezone=True), nullable=True)

    # Embedding column (pgvector). If pgvector/sqlalchemy isn't installed, we fallback to a Text column,
    # but note: semantic search won't work without pgvector.
    if PGVECTOR_AVAILABLE:
        embedding: Mapped[Any] = mapped_column(PGVector(EMBEDDING_DIM), nullable=False)
    else:
        # Fallback storage only (no semantic search). Strongly recommend installing pgvector.
        embedding: Mapped[str] = mapped_column(Text, nullable=False)

    __table_args__ = (
        CheckConstraint("importance_score >= 0.0 AND importance_score <= 1.0", name="importance_between_0_and_1"),
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
# DTOs
# ------------------------------------------------------------------------------
@dataclass
class RetrievedMemory:
    id: int
    content: str
    memory_type: MemoryType
    importance_score: float
    metadata: Dict[str, Any]
    created_at: datetime
    access_count: int
    last_accessed: Optional[datetime]
    similarity: Optional[float]  # 0..1 (None if pgvector unavailable)


# ------------------------------------------------------------------------------
# AgentMemoryStore
# ------------------------------------------------------------------------------
class AgentMemoryStore:
    def __init__(
        self,
        db_connection_string: Optional[str] = None,
        embedding_model: str = DEFAULT_MODEL,
        echo_sql: bool = False,
        pool_size: int = 5,
        max_overflow: int = 10,
    ):
        """
        Args:
            db_connection_string: Postgres URL. Defaults to DATABASE_URL env or YAML.
            embedding_model: Sentence Transformer model name.
            echo_sql: Log SQL statements.
            pool_size: SQLAlchemy pool size.
            max_overflow: SQLAlchemy overflow size.
        """
        self.db_url = db_connection_string or self._load_db_url_from_yaml() or DEFAULT_DB_URL
        self.engine: Engine = create_engine(
            self.db_url,
            echo=echo_sql,
            pool_size=pool_size,
            max_overflow=max_overflow,
            future=True,
        )
        self.SessionLocal = sessionmaker(bind=self.engine, expire_on_commit=False, autoflush=False, future=True)

        # Load model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.model = SentenceTransformer(embedding_model)
        actual_dim = self.model.get_sentence_embedding_dimension()
        if actual_dim != EMBEDDING_DIM and PGVECTOR_AVAILABLE:
            logger.warning(
                f"Model dimension {actual_dim} != configured EMBEDDING_DIM {EMBEDDING_DIM}. "
                "Ensure your table/migration matches the model's dimension."
            )
        logger.info(f"Initialized AgentMemoryStore with {embedding_model} (dim={actual_dim})")

    def _load_db_url_from_yaml(self) -> Optional[str]:
        params = load_params()
        db = params.get("database") or {}
        user = db.get("user")
        pwd = db.get("password")
        host = db.get("host")
        port = db.get("port")
        name = db.get("name") or db.get("db") or db.get("database")
        if all([user, pwd, host, port, name]):
            return f"postgresql://{user}:{pwd}@{host}:{port}/{name}"
        return None

    # ----------------------------- Sessions -----------------------------------
    def create_session(self, agent_name: str, agent_version: str = "1.0", metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a new agent session. Returns session ID (UUID string).
        """
        session_id = str(uuid.uuid4())
        try:
            with self.SessionLocal.begin() as db:
                db.add(
                    AgentSession(
                        id=session_id,
                        agent_name=agent_name,
                        agent_version=agent_version,
                        session_metadata=metadata or {},
                        status="active",
                    )
                )
            logger.info(f"Created session {session_id} for agent {agent_name}")
            return session_id
        except SQLAlchemyError as e:
            logger.exception("Failed to create session")
            raise

    def end_session(self, session_id: str) -> None:
        """
        Mark a session as completed.
        """
        try:
            with self.SessionLocal.begin() as db:
                db.execute(
                    text(
                        """
                        UPDATE agent_sessions
                        SET status = 'completed', ended_at = NOW()
                        WHERE id = :session_id
                        """
                    ),
                    {"session_id": session_id},
                )
            logger.info(f"Ended session {session_id}")
        except SQLAlchemyError:
            logger.exception("Failed to end session")
            raise

    # ----------------------------- Memories -----------------------------------
    def store_memory(
        self,
        session_id: str,
        content: str,
        memory_type: MemoryType | str = MemoryType.reasoning,
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
        expires_hours: Optional[int] = None,
    ) -> int:
        """
        Store a memory with its embedding. Returns memory ID.
        """
        if isinstance(memory_type, str):
            memory_type = MemoryType(memory_type)

        if not (0.0 <= importance <= 1.0):
            raise ValueError("importance must be between 0.0 and 1.0")

        if not content or not content.strip():
            raise ValueError("content cannot be empty")

        # Embed
        embedding = self.model.encode(content, normalize_embeddings=PGVECTOR_AVAILABLE).astype("float32").tolist()

        expires_at = None
        if expires_hours:
            expires_at = datetime.now(tz=timezone.utc) + timedelta(hours=expires_hours)

        try:
            with self.SessionLocal.begin() as db:
                mem = AgentMemory(
                    session_id=session_id,
                    memory_type=memory_type,
                    content=content,
                    importance_score=importance,
                    metadata_json=metadata or {},
                    expires_at=expires_at,
                    embedding=embedding if PGVECTOR_AVAILABLE else json.dumps(embedding),
                )
                db.add(mem)
                db.flush()  # populate mem.id
                memory_id = int(mem.id)
            logger.info(f"Stored memory {memory_id} in session {session_id}")
            return memory_id
        except SQLAlchemyError:
            logger.exception("Failed to store memory")
            raise

    def retrieve_memories(
        self,
        session_id: str,
        query: str,
        limit: int = 5,
        memory_type: Optional[MemoryType | str] = None,
        min_importance: float = 0.0,
    ) -> List[RetrievedMemory]:
        """
        Retrieve similar memories using semantic search (pgvector cosine distance).
        If pgvector is unavailable, falls back to importance/date ordering without semantic ranking.
        """
        if memory_type and isinstance(memory_type, str):
            memory_type = MemoryType(memory_type)

        if PGVECTOR_AVAILABLE:
            query_embedding = self.model.encode(query, normalize_embeddings=True).astype("float32").tolist()
        else:
            # No vector search available
            query_embedding = None

        where_clauses = ["session_id = :session_id", "importance_score >= :min_importance", "(expires_at IS NULL OR expires_at > NOW())"]
        params: Dict[str, Any] = {"session_id": session_id, "min_importance": min_importance, "limit": limit}

        if memory_type:
            where_clauses.append("memory_type = :memory_type")
            params["memory_type"] = memory_type.value

        try:
            with self.SessionLocal.begin() as db:
                if PGVECTOR_AVAILABLE:
                    # Use cosine distance; similarity = 1 - distance with cosine ops
                    sql = f"""
                        SELECT id,
                               content,
                               memory_type,
                               importance_score,
                               metadata_json,
                               created_at,
                               access_count,
                               last_accessed,
                               (embedding <=> :query_embedding) AS distance
                        FROM agent_memory
                        WHERE {' AND '.join(where_clauses)}
                        ORDER BY embedding <=> :query_embedding, importance_score DESC, created_at DESC
                        LIMIT :limit
                    """
                    params["query_embedding"] = query_embedding
                    rows = db.execute(text(sql), params).fetchall()
                    ids = [r.id for r in rows]
                    if ids:
                        db.execute(
                            text(
                                """
                                UPDATE agent_memory
                                SET access_count = access_count + 1,
                                    last_accessed = NOW()
                                WHERE id = ANY(:ids)
                                """
                            ),
                            {"ids": ids},
                        )
                    results: List[RetrievedMemory] = []
                    for r in rows:
                        distance = float(r.distance)
                        similarity = max(0.0, 1.0 - distance)
                        results.append(
                            RetrievedMemory(
                                id=int(r.id),
                                content=r.content,
                                memory_type=MemoryType(r.memory_type),
                                importance_score=float(r.importance_score),
                                metadata=r.metadata_json or {},
                                created_at=r.created_at,
                                access_count=int(r.access_count) + (1 if r.id in ids else 0),
                                last_accessed=r.last_accessed,
                                similarity=similarity,
                            )
                        )
                    logger.info(f"Retrieved {len(results)} memories for query in session {session_id}")
                    return results
                else:
                    # Fallback: no semantic ordering available
                    sql = f"""
                        SELECT id,
                               content,
                               memory_type,
                               importance_score,
                               metadata_json,
                               created_at,
                               access_count,
                               last_accessed
                        FROM agent_memory
                        WHERE {' AND '.join(where_clauses)}
                        ORDER BY importance_score DESC, created_at DESC
                        LIMIT :limit
                    """
                    rows = db.execute(text(sql), params).fetchall()
                    ids = [r.id for r in rows]
                    if ids:
                        db.execute(
                            text(
                                """
                                UPDATE agent_memory
                                SET access_count = access_count + 1,
                                    last_accessed = NOW()
                                WHERE id = ANY(:ids)
                                """
                            ),
                            {"ids": ids},
                        )
                    results = [
                        RetrievedMemory(
                            id=int(r.id),
                            content=r.content,
                            memory_type=MemoryType(r.memory_type),
                            importance_score=float(r.importance_score),
                            metadata=r.metadata_json or {},
                            created_at=r.created_at,
                            access_count=int(r.access_count) + (1 if r.id in ids else 0),
                            last_accessed=r.last_accessed,
                            similarity=None,
                        )
                        for r in rows
                    ]
                    logger.warning("pgvector not available; returned fallback ranking without semantic distance")
                    return results
        except SQLAlchemyError:
            logger.exception("Failed to retrieve memories")
            raise

    # --------------------------- Associations ----------------------------------
    def create_memory_association(
        self,
        source_memory_id: int,
        target_memory_id: int,
        association_type: AssociationType | str = AssociationType.similar,
        strength: float = 1.0,
    ) -> None:
        if isinstance(association_type, str):
            association_type = AssociationType(association_type)
        if not (0.0 <= strength <= 1.0):
            raise ValueError("strength must be between 0.0 and 1.0")

        try:
            with self.SessionLocal.begin() as db:
                # Upsert
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
            logger.info(
                f"Created {association_type.value} association between memories {source_memory_id} and {target_memory_id}"
            )
        except SQLAlchemyError:
            logger.exception("Failed to create memory association")
            raise

    # ------------------------------ Utilities ----------------------------------
    def get_session_memories(self, session_id: str) -> List[Dict[str, Any]]:
        try:
            with self.SessionLocal.begin() as db:
                rows = db.execute(
                    text(
                        """
                        SELECT id, content, memory_type, importance_score, created_at, access_count
                        FROM agent_memory
                        WHERE session_id = :session_id
                          AND (expires_at IS NULL OR expires_at > NOW())
                        ORDER BY created_at DESC
                        """
                    ),
                    {"session_id": session_id},
                ).fetchall()
            return [
                {
                    "id": int(r.id),
                    "content": r.content,
                    "memory_type": MemoryType(r.memory_type).value,
                    "importance_score": float(r.importance_score),
                    "created_at": r.created_at,
                    "access_count": int(r.access_count),
                }
                for r in rows
            ]
        except SQLAlchemyError:
            logger.exception("Failed to get session memories")
            raise

    def ensure_connection(self) -> None:
        """
        Verify DB connectivity and pgvector availability.
        """
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            if PGVECTOR_AVAILABLE:
                logger.info("Database connection OK and pgvector SQLAlchemy type available.")
            else:
                logger.warning("Database connection OK but pgvector SQLAlchemy type not installed. Install 'pgvector'.")
        except SQLAlchemyError:
            logger.exception("Database connectivity check failed")
            raise


# ------------------------------------------------------------------------------
# Demo
# ------------------------------------------------------------------------------
def demo_agent_memory() -> None:
    logger.info("Starting agent memory demonstration...")

    params = load_params()
    db_url = os.getenv("DATABASE_URL") or DEFAULT_DB_URL

    try:
        # Initialize memory store
        memory_store = AgentMemoryStore(db_connection_string=db_url)
        memory_store.ensure_connection()

        # Create agent session
        session_id = memory_store.create_session(
            agent_name="DuetMind_Alzheimer_Agent",
            agent_version="1.0",
            metadata={"model": "alzheimer_classifier", "domain": "healthcare"},
        )

        # Store some memories
        memories = [
            {
                "content": "Patient shows high MMSE score (28) but has APOE4 genotype, indicating complex risk profile",
                "type": "reasoning",
                "importance": 0.8,
            },
            {
                "content": "Age-related cognitive decline patterns suggest early intervention may be beneficial",
                "type": "reasoning",
                "importance": 0.7,
            },
            {
                "content": "Family history is a strong predictor but should be combined with biomarker data",
                "type": "knowledge",
                "importance": 0.9,
            },
            {
                "content": "Model confidence drops when education level is below 8 years",
                "type": "experience",
                "importance": 0.6,
            },
        ]

        memory_ids: List[int] = []
        for mem in memories:
            memory_id = memory_store.store_memory(
                session_id=session_id,
                content=mem["content"],
                memory_type=mem["type"],
                importance=mem["importance"],
            )
            memory_ids.append(memory_id)

        # Create associations
        memory_store.create_memory_association(memory_ids[0], memory_ids[2], "related", 0.8)

        # Retrieval
        query = "APOE4 genetic risk factors"
        retrieved = memory_store.retrieve_memories(session_id, query, limit=3)

        logger.info(f"Query: '{query}'")
        logger.info("Retrieved memories:")
        for i, mem in enumerate(retrieved, 1):
            sim_txt = f", Similarity: {mem.similarity:.3f}" if mem.similarity is not None else ""
            logger.info(
                f"  {i}. [{mem.memory_type.value}] {mem.content[:80]}... "
                f"(Importance: {mem.importance_score:.2f}, Accessed: {mem.access_count}{sim_txt})"
            )

        # Get all session memories
        all_memories = memory_store.get_session_memories(session_id)
        logger.info(f"Total memories in session: {len(all_memories)}")

        # End session
        memory_store.end_session(session_id)

        logger.info("Agent memory demonstration completed successfully!")

    except Exception as e:
        logger.exception("Error in agent memory demo")
        logger.info("Make sure PostgreSQL is running with pgvector and the schema is migrated.")
        logger.info("You can start it with: docker compose up -d && apply migrations in migrations/001_init.sql")


if __name__ == "__main__":
    demo_agent_memory()
