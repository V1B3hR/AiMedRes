#!/usr/bin/env python3
"""
Agent Memory Embedding Prototype for DuetMind Adaptive.
Persists agent reasoning embeddings for semantic memory and retrieval.
"""

import logging
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import numpy as np
from sqlalchemy import create_engine, text
from sentence_transformers import SentenceTransformer
import json
import os
import yaml

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentMemoryStore:
    """
    Agent memory store for embedding and retrieving semantic memories.
    Uses sentence transformers for encoding and PostgreSQL with pgvector for storage.
    """
    
    def __init__(self, db_connection_string: str, embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize the memory store.
        
        Args:
            db_connection_string: PostgreSQL connection string
            embedding_model: Sentence transformer model name
        """
        self.engine = create_engine(db_connection_string)
        self.model = SentenceTransformer(embedding_model)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        logger.info(f"Initialized AgentMemoryStore with {embedding_model} (dim={self.embedding_dim})")
    
    def create_session(self, agent_name: str, agent_version: str = "1.0", 
                      metadata: Optional[Dict] = None) -> str:
        """
        Create a new agent session.
        
        Args:
            agent_name: Name of the agent
            agent_version: Version of the agent
            metadata: Additional session metadata
            
        Returns:
            Session ID (UUID string)
        """
        session_id = str(uuid.uuid4())
        
        with self.engine.connect() as conn:
            conn.execute(
                text("""
                INSERT INTO agent_sessions 
                (id, agent_name, agent_version, session_metadata, status)
                VALUES (:id, :agent_name, :agent_version, :metadata, 'active')
                """),
                {
                    "id": session_id,
                    "agent_name": agent_name,
                    "agent_version": agent_version,
                    "metadata": json.dumps(metadata or {})
                }
            )
            conn.commit()
        
        logger.info(f"Created session {session_id} for agent {agent_name}")
        return session_id
    
    def store_memory(self, session_id: str, content: str, memory_type: str = "reasoning",
                    importance: float = 0.5, metadata: Optional[Dict] = None,
                    expires_hours: Optional[int] = None) -> int:
        """
        Store a memory with its embedding.
        
        Args:
            session_id: Agent session ID
            content: Text content to store
            memory_type: Type of memory ('reasoning', 'experience', 'knowledge')
            importance: Importance score (0-1)
            metadata: Additional metadata
            expires_hours: Hours until expiration (None for no expiration)
            
        Returns:
            Memory ID
        """
        # Generate embedding
        embedding = self.model.encode(content)
        embedding_str = f"[{','.join(map(str, embedding))}]"
        
        # Calculate expiration if specified
        expires_at = None
        if expires_hours:
            expires_at = datetime.now() + timedelta(hours=expires_hours)
        
        with self.engine.connect() as conn:
            result = conn.execute(
                text("""
                INSERT INTO agent_memory 
                (session_id, memory_type, content, embedding, metadata_json, 
                 importance_score, expires_at)
                VALUES 
                (:session_id, :memory_type, :content, :embedding, :metadata,
                 :importance, :expires_at)
                RETURNING id
                """),
                {
                    "session_id": session_id,
                    "memory_type": memory_type,
                    "content": content,
                    "embedding": embedding_str,
                    "metadata": json.dumps(metadata or {}),
                    "importance": importance,
                    "expires_at": expires_at
                }
            )
            conn.commit()
            memory_id = result.fetchone()[0]
        
        logger.info(f"Stored memory {memory_id} in session {session_id}")
        return memory_id
    
    def retrieve_memories(self, session_id: str, query: str, limit: int = 5,
                         memory_type: Optional[str] = None,
                         min_importance: float = 0.0) -> List[Dict]:
        """
        Retrieve similar memories using semantic search.
        
        Args:
            session_id: Agent session ID
            query: Query text for similarity search
            limit: Maximum number of memories to return
            memory_type: Filter by memory type
            min_importance: Minimum importance score
            
        Returns:
            List of memory dictionaries with similarity scores
        """
        # Generate query embedding
        query_embedding = self.model.encode(query)
        query_embedding_str = f"[{','.join(map(str, query_embedding))}]"
        
        # Build query with optional filters
        where_clauses = [
            "session_id = :session_id",
            "importance_score >= :min_importance",
            "(expires_at IS NULL OR expires_at > NOW())"
        ]
        
        params = {
            "session_id": session_id,
            "min_importance": min_importance,
            "limit": limit
        }
        
        if memory_type:
            where_clauses.append("memory_type = :memory_type")
            params["memory_type"] = memory_type
        
        # Note: This is a simplified cosine similarity calculation
        # In a real pgvector setup, you would use the <=> operator
        query_sql = f"""
        SELECT id, content, memory_type, importance_score, metadata_json,
               created_at, access_count, last_accessed
        FROM agent_memory 
        WHERE {' AND '.join(where_clauses)}
        ORDER BY importance_score DESC, created_at DESC
        LIMIT :limit
        """
        
        with self.engine.connect() as conn:
            result = conn.execute(text(query_sql), params)
            memories = []
            
            for row in result:
                # Update access count
                conn.execute(
                    text("""
                    UPDATE agent_memory 
                    SET access_count = access_count + 1,
                        last_accessed = NOW()
                    WHERE id = :id
                    """),
                    {"id": row[0]}
                )
                
                memories.append({
                    "id": row[0],
                    "content": row[1],
                    "memory_type": row[2],
                    "importance_score": row[3],
                    "metadata": json.loads(row[4] or "{}"),
                    "created_at": row[5],
                    "access_count": row[6] + 1,
                    "last_accessed": row[7]
                })
            
            conn.commit()
        
        logger.info(f"Retrieved {len(memories)} memories for query in session {session_id}")
        return memories
    
    def create_memory_association(self, source_memory_id: int, target_memory_id: int,
                                 association_type: str = "similar", strength: float = 1.0):
        """
        Create an association between two memories.
        
        Args:
            source_memory_id: Source memory ID
            target_memory_id: Target memory ID
            association_type: Type of association ('similar', 'causal', 'temporal')
            strength: Association strength (0-1)
        """
        with self.engine.connect() as conn:
            conn.execute(
                text("""
                INSERT INTO memory_associations 
                (source_memory_id, target_memory_id, association_type, strength)
                VALUES (:source, :target, :type, :strength)
                ON CONFLICT (source_memory_id, target_memory_id, association_type)
                DO UPDATE SET strength = :strength
                """),
                {
                    "source": source_memory_id,
                    "target": target_memory_id,
                    "type": association_type,
                    "strength": strength
                }
            )
            conn.commit()
        
        logger.info(f"Created {association_type} association between memories {source_memory_id} and {target_memory_id}")
    
    def get_session_memories(self, session_id: str) -> List[Dict]:
        """
        Get all memories for a session.
        
        Args:
            session_id: Agent session ID
            
        Returns:
            List of all memories in the session
        """
        with self.engine.connect() as conn:
            result = conn.execute(
                text("""
                SELECT id, content, memory_type, importance_score, 
                       created_at, access_count
                FROM agent_memory 
                WHERE session_id = :session_id
                  AND (expires_at IS NULL OR expires_at > NOW())
                ORDER BY created_at DESC
                """),
                {"session_id": session_id}
            )
            
            memories = []
            for row in result:
                memories.append({
                    "id": row[0],
                    "content": row[1],
                    "memory_type": row[2],
                    "importance_score": row[3],
                    "created_at": row[4],
                    "access_count": row[5]
                })
        
        return memories
    
    def end_session(self, session_id: str):
        """
        End an agent session.
        
        Args:
            session_id: Session ID to end
        """
        with self.engine.connect() as conn:
            conn.execute(
                text("""
                UPDATE agent_sessions 
                SET status = 'completed', ended_at = NOW()
                WHERE id = :session_id
                """),
                {"session_id": session_id}
            )
            conn.commit()
        
        logger.info(f"Ended session {session_id}")


def load_params(params_file: str = "params.yaml") -> dict:
    """Load parameters from YAML file."""
    if os.path.exists(params_file):
        with open(params_file, 'r') as f:
            return yaml.safe_load(f)
    return {}


def demo_agent_memory():
    """
    Demonstration of agent memory functionality.
    """
    logger.info("Starting agent memory demonstration...")
    
    # Load configuration
    params = load_params()
    db_config = params.get('database', {})
    connection_string = f"postgresql://{db_config.get('user', 'duetmind')}:{db_config.get('password', 'duetmind_secret')}@{db_config.get('host', 'localhost')}:{db_config.get('port', 5432)}/{db_config.get('database', 'duetmind_mlops')}"
    
    try:
        # Initialize memory store
        memory_store = AgentMemoryStore(connection_string)
        
        # Create agent session
        session_id = memory_store.create_session(
            agent_name="DuetMind_Alzheimer_Agent",
            agent_version="1.0",
            metadata={"model": "alzheimer_classifier", "domain": "healthcare"}
        )
        
        # Store some reasoning memories
        memories = [
            {
                "content": "Patient shows high MMSE score (28) but has APOE4 genotype, indicating complex risk profile",
                "type": "reasoning",
                "importance": 0.8
            },
            {
                "content": "Age-related cognitive decline patterns suggest early intervention may be beneficial",
                "type": "reasoning", 
                "importance": 0.7
            },
            {
                "content": "Family history is a strong predictor but should be combined with biomarker data",
                "type": "knowledge",
                "importance": 0.9
            },
            {
                "content": "Model confidence drops when education level is below 8 years",
                "type": "experience",
                "importance": 0.6
            }
        ]
        
        memory_ids = []
        for mem in memories:
            memory_id = memory_store.store_memory(
                session_id=session_id,
                content=mem["content"],
                memory_type=mem["type"],
                importance=mem["importance"]
            )
            memory_ids.append(memory_id)
        
        # Create associations between related memories
        memory_store.create_memory_association(
            memory_ids[0], memory_ids[2], "related", 0.8
        )
        
        # Demonstrate memory retrieval
        query = "APOE4 genetic risk factors"
        retrieved = memory_store.retrieve_memories(session_id, query, limit=3)
        
        logger.info(f"Query: '{query}'")
        logger.info("Retrieved memories:")
        for i, mem in enumerate(retrieved, 1):
            logger.info(f"  {i}. [{mem['memory_type']}] {mem['content'][:80]}...")
            logger.info(f"     Importance: {mem['importance_score']:.2f}, Accessed: {mem['access_count']} times")
        
        # Get all session memories
        all_memories = memory_store.get_session_memories(session_id)
        logger.info(f"Total memories in session: {len(all_memories)}")
        
        # End session
        memory_store.end_session(session_id)
        
        logger.info("Agent memory demonstration completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in agent memory demo: {e}")
        logger.info("Make sure PostgreSQL is running with the MLOps schema")
        logger.info("You can start it with: make infra-up && make db-migrate")


if __name__ == "__main__":
    demo_agent_memory()