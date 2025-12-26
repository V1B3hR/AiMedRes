"""
Advanced Memory Consolidation for DuetMind Adaptive Agent Memory.

Implements biological-inspired memory consolidation with dual-store architecture,
priority replay, synaptic tagging, and semantic conflict resolution.

Phase 1 Features:
- Dual-store architecture (EpisodicStore + SemanticStore)
- Memory metadata (recency, frequency, utility, clinical relevance)
- Basic consolidation scheduler
- Memory salience scoring system

Phase 2 Features:
- Priority replay with weighted sampling by novelty * uncertainty * reward
- Synaptic tagging for high-reward episodes
- Generative rehearsal with periodic summarization
- Advanced consolidation scheduling with idle time detection

Phase 3 Features:
- Semantic conflict resolver for contradicting facts
- Memory introspection API for decision traceability
- Enhanced biological plausibility with replay sampling
"""

import json
import logging
import sqlite3
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .embed_memory import AgentMemoryStore

logger = logging.getLogger("duetmind.memory.consolidation")


class MemoryStore(Enum):
    """Types of memory stores."""

    EPISODIC = "episodic"  # Raw events, time-indexed
    SEMANTIC = "semantic"  # Normalized facts, embeddings


@dataclass
class MemoryMetadata:
    """Enhanced metadata for memory entries."""

    recency_score: float = 0.0  # How recent the memory is (0-1)
    frequency_score: float = 0.0  # How often accessed (0-1)
    utility_score: float = 0.0  # How useful for reasoning (0-1)
    clinical_relevance: float = 0.0  # Clinical importance (0-1)
    novelty_score: float = 0.0  # How novel/unique (0-1)
    emotional_valence: float = 0.5  # Emotional weight (0-1, 0.5=neutral)
    consolidation_priority: float = 0.0  # Priority for consolidation (0-1)
    uncertainty_score: float = 0.0  # Epistemic uncertainty (0-1)
    reward_signal: float = 0.0  # Reward/feedback signal (-1 to 1)
    synaptic_tag: bool = False  # Tagged for priority consolidation
    last_consolidated: Optional[datetime] = None
    consolidation_count: int = 0
    conflict_flag: bool = False  # Marks semantic conflicts
    source_references: List[int] = None  # References to source memories

    def __post_init__(self):
        if self.source_references is None:
            self.source_references = []


@dataclass
class SemanticConflict:
    """Represents a conflict between semantic memories."""

    conflict_id: str
    memory_id1: int
    memory_id2: int
    conflict_type: str  # 'contradiction', 'inconsistency', 'outdated'
    confidence_score: float  # How confident we are about the conflict (0-1)
    resolution_status: str  # 'pending', 'resolved', 'escalated'
    created_at: datetime
    resolved_at: Optional[datetime] = None
    resolution_method: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ConsolidationEvent:
    """Represents a memory consolidation event."""

    event_id: str
    timestamp: datetime
    source_store: MemoryStore
    target_store: MemoryStore
    memory_ids: List[int]
    consolidation_type: str  # 'summarize', 'merge', 'promote', 'prune', 'replay', 'rehearsal'
    metadata: Dict[str, Any]


class MemoryConsolidator:
    """
    Advanced memory consolidation system for DuetMind Adaptive.

    Implements biological-inspired memory consolidation with:
    - Dual episodic/semantic stores
    - Salience-based consolidation prioritization
    - Priority replay with synaptic tagging
    - Scheduled consolidation windows
    - Semantic conflict resolution
    - Controlled forgetting mechanisms
    - Memory introspection API
    """

    def __init__(self, memory_store: AgentMemoryStore, config: Optional[Dict[str, Any]] = None):
        self.memory_store = memory_store
        self.config = config or {}

        # Consolidation parameters
        self.consolidation_interval_hours = self.config.get("consolidation_interval_hours", 6)
        self.max_episodic_memories = self.config.get("max_episodic_memories", 1000)
        self.min_salience_threshold = self.config.get("min_salience_threshold", 0.3)
        self.novelty_weight = self.config.get("novelty_weight", 0.25)
        self.frequency_weight = self.config.get("frequency_weight", 0.25)
        self.clinical_weight = self.config.get("clinical_weight", 0.3)
        self.recency_weight = self.config.get("recency_weight", 0.2)

        # Phase 2 parameters - Priority Replay
        self.replay_sample_size = self.config.get("replay_sample_size", 50)
        self.synaptic_tag_threshold = self.config.get("synaptic_tag_threshold", 0.8)
        self.rehearsal_interval_hours = self.config.get("rehearsal_interval_hours", 24)
        self.uncertainty_weight = self.config.get("uncertainty_weight", 0.3)
        self.reward_weight = self.config.get("reward_weight", 0.4)

        # Phase 3 parameters - Conflict Resolution
        self.conflict_detection_threshold = self.config.get("conflict_detection_threshold", 0.7)
        self.semantic_similarity_threshold = self.config.get("semantic_similarity_threshold", 0.85)

        # Consolidation database
        self.consolidation_db = self.config.get("consolidation_db", "memory_consolidation.db")
        self._init_consolidation_db()

        # Consolidation state
        self.consolidation_thread = None
        self.running = False
        self.last_consolidation = None
        self.last_rehearsal = None

        logger.info("Memory Consolidator initialized with Phase 2 & 3 enhancements")

    def _init_consolidation_db(self):
        """Initialize consolidation tracking database."""
        with sqlite3.connect(self.consolidation_db) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_metadata (
                    memory_id INTEGER PRIMARY KEY,
                    memory_store TEXT NOT NULL,
                    recency_score REAL DEFAULT 0.0,
                    frequency_score REAL DEFAULT 0.0,
                    utility_score REAL DEFAULT 0.0,
                    clinical_relevance REAL DEFAULT 0.0,
                    novelty_score REAL DEFAULT 0.0,
                    emotional_valence REAL DEFAULT 0.5,
                    consolidation_priority REAL DEFAULT 0.0,
                    uncertainty_score REAL DEFAULT 0.0,
                    reward_signal REAL DEFAULT 0.0,
                    synaptic_tag INTEGER DEFAULT 0,
                    conflict_flag INTEGER DEFAULT 0,
                    source_references TEXT DEFAULT '[]',
                    last_consolidated TEXT,
                    consolidation_count INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS consolidation_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT UNIQUE NOT NULL,
                    timestamp TEXT NOT NULL,
                    source_store TEXT NOT NULL,
                    target_store TEXT NOT NULL,
                    memory_ids TEXT NOT NULL,
                    consolidation_type TEXT NOT NULL,
                    metadata_json TEXT,
                    success INTEGER DEFAULT 1
                )
            """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS semantic_clusters (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cluster_id TEXT UNIQUE NOT NULL,
                    cluster_centroid TEXT,
                    memory_ids TEXT NOT NULL,
                    cluster_summary TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """
            )

            # Phase 3: Semantic conflicts table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS semantic_conflicts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conflict_id TEXT UNIQUE NOT NULL,
                    memory_id1 INTEGER NOT NULL,
                    memory_id2 INTEGER NOT NULL,
                    conflict_type TEXT NOT NULL,
                    confidence_score REAL NOT NULL,
                    resolution_status TEXT DEFAULT 'pending',
                    created_at TEXT NOT NULL,
                    resolved_at TEXT,
                    resolution_method TEXT,
                    metadata_json TEXT
                )
            """
            )

            # Phase 2: Priority replay tracking
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS replay_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    replay_id TEXT UNIQUE NOT NULL,
                    session_id TEXT NOT NULL,
                    memory_ids TEXT NOT NULL,
                    replay_priority_scores TEXT NOT NULL,
                    replay_type TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    success INTEGER DEFAULT 1,
                    metadata_json TEXT
                )
            """
            )

            # Create indexes
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_metadata_priority ON memory_metadata(consolidation_priority)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_metadata_store ON memory_metadata(memory_store)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_metadata_synaptic_tag ON memory_metadata(synaptic_tag)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_events_timestamp ON consolidation_events(timestamp)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_conflicts_status ON semantic_conflicts(resolution_status)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_replay_session ON replay_events(session_id)"
            )

            conn.commit()

    def calculate_salience_score(
        self,
        memory_id: int,
        session_id: str,
        uncertainty_score: float = 0.0,
        reward_signal: float = 0.0,
    ) -> float:
        """
        Calculate enhanced salience score for a memory based on multiple factors.

        Args:
            memory_id: Memory ID to score
            session_id: Session ID for context
            uncertainty_score: Epistemic uncertainty (0-1)
            reward_signal: Reward/feedback signal (-1 to 1)

        Returns:
            Salience score (0-1)
        """
        try:
            # Get memory details from main store
            memories = self.memory_store.get_session_memories(session_id)
            target_memory = next((m for m in memories if m["id"] == memory_id), None)

            if not target_memory:
                return 0.0

            # Calculate component scores
            recency_score = self._calculate_recency_score(target_memory["created_at"])
            frequency_score = self._calculate_frequency_score(target_memory["access_count"])
            utility_score = target_memory.get("importance_score", 0.5)
            clinical_relevance = self._calculate_clinical_relevance(target_memory["content"])
            novelty_score = self._calculate_novelty_score(target_memory, memories)

            # Phase 2 Enhancement: Include uncertainty and reward in salience calculation
            # S = αnovelty + βfrequency + γclinical_importance + δreward_delta + εuncertainty
            salience = (
                self.novelty_weight * novelty_score
                + self.frequency_weight * frequency_score
                + self.clinical_weight * clinical_relevance
                + self.recency_weight * recency_score
                + self.uncertainty_weight * uncertainty_score
                + self.reward_weight * max(0, reward_signal)  # Only positive rewards boost salience
                + 0.05 * utility_score  # Base importance
            )

            # Determine if memory should receive synaptic tag
            synaptic_tag = salience >= self.synaptic_tag_threshold and reward_signal > 0.5

            # Store enhanced metadata
            self._store_memory_metadata(
                memory_id,
                MemoryStore.EPISODIC,  # Assuming new memories start episodic
                MemoryMetadata(
                    recency_score=recency_score,
                    frequency_score=frequency_score,
                    utility_score=utility_score,
                    clinical_relevance=clinical_relevance,
                    novelty_score=novelty_score,
                    uncertainty_score=uncertainty_score,
                    reward_signal=reward_signal,
                    synaptic_tag=synaptic_tag,
                    consolidation_priority=salience,
                ),
            )

            return min(1.0, max(0.0, salience))

        except Exception as e:
            logger.error(f"Error calculating salience score: {e}")
            return 0.5  # Default moderate salience

    def _calculate_recency_score(self, created_at: datetime) -> float:
        """Calculate recency score (more recent = higher score)."""
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))

        hours_ago = (datetime.now() - created_at).total_seconds() / 3600

        # Exponential decay over 7 days (168 hours)
        decay_rate = 0.693 / 168  # Half-life of 1 week
        recency_score = np.exp(-decay_rate * hours_ago)

        return min(1.0, max(0.0, recency_score))

    def _calculate_frequency_score(self, access_count: int) -> float:
        """Calculate frequency score based on access patterns."""
        # Logarithmic scaling to prevent extremely high frequencies from dominating
        if access_count <= 0:
            return 0.0

        # Scale: 1 access = 0.1, 10 accesses = 0.3, 100 accesses = 0.5, etc.
        frequency_score = min(1.0, np.log10(access_count + 1) / 3.0)
        return frequency_score

    def _calculate_clinical_relevance(self, content: str) -> float:
        """Calculate clinical relevance score based on content analysis."""
        clinical_keywords = [
            "diagnosis",
            "treatment",
            "patient",
            "symptom",
            "medication",
            "clinical",
            "medical",
            "health",
            "disease",
            "therapy",
            "MMSE",
            "cognitive",
            "Alzheimer",
            "dementia",
            "biomarker",
            "APOE",
            "genetic",
            "risk",
            "assessment",
            "guideline",
        ]

        content_lower = content.lower()
        matches = sum(1 for keyword in clinical_keywords if keyword.lower() in content_lower)

        # Scale based on keyword density
        clinical_score = min(1.0, matches / 5.0)  # 5+ keywords = max score
        return clinical_score

    def _calculate_novelty_score(
        self, target_memory: Dict[str, Any], all_memories: List[Dict[str, Any]]
    ) -> float:
        """Calculate novelty score by comparing with existing memories."""
        if len(all_memories) <= 1:
            return 1.0  # First memory is always novel

        target_content = target_memory["content"].lower()
        target_words = set(target_content.split())

        similarities = []
        for memory in all_memories:
            if memory["id"] == target_memory["id"]:
                continue  # Skip self

            memory_words = set(memory["content"].lower().split())
            if len(target_words) == 0 or len(memory_words) == 0:
                continue

            # Jaccard similarity
            intersection = len(target_words.intersection(memory_words))
            union = len(target_words.union(memory_words))
            similarity = intersection / union if union > 0 else 0.0
            similarities.append(similarity)

        # Novelty is inverse of maximum similarity
        max_similarity = max(similarities) if similarities else 0.0
        novelty_score = 1.0 - max_similarity

        return novelty_score

    def _store_memory_metadata(
        self, memory_id: int, store_type: MemoryStore, metadata: MemoryMetadata
    ):
        """Store enhanced memory metadata in consolidation database."""
        with sqlite3.connect(self.consolidation_db) as conn:
            now = datetime.now().isoformat()

            conn.execute(
                """
                INSERT OR REPLACE INTO memory_metadata (
                    memory_id, memory_store, recency_score, frequency_score,
                    utility_score, clinical_relevance, novelty_score, emotional_valence,
                    consolidation_priority, uncertainty_score, reward_signal, synaptic_tag,
                    conflict_flag, source_references, last_consolidated, consolidation_count,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    memory_id,
                    store_type.value,
                    metadata.recency_score,
                    metadata.frequency_score,
                    metadata.utility_score,
                    metadata.clinical_relevance,
                    metadata.novelty_score,
                    metadata.emotional_valence,
                    metadata.consolidation_priority,
                    metadata.uncertainty_score,
                    metadata.reward_signal,
                    int(metadata.synaptic_tag),
                    int(metadata.conflict_flag),
                    json.dumps(metadata.source_references),
                    metadata.last_consolidated.isoformat() if metadata.last_consolidated else None,
                    metadata.consolidation_count,
                    now,
                    now,
                ),
            )
            conn.commit()

    def run_consolidation_cycle(self, session_id: str) -> Dict[str, Any]:
        """
        Run a complete consolidation cycle for a session.

        Args:
            session_id: Session to consolidate

        Returns:
            Consolidation results summary
        """
        start_time = datetime.now()
        results = {
            "session_id": session_id,
            "start_time": start_time.isoformat(),
            "memories_processed": 0,
            "memories_consolidated": 0,
            "memories_pruned": 0,
            "semantic_clusters_created": 0,
            "events": [],
        }

        try:
            # Get all memories for session
            memories = self.memory_store.get_session_memories(session_id)
            results["memories_processed"] = len(memories)

            if not memories:
                logger.info(f"No memories to consolidate for session {session_id}")
                return results

            # Calculate salience scores for all memories
            memory_salience = {}
            for memory in memories:
                salience = self.calculate_salience_score(memory["id"], session_id)
                memory_salience[memory["id"]] = salience

            # Sort memories by salience (highest first)
            sorted_memories = sorted(memories, key=lambda m: memory_salience[m["id"]], reverse=True)

            # Phase 1: Promote high-salience episodic memories to semantic store
            promoted_count = self._promote_to_semantic(sorted_memories, memory_salience, session_id)
            results["memories_consolidated"] += promoted_count

            # Phase 2: Priority Replay - sample memories for enhanced consolidation
            replay_count = self._run_priority_replay(sorted_memories, memory_salience, session_id)
            results["replay_events"] = replay_count

            # Phase 2: Create semantic clusters from related memories
            clusters_created = self._create_semantic_clusters(sorted_memories, session_id)
            results["semantic_clusters_created"] = clusters_created

            # Phase 3: Detect and resolve semantic conflicts
            conflicts_detected = self._detect_semantic_conflicts(sorted_memories, session_id)
            results["conflicts_detected"] = conflicts_detected

            # Phase 2: Generative rehearsal if enough time has passed
            if self._should_run_rehearsal():
                rehearsal_count = self._run_generative_rehearsal(session_id)
                results["rehearsal_summaries"] = rehearsal_count

            # Phase 3: Prune low-salience memories
            pruned_count = self._prune_low_salience_memories(
                sorted_memories, memory_salience, session_id
            )
            results["memories_pruned"] = pruned_count

            # Update consolidation timestamp
            self.last_consolidation = datetime.now()

            logger.info(
                f"Consolidation completed for session {session_id}: "
                f"{promoted_count} promoted, {clusters_created} clusters, {pruned_count} pruned"
            )

        except Exception as e:
            logger.error(f"Error in consolidation cycle: {e}")
            results["error"] = str(e)

        results["end_time"] = datetime.now().isoformat()
        results["duration_seconds"] = (datetime.now() - start_time).total_seconds()

        return results

    def _promote_to_semantic(
        self, memories: List[Dict[str, Any]], salience_scores: Dict[int, float], session_id: str
    ) -> int:
        """Promote high-salience episodic memories to semantic store."""
        promoted_count = 0
        promotion_threshold = 0.7  # Only promote highly salient memories

        for memory in memories:
            salience = salience_scores[memory["id"]]

            if salience >= promotion_threshold and memory["memory_type"] != "semantic":
                try:
                    # Create consolidated semantic memory
                    semantic_content = self._create_semantic_summary(memory)

                    # Store as new semantic memory
                    semantic_id = self.memory_store.store_memory(
                        session_id=session_id,
                        content=semantic_content,
                        memory_type="semantic",
                        importance=salience,
                        metadata={
                            "source_memory_id": memory["id"],
                            "consolidation_type": "promotion",
                            "original_type": memory["memory_type"],
                        },
                    )

                    # Update metadata
                    metadata = MemoryMetadata(
                        consolidation_priority=salience,
                        last_consolidated=datetime.now(),
                        consolidation_count=1,
                    )
                    self._store_memory_metadata(semantic_id, MemoryStore.SEMANTIC, metadata)

                    # Log consolidation event
                    self._log_consolidation_event(
                        source_store=MemoryStore.EPISODIC,
                        target_store=MemoryStore.SEMANTIC,
                        memory_ids=[memory["id"], semantic_id],
                        consolidation_type="promotion",
                        metadata={"salience_score": salience},
                    )

                    promoted_count += 1

                except Exception as e:
                    logger.error(f"Error promoting memory {memory['id']}: {e}")

        return promoted_count

    def _create_semantic_summary(self, memory: Dict[str, Any]) -> str:
        """Create a semantic summary of an episodic memory."""
        content = memory["content"]
        memory_type = memory["memory_type"]

        # Basic semantic extraction (could be enhanced with NLP)
        if memory_type == "reasoning":
            # Extract key reasoning patterns
            if "MMSE" in content:
                return f"Cognitive assessment insight: {content[:150]}..."
            elif "APOE" in content:
                return f"Genetic risk factor analysis: {content[:150]}..."
            else:
                return f"Clinical reasoning: {content[:150]}..."
        elif memory_type == "experience":
            return f"Clinical experience: {content[:150]}..."
        else:
            return f"Knowledge: {content[:150]}..."

    def _create_semantic_clusters(self, memories: List[Dict[str, Any]], session_id: str) -> int:
        """Create semantic clusters from related memories."""
        clusters_created = 0

        # Group memories by content similarity (simplified clustering)
        semantic_memories = [m for m in memories if m["memory_type"] == "semantic"]

        if len(semantic_memories) < 3:  # Need at least 3 for clustering
            return 0

        # Simple keyword-based clustering
        clusters = self._simple_keyword_clustering(semantic_memories)

        for cluster_keywords, cluster_memories in clusters.items():
            if len(cluster_memories) >= 2:  # At least 2 memories per cluster
                try:
                    cluster_id = str(uuid.uuid4())
                    memory_ids = [m["id"] for m in cluster_memories]

                    # Create cluster summary
                    cluster_summary = f"Semantic cluster: {cluster_keywords}. "
                    cluster_summary += f"Contains {len(cluster_memories)} related memories about "
                    cluster_summary += f"{', '.join(cluster_keywords.split('_'))}"

                    # Store cluster
                    with sqlite3.connect(self.consolidation_db) as conn:
                        now = datetime.now().isoformat()
                        conn.execute(
                            """
                            INSERT INTO semantic_clusters 
                            (cluster_id, memory_ids, cluster_summary, created_at, updated_at)
                            VALUES (?, ?, ?, ?, ?)
                        """,
                            (cluster_id, json.dumps(memory_ids), cluster_summary, now, now),
                        )
                        conn.commit()

                    clusters_created += 1

                except Exception as e:
                    logger.error(f"Error creating semantic cluster: {e}")

        return clusters_created

    def _simple_keyword_clustering(
        self, memories: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Simple keyword-based clustering of memories."""
        clusters = {}

        clinical_topics = {
            "cognitive_assessment": ["MMSE", "cognitive", "assessment", "testing"],
            "genetic_risk": ["APOE", "genetic", "gene", "hereditary"],
            "biomarkers": ["biomarker", "protein", "tau", "amyloid"],
            "treatment": ["treatment", "therapy", "intervention", "medication"],
            "diagnosis": ["diagnosis", "diagnostic", "differential", "criteria"],
        }

        for memory in memories:
            content_lower = memory["content"].lower()

            for topic, keywords in clinical_topics.items():
                if any(keyword.lower() in content_lower for keyword in keywords):
                    if topic not in clusters:
                        clusters[topic] = []
                    clusters[topic].append(memory)
                    break  # Assign to first matching cluster only

        return clusters

    def _prune_low_salience_memories(
        self, memories: List[Dict[str, Any]], salience_scores: Dict[int, float], session_id: str
    ) -> int:
        """Prune memories with very low salience scores."""
        pruned_count = 0
        prune_threshold = self.min_salience_threshold

        # Only prune if we have too many memories
        if len(memories) <= self.max_episodic_memories:
            return 0

        # Sort by salience (lowest first for pruning)
        low_salience_memories = [m for m in memories if salience_scores[m["id"]] < prune_threshold]

        # Prune oldest, lowest salience memories
        low_salience_memories.sort(key=lambda m: m["created_at"])

        memories_to_prune = low_salience_memories[: len(memories) - self.max_episodic_memories]

        for memory in memories_to_prune:
            try:
                # Mark as pruned (don't actually delete for safety)
                # In production, could implement soft delete or archival
                self._log_consolidation_event(
                    source_store=MemoryStore.EPISODIC,
                    target_store=MemoryStore.EPISODIC,  # Same store, just marking
                    memory_ids=[memory["id"]],
                    consolidation_type="prune",
                    metadata={"salience_score": salience_scores[memory["id"]]},
                )

                pruned_count += 1

            except Exception as e:
                logger.error(f"Error pruning memory {memory['id']}: {e}")

        return pruned_count

    def _run_priority_replay(
        self, memories: List[Dict[str, Any]], salience_scores: Dict[int, float], session_id: str
    ) -> int:
        """
        Phase 2: Run priority replay - weighted sampling by novelty * uncertainty * reward.
        """
        replay_count = 0

        try:
            # Get memories with metadata for priority calculation
            replay_candidates = []

            with sqlite3.connect(self.consolidation_db) as conn:
                for memory in memories:
                    cursor = conn.execute(
                        """
                        SELECT uncertainty_score, reward_signal, novelty_score, synaptic_tag
                        FROM memory_metadata WHERE memory_id = ?
                    """,
                        (memory["id"],),
                    )

                    row = cursor.fetchone()
                    if row:
                        uncertainty, reward, novelty, synaptic_tag = row

                        # Priority Replay Score = novelty * uncertainty * reward (with synaptic boost)
                        replay_priority = novelty * uncertainty * max(0, reward)
                        if synaptic_tag:
                            replay_priority *= 1.5  # Boost for synaptic tagged memories

                        replay_candidates.append(
                            {
                                "memory": memory,
                                "priority": replay_priority,
                                "salience": salience_scores[memory["id"]],
                            }
                        )

            # Sort by replay priority and sample top candidates
            replay_candidates.sort(key=lambda x: x["priority"], reverse=True)
            selected_for_replay = replay_candidates[
                : min(self.replay_sample_size, len(replay_candidates))
            ]

            if selected_for_replay:
                # Record replay event
                replay_id = str(uuid.uuid4())
                memory_ids = [c["memory"]["id"] for c in selected_for_replay]
                priority_scores = [c["priority"] for c in selected_for_replay]

                with sqlite3.connect(self.consolidation_db) as conn:
                    conn.execute(
                        """
                        INSERT INTO replay_events 
                        (replay_id, session_id, memory_ids, replay_priority_scores, 
                         replay_type, timestamp, metadata_json)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            replay_id,
                            session_id,
                            json.dumps(memory_ids),
                            json.dumps(priority_scores),
                            "priority_replay",
                            datetime.now().isoformat(),
                            json.dumps({"sample_size": len(selected_for_replay)}),
                        ),
                    )
                    conn.commit()

                replay_count = len(selected_for_replay)
                logger.info(f"Priority replay completed: {replay_count} memories processed")

        except Exception as e:
            logger.error(f"Error in priority replay: {e}")

        return replay_count

    def _should_run_rehearsal(self) -> bool:
        """Check if generative rehearsal should run."""
        if not self.last_rehearsal:
            return True

        hours_since = (datetime.now() - self.last_rehearsal).total_seconds() / 3600
        return hours_since >= self.rehearsal_interval_hours

    def _run_generative_rehearsal(self, session_id: str) -> int:
        """
        Phase 2: Run generative rehearsal - periodic summarization of memories.
        """
        rehearsal_count = 0

        try:
            # Get high-salience memories for rehearsal
            with sqlite3.connect(self.consolidation_db) as conn:
                cursor = conn.execute(
                    """
                    SELECT memory_id, consolidation_priority 
                    FROM memory_metadata 
                    WHERE consolidation_priority > 0.6
                    ORDER BY consolidation_priority DESC
                    LIMIT 20
                """
                )

                high_salience_ids = [row[0] for row in cursor.fetchall()]

            if high_salience_ids:
                # Get full memory details
                memories = self.memory_store.get_session_memories(session_id)
                rehearsal_memories = [m for m in memories if m["id"] in high_salience_ids]

                # Create rehearsal summary
                summary_content = self._create_rehearsal_summary(rehearsal_memories)

                # Store as new semantic memory
                summary_id = self.memory_store.store_memory(
                    session_id=session_id,
                    content=summary_content,
                    memory_type="semantic",
                    importance=0.8,
                    metadata={
                        "consolidation_type": "rehearsal",
                        "source_memory_count": len(rehearsal_memories),
                        "rehearsal_timestamp": datetime.now().isoformat(),
                    },
                )

                rehearsal_count = 1
                self.last_rehearsal = datetime.now()

                # Log rehearsal event
                self._log_consolidation_event(
                    source_store=MemoryStore.EPISODIC,
                    target_store=MemoryStore.SEMANTIC,
                    memory_ids=[summary_id],
                    consolidation_type="rehearsal",
                    metadata={"source_memories": high_salience_ids},
                )

                logger.info(
                    f"Generative rehearsal completed: {len(rehearsal_memories)} memories summarized"
                )

        except Exception as e:
            logger.error(f"Error in generative rehearsal: {e}")

        return rehearsal_count

    def _create_rehearsal_summary(self, memories: List[Dict[str, Any]]) -> str:
        """Create a generative summary of multiple memories for rehearsal."""
        if not memories:
            return "Empty rehearsal summary"

        # Group memories by type
        memory_groups = {}
        for memory in memories:
            mem_type = memory.get("memory_type", "general")
            if mem_type not in memory_groups:
                memory_groups[mem_type] = []
            memory_groups[mem_type].append(memory["content"])

        # Create structured summary
        summary_parts = [f"Rehearsal Summary - {datetime.now().strftime('%Y-%m-%d %H:%M')}"]

        for mem_type, contents in memory_groups.items():
            summary_parts.append(f"\n{mem_type.title()} Insights:")

            # Extract key patterns (simplified - could use NLP summarization)
            key_terms = set()
            for content in contents:
                words = content.lower().split()
                clinical_terms = [
                    w
                    for w in words
                    if any(
                        term in w
                        for term in [
                            "mmse",
                            "apoe",
                            "cognitive",
                            "diagnosis",
                            "treatment",
                            "patient",
                            "biomarker",
                            "risk",
                            "alzheimer",
                            "dementia",
                        ]
                    )
                ]
                key_terms.update(clinical_terms[:3])  # Top 3 terms per memory

            if key_terms:
                summary_parts.append(f"- Key concepts: {', '.join(sorted(key_terms))}")

            # Include most important content snippet
            if contents:
                summary_parts.append(f"- Primary insight: {contents[0][:200]}...")

        return "\n".join(summary_parts)

    def _detect_semantic_conflicts(self, memories: List[Dict[str, Any]], session_id: str) -> int:
        """
        Phase 3: Detect semantic conflicts between memories.
        """
        conflicts_detected = 0

        try:
            # Focus on semantic memories for conflict detection
            semantic_memories = [m for m in memories if m.get("memory_type") == "semantic"]

            if len(semantic_memories) < 2:
                return 0

            # Pairwise conflict detection
            for i, mem1 in enumerate(semantic_memories):
                for mem2 in semantic_memories[i + 1 :]:
                    conflict = self._check_memory_conflict(mem1, mem2)

                    if conflict:
                        conflict_id = str(uuid.uuid4())

                        # Store conflict record
                        with sqlite3.connect(self.consolidation_db) as conn:
                            conn.execute(
                                """
                                INSERT INTO semantic_conflicts 
                                (conflict_id, memory_id1, memory_id2, conflict_type, 
                                 confidence_score, created_at, metadata_json)
                                VALUES (?, ?, ?, ?, ?, ?, ?)
                            """,
                                (
                                    conflict_id,
                                    mem1["id"],
                                    mem2["id"],
                                    conflict["type"],
                                    conflict["confidence"],
                                    datetime.now().isoformat(),
                                    json.dumps(conflict["metadata"]),
                                ),
                            )

                            # Mark memories with conflict flag
                            for mem_id in [mem1["id"], mem2["id"]]:
                                conn.execute(
                                    """
                                    UPDATE memory_metadata 
                                    SET conflict_flag = 1, updated_at = ?
                                    WHERE memory_id = ?
                                """,
                                    (datetime.now().isoformat(), mem_id),
                                )

                            conn.commit()

                        conflicts_detected += 1
                        logger.info(
                            f"Semantic conflict detected: {conflict['type']} between memories {mem1['id']} and {mem2['id']}"
                        )

        except Exception as e:
            logger.error(f"Error detecting semantic conflicts: {e}")

        return conflicts_detected

    def _check_memory_conflict(self, mem1: Dict[str, Any], mem2: Dict[str, Any]) -> Optional[Dict]:
        """Check if two memories are in semantic conflict."""
        content1 = mem1["content"].lower()
        content2 = mem2["content"].lower()

        # Simple conflict detection patterns (could be enhanced with NLP)
        contradiction_patterns = [
            ("recommends", "not recommend"),
            ("shows", "does not show"),
            ("increases", "decreases"),
            ("high", "low"),
            ("positive", "negative"),
            ("effective", "ineffective"),
        ]

        # Check for contradictory statements about the same topic
        shared_terms = set(content1.split()) & set(content2.split())
        clinical_shared = [
            term
            for term in shared_terms
            if any(
                clinical in term
                for clinical in ["mmse", "apoe", "patient", "treatment", "diagnosis", "biomarker"]
            )
        ]

        if clinical_shared:  # Same clinical topic
            for positive, negative in contradiction_patterns:
                if positive in content1 and negative in content2:
                    return {
                        "type": "contradiction",
                        "confidence": 0.8,
                        "metadata": {
                            "shared_terms": clinical_shared,
                            "pattern": f"{positive} vs {negative}",
                        },
                    }
                elif negative in content1 and positive in content2:
                    return {
                        "type": "contradiction",
                        "confidence": 0.8,
                        "metadata": {
                            "shared_terms": clinical_shared,
                            "pattern": f"{negative} vs {positive}",
                        },
                    }

        # Check for outdated information (simple heuristic)
        if (
            abs((mem1["created_at"] - mem2["created_at"]).total_seconds()) > 30 * 24 * 3600
        ):  # 30 days
            similarity_score = self._calculate_content_similarity(content1, content2)
            if similarity_score > self.semantic_similarity_threshold:
                return {
                    "type": "outdated",
                    "confidence": similarity_score,
                    "metadata": {
                        "age_difference_days": abs((mem1["created_at"] - mem2["created_at"]).days),
                        "similarity_score": similarity_score,
                    },
                }

        return None

    def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """Calculate similarity between two content strings (simplified Jaccard)."""
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

    def get_memory_introspection(self, decision_context: str, session_id: str) -> Dict[str, Any]:
        """
        Phase 3: Memory introspection API - trace decision reasoning back to memories.

        Args:
            decision_context: Context or query about a decision
            session_id: Session to analyze

        Returns:
            Dictionary with decision trace and memory chain
        """
        try:
            # Get relevant memories based on context
            relevant_memories = self.memory_store.retrieve_memories(
                session_id=session_id, query=decision_context, limit=10, min_importance=0.3
            )

            # Build memory chain with metadata
            memory_chain = []
            total_influence = 0.0

            with sqlite3.connect(self.consolidation_db) as conn:
                for memory in relevant_memories:
                    cursor = conn.execute(
                        """
                        SELECT consolidation_priority, clinical_relevance, novelty_score,
                               synaptic_tag, conflict_flag, source_references
                        FROM memory_metadata WHERE memory_id = ?
                    """,
                        (memory["id"],),
                    )

                    row = cursor.fetchone()
                    if row:
                        priority, clinical, novelty, synaptic, conflict, sources = row
                        source_refs = json.loads(sources or "[]")

                        # Calculate influence weight
                        influence = priority * clinical * (1.5 if synaptic else 1.0)
                        if conflict:
                            influence *= 0.7  # Reduce influence for conflicted memories

                        total_influence += influence

                        memory_chain.append(
                            {
                                "memory_id": memory["id"],
                                "content_preview": memory["content"][:150] + "...",
                                "memory_type": memory["memory_type"],
                                "influence_weight": influence,
                                "clinical_relevance": clinical,
                                "novelty_score": novelty,
                                "has_synaptic_tag": bool(synaptic),
                                "has_conflicts": bool(conflict),
                                "source_references": source_refs,
                                "created_at": memory["created_at"],
                                "access_count": memory["access_count"],
                            }
                        )

            # Sort by influence weight
            memory_chain.sort(key=lambda x: x["influence_weight"], reverse=True)

            # Calculate confidence in the decision trace
            confidence = min(1.0, total_influence / len(memory_chain)) if memory_chain else 0.0

            # Get related conflicts
            conflicts = self._get_relevant_conflicts([m["memory_id"] for m in memory_chain])

            return {
                "decision_context": decision_context,
                "session_id": session_id,
                "memory_chain": memory_chain,
                "trace_confidence": confidence,
                "total_memories_analyzed": len(relevant_memories),
                "influential_memories": len(
                    [m for m in memory_chain if m["influence_weight"] > 0.5]
                ),
                "conflicted_memories": len([m for m in memory_chain if m["has_conflicts"]]),
                "related_conflicts": conflicts,
                "generated_at": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error in memory introspection: {e}")
            return {
                "error": str(e),
                "decision_context": decision_context,
                "session_id": session_id,
                "memory_chain": [],
                "trace_confidence": 0.0,
            }

    def _get_relevant_conflicts(self, memory_ids: List[int]) -> List[Dict]:
        """Get conflicts involving any of the given memory IDs."""
        conflicts = []

        try:
            with sqlite3.connect(self.consolidation_db) as conn:
                placeholders = ",".join(["?" for _ in memory_ids])
                cursor = conn.execute(
                    f"""
                    SELECT conflict_id, memory_id1, memory_id2, conflict_type,
                           confidence_score, resolution_status, created_at, metadata_json
                    FROM semantic_conflicts 
                    WHERE memory_id1 IN ({placeholders}) OR memory_id2 IN ({placeholders})
                """,
                    memory_ids + memory_ids,
                )

                for row in cursor.fetchall():
                    conflicts.append(
                        {
                            "conflict_id": row[0],
                            "memory_id1": row[1],
                            "memory_id2": row[2],
                            "conflict_type": row[3],
                            "confidence_score": row[4],
                            "resolution_status": row[5],
                            "created_at": row[6],
                            "metadata": json.loads(row[7] or "{}"),
                        }
                    )

        except Exception as e:
            logger.error(f"Error getting relevant conflicts: {e}")

        return conflicts

    def resolve_semantic_conflict(
        self, conflict_id: str, resolution_method: str, winning_memory_id: Optional[int] = None
    ) -> bool:
        """
        Resolve a semantic conflict.

        Args:
            conflict_id: ID of the conflict to resolve
            resolution_method: Method used ('manual', 'timestamp', 'confidence', 'escalate')
            winning_memory_id: ID of the memory to keep (if applicable)

        Returns:
            True if resolved successfully
        """
        try:
            with sqlite3.connect(self.consolidation_db) as conn:
                # Update conflict status
                conn.execute(
                    """
                    UPDATE semantic_conflicts 
                    SET resolution_status = 'resolved', 
                        resolved_at = ?, 
                        resolution_method = ?
                    WHERE conflict_id = ?
                """,
                    (datetime.now().isoformat(), resolution_method, conflict_id),
                )

                # Get conflict details
                cursor = conn.execute(
                    """
                    SELECT memory_id1, memory_id2 FROM semantic_conflicts 
                    WHERE conflict_id = ?
                """,
                    (conflict_id,),
                )

                row = cursor.fetchone()
                if row:
                    mem_id1, mem_id2 = row

                    # Clear conflict flags for both memories
                    for mem_id in [mem_id1, mem_id2]:
                        conn.execute(
                            """
                            UPDATE memory_metadata 
                            SET conflict_flag = 0, updated_at = ?
                            WHERE memory_id = ?
                        """,
                            (datetime.now().isoformat(), mem_id),
                        )

                    # If one memory is chosen as winner, optionally mark the other
                    if winning_memory_id and winning_memory_id in [mem_id1, mem_id2]:
                        losing_id = mem_id2 if winning_memory_id == mem_id1 else mem_id1

                        # Could mark losing memory with lower priority or archive it
                        conn.execute(
                            """
                            UPDATE memory_metadata 
                            SET consolidation_priority = consolidation_priority * 0.5,
                                updated_at = ?
                            WHERE memory_id = ?
                        """,
                            (datetime.now().isoformat(), losing_id),
                        )

                conn.commit()
                logger.info(f"Semantic conflict {conflict_id} resolved using {resolution_method}")
                return True

        except Exception as e:
            logger.error(f"Error resolving semantic conflict: {e}")
            return False

    def _log_consolidation_event(
        self,
        source_store: MemoryStore,
        target_store: MemoryStore,
        memory_ids: List[int],
        consolidation_type: str,
        metadata: Dict[str, Any],
    ):
        """Log a consolidation event."""
        event = ConsolidationEvent(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            source_store=source_store,
            target_store=target_store,
            memory_ids=memory_ids,
            consolidation_type=consolidation_type,
            metadata=metadata,
        )

        with sqlite3.connect(self.consolidation_db) as conn:
            conn.execute(
                """
                INSERT INTO consolidation_events 
                (event_id, timestamp, source_store, target_store, memory_ids, 
                 consolidation_type, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    event.event_id,
                    event.timestamp.isoformat(),
                    event.source_store.value,
                    event.target_store.value,
                    json.dumps(event.memory_ids),
                    event.consolidation_type,
                    json.dumps(event.metadata),
                ),
            )
            conn.commit()

    def get_consolidation_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get enhanced consolidation activity summary."""
        cutoff_time = (datetime.now() - timedelta(hours=hours)).isoformat()

        with sqlite3.connect(self.consolidation_db) as conn:
            # Overall statistics
            cursor = conn.execute(
                """
                SELECT 
                    COUNT(*) as total_events,
                    COUNT(DISTINCT consolidation_type) as event_types
                FROM consolidation_events 
                WHERE timestamp > ?
            """,
                (cutoff_time,),
            )

            stats = cursor.fetchone()

            # Events by type
            cursor = conn.execute(
                """
                SELECT consolidation_type, COUNT(*) as count
                FROM consolidation_events 
                WHERE timestamp > ?
                GROUP BY consolidation_type
            """,
                (cutoff_time,),
            )

            event_types = {row[0]: row[1] for row in cursor.fetchall()}

            # Memory store distribution
            cursor = conn.execute(
                """
                SELECT memory_store, COUNT(*) as count
                FROM memory_metadata
                GROUP BY memory_store
            """,
                (),
            )

            store_distribution = {row[0]: row[1] for row in cursor.fetchall()}

            # Phase 2 & 3 specific metrics

            # Synaptic tagged memories
            cursor = conn.execute(
                """
                SELECT COUNT(*) FROM memory_metadata WHERE synaptic_tag = 1
            """,
                (),
            )
            synaptic_tagged = cursor.fetchone()[0]

            # Replay events
            cursor = conn.execute(
                """
                SELECT COUNT(*) FROM replay_events WHERE timestamp > ?
            """,
                (cutoff_time,),
            )
            replay_events = cursor.fetchone()[0]

            # Semantic conflicts
            cursor = conn.execute(
                """
                SELECT resolution_status, COUNT(*) as count
                FROM semantic_conflicts
                GROUP BY resolution_status
            """,
                (),
            )
            conflict_stats = {row[0]: row[1] for row in cursor.fetchall()}

            # Priority distribution
            cursor = conn.execute(
                """
                SELECT 
                    COUNT(CASE WHEN consolidation_priority > 0.8 THEN 1 END) as high_priority,
                    COUNT(CASE WHEN consolidation_priority BETWEEN 0.5 AND 0.8 THEN 1 END) as medium_priority,
                    COUNT(CASE WHEN consolidation_priority < 0.5 THEN 1 END) as low_priority
                FROM memory_metadata
            """,
                (),
            )

            priority_dist = cursor.fetchone()

        return {
            "summary_period_hours": hours,
            "total_consolidation_events": stats[0] or 0,
            "event_types_active": stats[1] or 0,
            "events_by_type": event_types,
            "memory_store_distribution": store_distribution,
            # Phase 2 metrics
            "synaptic_tagged_memories": synaptic_tagged,
            "priority_replay_events": replay_events,
            "priority_distribution": {
                "high": priority_dist[0] or 0,
                "medium": priority_dist[1] or 0,
                "low": priority_dist[2] or 0,
            },
            # Phase 3 metrics
            "semantic_conflicts": conflict_stats,
            "total_conflicts": sum(conflict_stats.values()),
            "pending_conflicts": conflict_stats.get("pending", 0),
            "resolved_conflicts": conflict_stats.get("resolved", 0),
            # General metrics
            "last_consolidation": (
                self.last_consolidation.isoformat() if self.last_consolidation else None
            ),
            "last_rehearsal": self.last_rehearsal.isoformat() if self.last_rehearsal else None,
            "consolidation_interval_hours": self.consolidation_interval_hours,
            "generated_at": datetime.now().isoformat(),
        }

    def start_scheduled_consolidation(self, session_id: str):
        """Start scheduled consolidation for a session."""
        if not self.running:
            self.running = True
            self.consolidation_thread = threading.Thread(
                target=self._consolidation_loop, args=(session_id,), daemon=True
            )
            self.consolidation_thread.start()
            logger.info(f"Started scheduled consolidation for session {session_id}")

    def stop_consolidation(self):
        """Stop scheduled consolidation."""
        self.running = False
        if self.consolidation_thread and self.consolidation_thread.is_alive():
            self.consolidation_thread.join(timeout=5)
        logger.info("Stopped scheduled consolidation")

    def _consolidation_loop(self, session_id: str):
        """Main consolidation scheduling loop."""
        while self.running:
            try:
                # Run consolidation cycle
                results = self.run_consolidation_cycle(session_id)

                if results.get("memories_processed", 0) > 0:
                    logger.info(f"Consolidation cycle completed: {results}")

                # Wait for next consolidation window
                time.sleep(self.consolidation_interval_hours * 3600)

            except Exception as e:
                logger.error(f"Error in consolidation loop: {e}")
                time.sleep(3600)  # Wait 1 hour on error
