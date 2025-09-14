"""
Advanced Memory Consolidation for DuetMind Adaptive Agent Memory.

Implements dual-store architecture with episodic and semantic memory stores,
memory consolidation scheduling, salience scoring, and controlled forgetting.

Phase 1 Features:
- Dual-store architecture (EpisodicStore + SemanticStore)  
- Memory metadata (recency, frequency, utility, clinical relevance)
- Basic consolidation scheduler
- Memory salience scoring system
"""

import time
import threading
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import json
import numpy as np
import uuid

from .embed_memory import AgentMemoryStore

logger = logging.getLogger('duetmind.memory.consolidation')


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
    last_consolidated: Optional[datetime] = None
    consolidation_count: int = 0


@dataclass
class ConsolidationEvent:
    """Represents a memory consolidation event."""
    event_id: str
    timestamp: datetime
    source_store: MemoryStore
    target_store: MemoryStore
    memory_ids: List[int]
    consolidation_type: str  # 'summarize', 'merge', 'promote', 'prune'
    metadata: Dict[str, Any]


class MemoryConsolidator:
    """
    Advanced memory consolidation system for DuetMind Adaptive.
    
    Implements biological-inspired memory consolidation with:
    - Dual episodic/semantic stores
    - Salience-based consolidation prioritization
    - Scheduled consolidation windows
    - Controlled forgetting mechanisms
    """
    
    def __init__(self, memory_store: AgentMemoryStore, config: Optional[Dict[str, Any]] = None):
        self.memory_store = memory_store
        self.config = config or {}
        
        # Consolidation parameters
        self.consolidation_interval_hours = self.config.get('consolidation_interval_hours', 6)
        self.max_episodic_memories = self.config.get('max_episodic_memories', 1000)
        self.min_salience_threshold = self.config.get('min_salience_threshold', 0.3)
        self.novelty_weight = self.config.get('novelty_weight', 0.25)
        self.frequency_weight = self.config.get('frequency_weight', 0.25)
        self.clinical_weight = self.config.get('clinical_weight', 0.3)
        self.recency_weight = self.config.get('recency_weight', 0.2)
        
        # Consolidation database
        self.consolidation_db = self.config.get('consolidation_db', 'memory_consolidation.db')
        self._init_consolidation_db()
        
        # Consolidation state
        self.consolidation_thread = None
        self.running = False
        self.last_consolidation = None
        
        logger.info("Memory Consolidator initialized")
    
    def _init_consolidation_db(self):
        """Initialize consolidation tracking database."""
        with sqlite3.connect(self.consolidation_db) as conn:
            conn.execute("""
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
                    last_consolidated TEXT,
                    consolidation_count INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            
            conn.execute("""
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
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS semantic_clusters (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cluster_id TEXT UNIQUE NOT NULL,
                    cluster_centroid TEXT,
                    memory_ids TEXT NOT NULL,
                    cluster_summary TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            
            # Create indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_metadata_priority ON memory_metadata(consolidation_priority)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_metadata_store ON memory_metadata(memory_store)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_timestamp ON consolidation_events(timestamp)")
            
            conn.commit()
    
    def calculate_salience_score(self, memory_id: int, session_id: str) -> float:
        """
        Calculate salience score for a memory based on multiple factors.
        
        Args:
            memory_id: Memory ID to score
            session_id: Session ID for context
            
        Returns:
            Salience score (0-1)
        """
        try:
            # Get memory details from main store
            memories = self.memory_store.get_session_memories(session_id)
            target_memory = next((m for m in memories if m['id'] == memory_id), None)
            
            if not target_memory:
                return 0.0
            
            # Calculate component scores
            recency_score = self._calculate_recency_score(target_memory['created_at'])
            frequency_score = self._calculate_frequency_score(target_memory['access_count'])
            utility_score = target_memory.get('importance_score', 0.5)
            clinical_relevance = self._calculate_clinical_relevance(target_memory['content'])
            novelty_score = self._calculate_novelty_score(target_memory, memories)
            
            # Weighted combination
            salience = (
                self.recency_weight * recency_score +
                self.frequency_weight * frequency_score +
                self.clinical_weight * clinical_relevance +
                self.novelty_weight * novelty_score +
                0.1 * utility_score  # Base importance
            )
            
            # Store metadata
            self._store_memory_metadata(
                memory_id, 
                MemoryStore.EPISODIC,  # Assuming new memories start episodic
                MemoryMetadata(
                    recency_score=recency_score,
                    frequency_score=frequency_score,
                    utility_score=utility_score,
                    clinical_relevance=clinical_relevance,
                    novelty_score=novelty_score,
                    consolidation_priority=salience
                )
            )
            
            return min(1.0, max(0.0, salience))
            
        except Exception as e:
            logger.error(f"Error calculating salience score: {e}")
            return 0.5  # Default moderate salience
    
    def _calculate_recency_score(self, created_at: datetime) -> float:
        """Calculate recency score (more recent = higher score)."""
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
        
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
            'diagnosis', 'treatment', 'patient', 'symptom', 'medication', 
            'clinical', 'medical', 'health', 'disease', 'therapy',
            'MMSE', 'cognitive', 'Alzheimer', 'dementia', 'biomarker',
            'APOE', 'genetic', 'risk', 'assessment', 'guideline'
        ]
        
        content_lower = content.lower()
        matches = sum(1 for keyword in clinical_keywords if keyword.lower() in content_lower)
        
        # Scale based on keyword density
        clinical_score = min(1.0, matches / 5.0)  # 5+ keywords = max score
        return clinical_score
    
    def _calculate_novelty_score(self, target_memory: Dict[str, Any], all_memories: List[Dict[str, Any]]) -> float:
        """Calculate novelty score by comparing with existing memories."""
        if len(all_memories) <= 1:
            return 1.0  # First memory is always novel
        
        target_content = target_memory['content'].lower()
        target_words = set(target_content.split())
        
        similarities = []
        for memory in all_memories:
            if memory['id'] == target_memory['id']:
                continue  # Skip self
            
            memory_words = set(memory['content'].lower().split())
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
    
    def _store_memory_metadata(self, memory_id: int, store_type: MemoryStore, metadata: MemoryMetadata):
        """Store memory metadata in consolidation database."""
        with sqlite3.connect(self.consolidation_db) as conn:
            now = datetime.now().isoformat()
            
            conn.execute("""
                INSERT OR REPLACE INTO memory_metadata (
                    memory_id, memory_store, recency_score, frequency_score,
                    utility_score, clinical_relevance, novelty_score, emotional_valence,
                    consolidation_priority, last_consolidated, consolidation_count,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                memory_id, store_type.value, metadata.recency_score, metadata.frequency_score,
                metadata.utility_score, metadata.clinical_relevance, metadata.novelty_score,
                metadata.emotional_valence, metadata.consolidation_priority,
                metadata.last_consolidated.isoformat() if metadata.last_consolidated else None,
                metadata.consolidation_count, now, now
            ))
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
            'session_id': session_id,
            'start_time': start_time.isoformat(),
            'memories_processed': 0,
            'memories_consolidated': 0,
            'memories_pruned': 0,
            'semantic_clusters_created': 0,
            'events': []
        }
        
        try:
            # Get all memories for session
            memories = self.memory_store.get_session_memories(session_id)
            results['memories_processed'] = len(memories)
            
            if not memories:
                logger.info(f"No memories to consolidate for session {session_id}")
                return results
            
            # Calculate salience scores for all memories
            memory_salience = {}
            for memory in memories:
                salience = self.calculate_salience_score(memory['id'], session_id)
                memory_salience[memory['id']] = salience
            
            # Sort memories by salience (highest first)
            sorted_memories = sorted(memories, key=lambda m: memory_salience[m['id']], reverse=True)
            
            # Phase 1: Promote high-salience episodic memories to semantic store
            promoted_count = self._promote_to_semantic(sorted_memories, memory_salience, session_id)
            results['memories_consolidated'] += promoted_count
            
            # Phase 2: Create semantic clusters from related memories
            clusters_created = self._create_semantic_clusters(sorted_memories, session_id)
            results['semantic_clusters_created'] = clusters_created
            
            # Phase 3: Prune low-salience memories
            pruned_count = self._prune_low_salience_memories(sorted_memories, memory_salience, session_id)
            results['memories_pruned'] = pruned_count
            
            # Update consolidation timestamp
            self.last_consolidation = datetime.now()
            
            logger.info(f"Consolidation completed for session {session_id}: "
                       f"{promoted_count} promoted, {clusters_created} clusters, {pruned_count} pruned")
            
        except Exception as e:
            logger.error(f"Error in consolidation cycle: {e}")
            results['error'] = str(e)
        
        results['end_time'] = datetime.now().isoformat()
        results['duration_seconds'] = (datetime.now() - start_time).total_seconds()
        
        return results
    
    def _promote_to_semantic(self, memories: List[Dict[str, Any]], 
                           salience_scores: Dict[int, float], session_id: str) -> int:
        """Promote high-salience episodic memories to semantic store."""
        promoted_count = 0
        promotion_threshold = 0.7  # Only promote highly salient memories
        
        for memory in memories:
            salience = salience_scores[memory['id']]
            
            if salience >= promotion_threshold and memory['memory_type'] != 'semantic':
                try:
                    # Create consolidated semantic memory
                    semantic_content = self._create_semantic_summary(memory)
                    
                    # Store as new semantic memory
                    semantic_id = self.memory_store.store_memory(
                        session_id=session_id,
                        content=semantic_content,
                        memory_type='semantic',
                        importance=salience,
                        metadata={
                            'source_memory_id': memory['id'],
                            'consolidation_type': 'promotion',
                            'original_type': memory['memory_type']
                        }
                    )
                    
                    # Update metadata
                    metadata = MemoryMetadata(
                        consolidation_priority=salience,
                        last_consolidated=datetime.now(),
                        consolidation_count=1
                    )
                    self._store_memory_metadata(semantic_id, MemoryStore.SEMANTIC, metadata)
                    
                    # Log consolidation event
                    self._log_consolidation_event(
                        source_store=MemoryStore.EPISODIC,
                        target_store=MemoryStore.SEMANTIC,
                        memory_ids=[memory['id'], semantic_id],
                        consolidation_type='promotion',
                        metadata={'salience_score': salience}
                    )
                    
                    promoted_count += 1
                    
                except Exception as e:
                    logger.error(f"Error promoting memory {memory['id']}: {e}")
        
        return promoted_count
    
    def _create_semantic_summary(self, memory: Dict[str, Any]) -> str:
        """Create a semantic summary of an episodic memory."""
        content = memory['content']
        memory_type = memory['memory_type']
        
        # Basic semantic extraction (could be enhanced with NLP)
        if memory_type == 'reasoning':
            # Extract key reasoning patterns
            if 'MMSE' in content:
                return f"Cognitive assessment insight: {content[:150]}..."
            elif 'APOE' in content:
                return f"Genetic risk factor analysis: {content[:150]}..."
            else:
                return f"Clinical reasoning: {content[:150]}..."
        elif memory_type == 'experience':
            return f"Clinical experience: {content[:150]}..."
        else:
            return f"Knowledge: {content[:150]}..."
    
    def _create_semantic_clusters(self, memories: List[Dict[str, Any]], session_id: str) -> int:
        """Create semantic clusters from related memories."""
        clusters_created = 0
        
        # Group memories by content similarity (simplified clustering)
        semantic_memories = [m for m in memories if m['memory_type'] == 'semantic']
        
        if len(semantic_memories) < 3:  # Need at least 3 for clustering
            return 0
        
        # Simple keyword-based clustering
        clusters = self._simple_keyword_clustering(semantic_memories)
        
        for cluster_keywords, cluster_memories in clusters.items():
            if len(cluster_memories) >= 2:  # At least 2 memories per cluster
                try:
                    cluster_id = str(uuid.uuid4())
                    memory_ids = [m['id'] for m in cluster_memories]
                    
                    # Create cluster summary
                    cluster_summary = f"Semantic cluster: {cluster_keywords}. "
                    cluster_summary += f"Contains {len(cluster_memories)} related memories about "
                    cluster_summary += f"{', '.join(cluster_keywords.split('_'))}"
                    
                    # Store cluster
                    with sqlite3.connect(self.consolidation_db) as conn:
                        now = datetime.now().isoformat()
                        conn.execute("""
                            INSERT INTO semantic_clusters 
                            (cluster_id, memory_ids, cluster_summary, created_at, updated_at)
                            VALUES (?, ?, ?, ?, ?)
                        """, (cluster_id, json.dumps(memory_ids), cluster_summary, now, now))
                        conn.commit()
                    
                    clusters_created += 1
                    
                except Exception as e:
                    logger.error(f"Error creating semantic cluster: {e}")
        
        return clusters_created
    
    def _simple_keyword_clustering(self, memories: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Simple keyword-based clustering of memories."""
        clusters = {}
        
        clinical_topics = {
            'cognitive_assessment': ['MMSE', 'cognitive', 'assessment', 'testing'],
            'genetic_risk': ['APOE', 'genetic', 'gene', 'hereditary'],
            'biomarkers': ['biomarker', 'protein', 'tau', 'amyloid'],
            'treatment': ['treatment', 'therapy', 'intervention', 'medication'],
            'diagnosis': ['diagnosis', 'diagnostic', 'differential', 'criteria']
        }
        
        for memory in memories:
            content_lower = memory['content'].lower()
            
            for topic, keywords in clinical_topics.items():
                if any(keyword.lower() in content_lower for keyword in keywords):
                    if topic not in clusters:
                        clusters[topic] = []
                    clusters[topic].append(memory)
                    break  # Assign to first matching cluster only
        
        return clusters
    
    def _prune_low_salience_memories(self, memories: List[Dict[str, Any]], 
                                   salience_scores: Dict[int, float], session_id: str) -> int:
        """Prune memories with very low salience scores."""
        pruned_count = 0
        prune_threshold = self.min_salience_threshold
        
        # Only prune if we have too many memories
        if len(memories) <= self.max_episodic_memories:
            return 0
        
        # Sort by salience (lowest first for pruning)
        low_salience_memories = [
            m for m in memories 
            if salience_scores[m['id']] < prune_threshold
        ]
        
        # Prune oldest, lowest salience memories
        low_salience_memories.sort(key=lambda m: m['created_at'])
        
        memories_to_prune = low_salience_memories[:len(memories) - self.max_episodic_memories]
        
        for memory in memories_to_prune:
            try:
                # Mark as pruned (don't actually delete for safety)
                # In production, could implement soft delete or archival
                self._log_consolidation_event(
                    source_store=MemoryStore.EPISODIC,
                    target_store=MemoryStore.EPISODIC,  # Same store, just marking
                    memory_ids=[memory['id']],
                    consolidation_type='prune',
                    metadata={'salience_score': salience_scores[memory['id']]}
                )
                
                pruned_count += 1
                
            except Exception as e:
                logger.error(f"Error pruning memory {memory['id']}: {e}")
        
        return pruned_count
    
    def _log_consolidation_event(self, source_store: MemoryStore, target_store: MemoryStore,
                               memory_ids: List[int], consolidation_type: str,
                               metadata: Dict[str, Any]):
        """Log a consolidation event."""
        event = ConsolidationEvent(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            source_store=source_store,
            target_store=target_store,
            memory_ids=memory_ids,
            consolidation_type=consolidation_type,
            metadata=metadata
        )
        
        with sqlite3.connect(self.consolidation_db) as conn:
            conn.execute("""
                INSERT INTO consolidation_events 
                (event_id, timestamp, source_store, target_store, memory_ids, 
                 consolidation_type, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                event.event_id,
                event.timestamp.isoformat(),
                event.source_store.value,
                event.target_store.value,
                json.dumps(event.memory_ids),
                event.consolidation_type,
                json.dumps(event.metadata)
            ))
            conn.commit()
    
    def get_consolidation_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get consolidation activity summary."""
        cutoff_time = (datetime.now() - timedelta(hours=hours)).isoformat()
        
        with sqlite3.connect(self.consolidation_db) as conn:
            # Overall statistics
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_events,
                    COUNT(DISTINCT consolidation_type) as event_types
                FROM consolidation_events 
                WHERE timestamp > ?
            """, (cutoff_time,))
            
            stats = cursor.fetchone()
            
            # Events by type
            cursor = conn.execute("""
                SELECT consolidation_type, COUNT(*) as count
                FROM consolidation_events 
                WHERE timestamp > ?
                GROUP BY consolidation_type
            """, (cutoff_time,))
            
            event_types = {row[0]: row[1] for row in cursor.fetchall()}
            
            # Memory store distribution
            cursor = conn.execute("""
                SELECT memory_store, COUNT(*) as count
                FROM memory_metadata
                GROUP BY memory_store
            """, ())
            
            store_distribution = {row[0]: row[1] for row in cursor.fetchall()}
        
        return {
            'summary_period_hours': hours,
            'total_consolidation_events': stats[0] or 0,
            'event_types_active': stats[1] or 0,
            'events_by_type': event_types,
            'memory_store_distribution': store_distribution,
            'last_consolidation': self.last_consolidation.isoformat() if self.last_consolidation else None,
            'consolidation_interval_hours': self.consolidation_interval_hours,
            'generated_at': datetime.now().isoformat()
        }
    
    def start_scheduled_consolidation(self, session_id: str):
        """Start scheduled consolidation for a session."""
        if not self.running:
            self.running = True
            self.consolidation_thread = threading.Thread(
                target=self._consolidation_loop, 
                args=(session_id,), 
                daemon=True
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
                
                if results.get('memories_processed', 0) > 0:
                    logger.info(f"Consolidation cycle completed: {results}")
                
                # Wait for next consolidation window
                time.sleep(self.consolidation_interval_hours * 3600)
                
            except Exception as e:
                logger.error(f"Error in consolidation loop: {e}")
                time.sleep(3600)  # Wait 1 hour on error