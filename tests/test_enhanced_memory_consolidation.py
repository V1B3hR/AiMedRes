"""
Tests for Enhanced Memory Consolidation System

Tests the Phase 2 and Phase 3 enhancements including:
- Priority replay with synaptic tagging
- Semantic conflict detection and resolution
- Memory introspection API
- Enhanced salience scoring with uncertainty and reward
"""

import pytest
import tempfile
import os
import sqlite3
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

# Import our memory consolidation system
from aimedres.agent_memory.memory_consolidation import (
    MemoryConsolidator, MemoryStore, MemoryMetadata, 
    SemanticConflict, ConsolidationEvent
)
from aimedres.agent_memory.embed_memory import AgentMemoryStore


class MockMemoryStore:
    """Mock memory store for testing."""
    
    def __init__(self):
        self.memories = {}
        self.session_memories = {}
        self.next_id = 1
    
    def get_session_memories(self, session_id: str):
        return self.session_memories.get(session_id, [])
    
    def store_memory(self, session_id: str, content: str, memory_type: str = "reasoning",
                    importance: float = 0.5, metadata: dict = None):
        memory = {
            'id': self.next_id,
            'session_id': session_id,
            'content': content,
            'memory_type': memory_type,
            'importance_score': importance,
            'metadata': metadata or {},
            'created_at': datetime.now(),
            'access_count': 0
        }
        
        self.memories[self.next_id] = memory
        
        if session_id not in self.session_memories:
            self.session_memories[session_id] = []
        self.session_memories[session_id].append(memory)
        
        memory_id = self.next_id
        self.next_id += 1
        return memory_id
    
    def retrieve_memories(self, session_id: str, query: str, limit: int = 5,
                         memory_type: str = None, min_importance: float = 0.0):
        memories = self.get_session_memories(session_id)
        
        # Simple keyword matching for testing
        filtered = []
        for memory in memories:
            if query.lower() in memory['content'].lower():
                if not memory_type or memory['memory_type'] == memory_type:
                    if memory['importance_score'] >= min_importance:
                        filtered.append(memory)
        
        return filtered[:limit]


@pytest.fixture
def temp_db():
    """Create temporary database for testing."""
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    yield path
    os.unlink(path)


@pytest.fixture
def mock_memory_store():
    """Create mock memory store."""
    return MockMemoryStore()


@pytest.fixture
def consolidator(temp_db, mock_memory_store):
    """Create memory consolidator for testing."""
    config = {
        'consolidation_db': temp_db,
        'consolidation_interval_hours': 1,
        'max_episodic_memories': 100,
        'min_salience_threshold': 0.3,
        'replay_sample_size': 10,
        'synaptic_tag_threshold': 0.8,
        'uncertainty_weight': 0.3,
        'reward_weight': 0.4
    }
    
    return MemoryConsolidator(mock_memory_store, config)


class TestEnhancedSalienceScoring:
    """Test enhanced salience scoring with uncertainty and reward."""
    
    def test_basic_salience_calculation(self, consolidator, mock_memory_store):
        """Test basic salience score calculation."""
        session_id = "test_session"
        
        # Add a memory
        memory_id = mock_memory_store.store_memory(
            session_id=session_id,
            content="Patient shows high MMSE score (28) but has APOE4 genotype",
            memory_type="reasoning",
            importance=0.7
        )
        
        # Update access count for frequency scoring
        mock_memory_store.memories[memory_id]['access_count'] = 5
        
        # Calculate salience with uncertainty and reward
        salience = consolidator.calculate_salience_score(
            memory_id, session_id, uncertainty_score=0.6, reward_signal=0.8
        )
        
        assert 0.0 <= salience <= 1.0
        assert salience > 0.5  # Should be high due to clinical content and reward
    
    def test_synaptic_tagging(self, consolidator, mock_memory_store):
        """Test synaptic tagging for high-reward memories."""
        session_id = "test_session"
        
        memory_id = mock_memory_store.store_memory(
            session_id=session_id,
            content="Critical clinical insight with high confidence",
            memory_type="reasoning",
            importance=0.9
        )
        
        # High reward should trigger synaptic tagging
        salience = consolidator.calculate_salience_score(
            memory_id, session_id, uncertainty_score=0.7, reward_signal=0.9
        )
        
        # Check if memory received synaptic tag
        with sqlite3.connect(consolidator.consolidation_db) as conn:
            cursor = conn.execute("""
                SELECT synaptic_tag FROM memory_metadata WHERE memory_id = ?
            """, (memory_id,))
            row = cursor.fetchone()
            
        assert row is not None
        assert row[0] == 1  # Synaptic tag should be set
    
    def test_low_reward_no_tagging(self, consolidator, mock_memory_store):
        """Test that low reward doesn't trigger synaptic tagging."""
        session_id = "test_session"
        
        memory_id = mock_memory_store.store_memory(
            session_id=session_id,
            content="Low confidence observation",
            memory_type="reasoning",
            importance=0.4
        )
        
        # Low reward should not trigger synaptic tagging
        salience = consolidator.calculate_salience_score(
            memory_id, session_id, uncertainty_score=0.3, reward_signal=0.2
        )
        
        with sqlite3.connect(consolidator.consolidation_db) as conn:
            cursor = conn.execute("""
                SELECT synaptic_tag FROM memory_metadata WHERE memory_id = ?
            """, (memory_id,))
            row = cursor.fetchone()
            
        assert row is not None
        assert row[0] == 0  # No synaptic tag


class TestPriorityReplay:
    """Test priority replay functionality."""
    
    def test_priority_replay_execution(self, consolidator, mock_memory_store):
        """Test priority replay with weighted sampling."""
        session_id = "test_session"
        
        # Add memories with different characteristics
        memory_ids = []
        for i in range(5):
            memory_id = mock_memory_store.store_memory(
                session_id=session_id,
                content=f"Clinical memory {i} with varying importance",
                memory_type="reasoning",
                importance=0.5 + (i * 0.1)
            )
            memory_ids.append(memory_id)
            
            # Calculate salience with different uncertainty/reward values
            consolidator.calculate_salience_score(
                memory_id, session_id, 
                uncertainty_score=0.3 + (i * 0.1),
                reward_signal=0.2 + (i * 0.15)
            )
        
        memories = mock_memory_store.get_session_memories(session_id)
        memory_salience = {m['id']: 0.5 + (m['id'] * 0.1) for m in memories}
        
        # Run priority replay
        replay_count = consolidator._run_priority_replay(memories, memory_salience, session_id)
        
        assert replay_count > 0
        
        # Check that replay event was recorded
        with sqlite3.connect(consolidator.consolidation_db) as conn:
            cursor = conn.execute("""
                SELECT COUNT(*) FROM replay_events WHERE session_id = ?
            """, (session_id,))
            count = cursor.fetchone()[0]
            
        assert count > 0
    
    def test_empty_memories_no_replay(self, consolidator, mock_memory_store):
        """Test that empty memory set doesn't trigger replay."""
        session_id = "empty_session"
        
        memories = mock_memory_store.get_session_memories(session_id)
        memory_salience = {}
        
        replay_count = consolidator._run_priority_replay(memories, memory_salience, session_id)
        
        assert replay_count == 0


class TestGenerativeRehearsal:
    """Test generative rehearsal functionality."""
    
    def test_rehearsal_summary_creation(self, consolidator, mock_memory_store):
        """Test creation of rehearsal summaries."""
        session_id = "test_session"
        
        # Add memories with clinical content
        memory_contents = [
            "MMSE score indicates moderate cognitive decline",
            "APOE4 genotype presents increased Alzheimer's risk",
            "Patient shows difficulty with executive function tasks"
        ]
        
        for content in memory_contents:
            memory_id = mock_memory_store.store_memory(
                session_id=session_id,
                content=content,
                memory_type="reasoning",
                importance=0.8
            )
            
            # Mark as high salience
            consolidator.calculate_salience_score(
                memory_id, session_id, uncertainty_score=0.5, reward_signal=0.7
            )
        
        # Force rehearsal timing
        consolidator.last_rehearsal = None
        
        rehearsal_count = consolidator._run_generative_rehearsal(session_id)
        
        assert rehearsal_count > 0
        
        # Check that rehearsal summary was created
        memories = mock_memory_store.get_session_memories(session_id)
        rehearsal_memories = [m for m in memories if 'rehearsal' in m.get('metadata', {})]
        
        assert len(rehearsal_memories) > 0
    
    def test_rehearsal_timing_control(self, consolidator, mock_memory_store):
        """Test that rehearsal respects timing intervals."""
        # Set recent rehearsal time
        consolidator.last_rehearsal = datetime.now() - timedelta(hours=1)
        consolidator.rehearsal_interval_hours = 24
        
        # Should not run rehearsal
        should_run = consolidator._should_run_rehearsal()
        assert not should_run
        
        # Set old rehearsal time
        consolidator.last_rehearsal = datetime.now() - timedelta(hours=25)
        
        # Should run rehearsal
        should_run = consolidator._should_run_rehearsal()
        assert should_run


class TestSemanticConflictDetection:
    """Test semantic conflict detection and resolution."""
    
    def test_contradiction_detection(self, consolidator, mock_memory_store):
        """Test detection of contradictory statements."""
        session_id = "test_session"
        
        # Add contradictory semantic memories
        memory1_id = mock_memory_store.store_memory(
            session_id=session_id,
            content="Treatment X shows positive effects on MMSE scores",
            memory_type="semantic",
            importance=0.8
        )
        
        memory2_id = mock_memory_store.store_memory(
            session_id=session_id,
            content="Treatment X does not show significant effects on MMSE scores",
            memory_type="semantic",
            importance=0.7
        )
        
        memories = mock_memory_store.get_session_memories(session_id)
        
        # Run conflict detection
        conflicts_detected = consolidator._detect_semantic_conflicts(memories, session_id)
        
        assert conflicts_detected > 0
        
        # Check that conflict was recorded in database
        with sqlite3.connect(consolidator.consolidation_db) as conn:
            cursor = conn.execute("""
                SELECT COUNT(*) FROM semantic_conflicts 
                WHERE memory_id1 = ? OR memory_id2 = ?
            """, (memory1_id, memory2_id))
            count = cursor.fetchone()[0]
            
        assert count > 0
    
    def test_conflict_resolution(self, consolidator, mock_memory_store):
        """Test manual conflict resolution."""
        session_id = "test_session"
        
        # Create a conflict record manually
        conflict_id = "test_conflict_123"
        memory1_id = mock_memory_store.store_memory(
            session_id=session_id,
            content="Conflicting memory 1",
            memory_type="semantic"
        )
        memory2_id = mock_memory_store.store_memory(
            session_id=session_id,
            content="Conflicting memory 2", 
            memory_type="semantic"
        )
        
        # Insert conflict into database
        with sqlite3.connect(consolidator.consolidation_db) as conn:
            conn.execute("""
                INSERT INTO semantic_conflicts 
                (conflict_id, memory_id1, memory_id2, conflict_type, 
                 confidence_score, created_at, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                conflict_id, memory1_id, memory2_id, 'contradiction',
                0.8, datetime.now().isoformat(), '{}'
            ))
            conn.commit()
        
        # Resolve the conflict
        success = consolidator.resolve_semantic_conflict(
            conflict_id, 'manual', winning_memory_id=memory1_id
        )
        
        assert success
        
        # Check that conflict is marked as resolved
        with sqlite3.connect(consolidator.consolidation_db) as conn:
            cursor = conn.execute("""
                SELECT resolution_status FROM semantic_conflicts 
                WHERE conflict_id = ?
            """, (conflict_id,))
            row = cursor.fetchone()
            
        assert row is not None
        assert row[0] == 'resolved'
    
    def test_no_conflicts_with_different_topics(self, consolidator, mock_memory_store):
        """Test that memories about different topics don't conflict."""
        session_id = "test_session"
        
        # Add memories about different topics
        mock_memory_store.store_memory(
            session_id=session_id,
            content="MMSE scores are useful for cognitive assessment",
            memory_type="semantic"
        )
        
        mock_memory_store.store_memory(
            session_id=session_id,
            content="Blood pressure medication should be taken daily",
            memory_type="semantic"
        )
        
        memories = mock_memory_store.get_session_memories(session_id)
        
        # Should not detect conflicts
        conflicts_detected = consolidator._detect_semantic_conflicts(memories, session_id)
        
        assert conflicts_detected == 0


class TestMemoryIntrospection:
    """Test memory introspection API."""
    
    def test_decision_trace_generation(self, consolidator, mock_memory_store):
        """Test generation of decision traces."""
        session_id = "test_session"
        
        # Add relevant memories
        memory_contents = [
            "APOE4 increases Alzheimer's risk by 3-fold",
            "MMSE score below 24 indicates cognitive impairment", 
            "Family history is a strong predictor of dementia"
        ]
        
        for i, content in enumerate(memory_contents):
            memory_id = mock_memory_store.store_memory(
                session_id=session_id,
                content=content,
                memory_type="reasoning",
                importance=0.7 + (i * 0.1)
            )
            
            # Add metadata for tracing
            consolidator.calculate_salience_score(
                memory_id, session_id, 
                uncertainty_score=0.4, 
                reward_signal=0.6
            )
        
        # Get introspection for APOE4-related decision
        decision_context = "Should we test for APOE4 genotype?"
        introspection = consolidator.get_memory_introspection(decision_context, session_id)
        
        assert 'memory_chain' in introspection
        assert 'trace_confidence' in introspection
        assert len(introspection['memory_chain']) > 0
        assert introspection['trace_confidence'] > 0.0
        
        # Check that relevant memory is in the chain
        memory_chain = introspection['memory_chain']
        apoe_memory = next((m for m in memory_chain if 'APOE4' in m['content_preview']), None)
        assert apoe_memory is not None
    
    def test_introspection_with_conflicts(self, consolidator, mock_memory_store):
        """Test introspection when conflicted memories are involved."""
        session_id = "test_session"
        
        # Add a conflicted memory
        memory_id = mock_memory_store.store_memory(
            session_id=session_id,
            content="APOE4 testing results may be inconclusive",
            memory_type="reasoning",
            importance=0.6
        )
        
        consolidator.calculate_salience_score(memory_id, session_id)
        
        # Mark memory as conflicted in metadata
        with sqlite3.connect(consolidator.consolidation_db) as conn:
            conn.execute("""
                UPDATE memory_metadata 
                SET conflict_flag = 1 
                WHERE memory_id = ?
            """, (memory_id,))
            conn.commit()
        
        decision_context = "APOE4 testing recommendations"
        introspection = consolidator.get_memory_introspection(decision_context, session_id)
        
        assert introspection['conflicted_memories'] > 0
        
        # Check that conflicted memory has reduced influence
        memory_chain = introspection['memory_chain']
        conflicted_memory = next((m for m in memory_chain if m['has_conflicts']), None)
        assert conflicted_memory is not None


class TestConsolidationSummary:
    """Test enhanced consolidation summary reporting."""
    
    def test_enhanced_summary_metrics(self, consolidator, mock_memory_store):
        """Test that summary includes Phase 2 & 3 metrics."""
        session_id = "test_session"
        
        # Add some memories and run consolidation
        for i in range(3):
            memory_id = mock_memory_store.store_memory(
                session_id=session_id,
                content=f"Test memory {i}",
                memory_type="reasoning",
                importance=0.6
            )
            
            consolidator.calculate_salience_score(
                memory_id, session_id, 
                uncertainty_score=0.5, 
                reward_signal=0.8  # High reward for synaptic tagging
            )
        
        # Run a consolidation cycle to generate data
        consolidator.run_consolidation_cycle(session_id)
        
        # Get enhanced summary
        summary = consolidator.get_consolidation_summary(hours=24)
        
        # Check for Phase 2 & 3 metrics
        assert 'synaptic_tagged_memories' in summary
        assert 'priority_replay_events' in summary
        assert 'semantic_conflicts' in summary
        assert 'priority_distribution' in summary
        
        # Should have some synaptic tagged memories
        assert summary['synaptic_tagged_memories'] > 0


@pytest.mark.integration
class TestFullConsolidationCycle:
    """Integration tests for complete consolidation cycles."""
    
    def test_complete_enhanced_cycle(self, consolidator, mock_memory_store):
        """Test a complete consolidation cycle with all enhancements."""
        session_id = "integration_test"
        
        # Create diverse memory set
        memory_types = ['reasoning', 'experience', 'semantic']
        contents = [
            "High MMSE score with APOE4 presents complex risk profile",
            "Patient experience shows variable cognitive decline patterns", 
            "Clinical guidelines recommend multi-modal assessment",
            "Treatment A shows positive effects on cognitive function",
            "Treatment A does not show significant cognitive benefits"  # Conflicting
        ]
        
        for i, (content, mem_type) in enumerate(zip(contents, memory_types * 2)):
            memory_id = mock_memory_store.store_memory(
                session_id=session_id,
                content=content,
                memory_type=mem_type,
                importance=0.5 + (i * 0.1)
            )
            
            # Add varying uncertainty and reward signals
            consolidator.calculate_salience_score(
                memory_id, session_id,
                uncertainty_score=0.3 + (i * 0.1),
                reward_signal=0.2 + (i * 0.15)
            )
        
        # Run complete consolidation cycle
        results = consolidator.run_consolidation_cycle(session_id)
        
        # Verify all processes ran
        assert results['memories_processed'] > 0
        assert 'replay_events' in results
        assert 'conflicts_detected' in results
        assert results['semantic_clusters_created'] >= 0
        
        # Get final summary
        summary = consolidator.get_consolidation_summary(hours=1)
        
        # Should have activity in all areas
        assert summary['total_consolidation_events'] > 0
        assert summary['synaptic_tagged_memories'] >= 0
        assert summary['semantic_conflicts']['pending'] >= 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])