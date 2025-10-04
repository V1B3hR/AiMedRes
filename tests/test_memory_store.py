import os
import uuid
import pytest

from aimedres.agent_memory.embed_memory import AgentMemoryStore, MemoryType

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://duetmind:duetmind_secret@localhost:5432/duetmind")

@pytest.mark.integration
def test_store_and_retrieve_memories():
    store = AgentMemoryStore(db_connection_string=DATABASE_URL)
    store.ensure_connection()

    session_id = store.create_session("TestAgent", "1.0", {"env": "test"})
    try:
        m1 = store.store_memory(session_id, "APOE4 is associated with increased risk of AD", MemoryType.knowledge, 0.9)
        m2 = store.store_memory(session_id, "Patient MMSE score is 28", MemoryType.reasoning, 0.7)
        m3 = store.store_memory(session_id, "Low education years can lower model confidence", MemoryType.experience, 0.6)

        res = store.retrieve_memories(session_id, "APOE4 genetic risk", limit=2)
        assert len(res) == 2
        assert any("APOE4" in r.content for r in res)

        all_mems = store.get_session_memories(session_id)
        assert len(all_mems) >= 3

    finally:
        store.end_session(session_id)
