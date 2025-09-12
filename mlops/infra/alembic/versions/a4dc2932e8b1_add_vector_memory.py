"""add_vector_memory

Revision ID: a4dc2932e8b1
Revises: bff30a888419
Create Date: 2025-09-12 12:33:41.289982

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = 'a4dc2932e8b1'
down_revision: Union[str, Sequence[str], None] = 'bff30a888419'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add pgvector extension and agent memory tables."""
    
    # Enable pgvector extension (will be created in init.sql if available)
    op.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    
    # Agent Memory Table for storing reasoning embeddings
    op.create_table('agent_memory',
        sa.Column('id', sa.Integer, primary_key=True, autoincrement=True),
        sa.Column('session_id', sa.String(100), nullable=False),  # Agent session identifier
        sa.Column('memory_type', sa.String(50), nullable=False),  # 'reasoning', 'experience', 'knowledge'
        sa.Column('content', sa.Text, nullable=False),  # Original text content
        sa.Column('embedding', sa.Text, nullable=False),  # Vector embedding as text (pgvector format)
        sa.Column('metadata_json', postgresql.JSONB),  # Additional metadata
        sa.Column('importance_score', sa.Float, default=0.0),  # Memory importance (0-1)
        sa.Column('access_count', sa.Integer, default=0),  # How many times accessed
        sa.Column('last_accessed', sa.DateTime),  # Last access timestamp
        sa.Column('created_at', sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column('expires_at', sa.DateTime),  # Optional expiration
    )
    
    # Memory Associations Table (for memory relationships)
    op.create_table('memory_associations',
        sa.Column('id', sa.Integer, primary_key=True, autoincrement=True),
        sa.Column('source_memory_id', sa.Integer, sa.ForeignKey('agent_memory.id'), nullable=False),
        sa.Column('target_memory_id', sa.Integer, sa.ForeignKey('agent_memory.id'), nullable=False),
        sa.Column('association_type', sa.String(50), nullable=False),  # 'similar', 'causal', 'temporal'
        sa.Column('strength', sa.Float, default=0.0),  # Association strength (0-1)
        sa.Column('created_at', sa.DateTime, nullable=False, server_default=sa.func.now()),
    )
    
    # Agent Sessions Table
    op.create_table('agent_sessions',
        sa.Column('id', sa.String(100), primary_key=True),  # UUID session ID
        sa.Column('agent_name', sa.String(100), nullable=False),
        sa.Column('agent_version', sa.String(50)),
        sa.Column('session_metadata', postgresql.JSONB),
        sa.Column('started_at', sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column('ended_at', sa.DateTime),
        sa.Column('status', sa.String(20), default='active'),  # 'active', 'completed', 'aborted'
    )
    
    # Create indexes for vector similarity search and performance
    op.create_index('ix_agent_memory_session', 'agent_memory', ['session_id'])
    op.create_index('ix_agent_memory_type', 'agent_memory', ['memory_type'])
    op.create_index('ix_agent_memory_importance', 'agent_memory', ['importance_score'])
    op.create_index('ix_agent_memory_created', 'agent_memory', ['created_at'])
    op.create_index('ix_memory_associations_source', 'memory_associations', ['source_memory_id'])
    op.create_index('ix_memory_associations_target', 'memory_associations', ['target_memory_id'])
    op.create_index('ix_agent_sessions_name', 'agent_sessions', ['agent_name'])
    op.create_index('ix_agent_sessions_status', 'agent_sessions', ['status'])
    
    # Create unique constraint to prevent duplicate associations
    op.create_unique_constraint(
        'uq_memory_association',
        'memory_associations',
        ['source_memory_id', 'target_memory_id', 'association_type']
    )


def downgrade() -> None:
    """Remove vector memory tables and extension."""
    op.drop_table('memory_associations')
    op.drop_table('agent_memory')
    op.drop_table('agent_sessions')
    
    # Note: We don't drop the vector extension as it might be used elsewhere
