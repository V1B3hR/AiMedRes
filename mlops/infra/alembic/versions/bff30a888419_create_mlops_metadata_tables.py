"""create_mlops_metadata_tables

Revision ID: bff30a888419
Revises: 
Create Date: 2025-09-12 12:31:15.062102

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = 'bff30a888419'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create MLOps metadata tables for DuetMind Adaptive."""
    
    # Dataset Catalog Table
    op.create_table('dataset_catalog',
        sa.Column('id', sa.Integer, primary_key=True, autoincrement=True),
        sa.Column('name', sa.String(255), nullable=False, unique=True),
        sa.Column('description', sa.Text),
        sa.Column('source_type', sa.String(50), nullable=False),  # 'raw', 'processed', 'external'
        sa.Column('file_path', sa.String(500), nullable=False),
        sa.Column('file_format', sa.String(20), nullable=False),  # 'csv', 'parquet', 'json'
        sa.Column('schema_hash', sa.String(32)),  # Feature schema hash
        sa.Column('size_bytes', sa.BigInteger),
        sa.Column('row_count', sa.Integer),
        sa.Column('column_count', sa.Integer),
        sa.Column('created_at', sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column('metadata_json', postgresql.JSONB),  # Additional metadata
    )
    
    # Feature View Table
    op.create_table('feature_view',
        sa.Column('id', sa.Integer, primary_key=True, autoincrement=True),
        sa.Column('name', sa.String(255), nullable=False, unique=True),
        sa.Column('description', sa.Text),
        sa.Column('feature_hash', sa.String(32), nullable=False),  # Feature schema hash
        sa.Column('source_dataset_id', sa.Integer, sa.ForeignKey('dataset_catalog.id')),
        sa.Column('feature_list', postgresql.JSONB),  # List of feature names and types
        sa.Column('transformation_logic', sa.Text),  # Description of transformations
        sa.Column('validation_schema', postgresql.JSONB),  # Pandera schema JSON
        sa.Column('created_at', sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime, nullable=False, server_default=sa.func.now()),
    )
    
    # Model Registry Table  
    op.create_table('model_registry',
        sa.Column('id', sa.Integer, primary_key=True, autoincrement=True),
        sa.Column('name', sa.String(255), nullable=False, unique=True),
        sa.Column('description', sa.Text),
        sa.Column('model_type', sa.String(100), nullable=False),  # 'classification', 'regression', etc.
        sa.Column('framework', sa.String(50), nullable=False),  # 'sklearn', 'pytorch', etc.
        sa.Column('created_at', sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime, nullable=False, server_default=sa.func.now()),
    )
    
    # Model Version Detail Table
    op.create_table('model_version_detail',
        sa.Column('id', sa.Integer, primary_key=True, autoincrement=True),
        sa.Column('model_id', sa.Integer, sa.ForeignKey('model_registry.id'), nullable=False),
        sa.Column('version', sa.String(50), nullable=False),
        sa.Column('mlflow_run_id', sa.String(100), nullable=False, unique=True),
        sa.Column('feature_hash', sa.String(32), nullable=False),  # Must match feature view
        sa.Column('feature_view_id', sa.Integer, sa.ForeignKey('feature_view.id'), nullable=False),
        sa.Column('model_path', sa.String(500)),  # Local model artifact path
        sa.Column('model_uri', sa.String(500)),   # MLflow model URI
        sa.Column('status', sa.String(20), nullable=False, default='staging'),  # 'staging', 'production', 'archived'
        sa.Column('performance_metrics', postgresql.JSONB),  # Model performance metrics
        sa.Column('hyperparameters', postgresql.JSONB),      # Model hyperparameters
        sa.Column('training_dataset_id', sa.Integer, sa.ForeignKey('dataset_catalog.id')),
        sa.Column('validation_dataset_id', sa.Integer, sa.ForeignKey('dataset_catalog.id')),
        sa.Column('created_at', sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column('promoted_at', sa.DateTime),  # When promoted to production
        sa.Column('deprecated_at', sa.DateTime), # When deprecated
    )
    
    # Experiment Tracking Table (supplements MLflow)
    op.create_table('experiment_tracking',
        sa.Column('id', sa.Integer, primary_key=True, autoincrement=True),
        sa.Column('mlflow_experiment_id', sa.String(100), nullable=False),
        sa.Column('mlflow_run_id', sa.String(100), nullable=False, unique=True),
        sa.Column('experiment_name', sa.String(255), nullable=False),
        sa.Column('run_name', sa.String(255)),
        sa.Column('git_commit_hash', sa.String(40)),  # Git commit for reproducibility
        sa.Column('environment_info', postgresql.JSONB),  # Python version, packages, etc.
        sa.Column('created_at', sa.DateTime, nullable=False, server_default=sa.func.now()),
    )
    
    # Data Quality Metrics Table
    op.create_table('data_quality_metrics',
        sa.Column('id', sa.Integer, primary_key=True, autoincrement=True),
        sa.Column('dataset_id', sa.Integer, sa.ForeignKey('dataset_catalog.id'), nullable=False),
        sa.Column('check_timestamp', sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column('missing_values_count', sa.Integer),
        sa.Column('duplicate_rows_count', sa.Integer),
        sa.Column('schema_violations', postgresql.JSONB),
        sa.Column('quality_score', sa.Float),  # Overall quality score 0-1
        sa.Column('quality_report', postgresql.JSONB),  # Detailed quality report
        sa.Column('passed_validation', sa.Boolean, nullable=False),
    )
    
    # Create indexes for performance
    op.create_index('ix_dataset_catalog_name', 'dataset_catalog', ['name'])
    op.create_index('ix_dataset_catalog_schema_hash', 'dataset_catalog', ['schema_hash'])
    op.create_index('ix_feature_view_hash', 'feature_view', ['feature_hash'])
    op.create_index('ix_model_version_mlflow_run', 'model_version_detail', ['mlflow_run_id'])
    op.create_index('ix_model_version_status', 'model_version_detail', ['status'])
    op.create_index('ix_experiment_tracking_run', 'experiment_tracking', ['mlflow_run_id'])
    
    # Add unique constraint for model name + version
    op.create_unique_constraint(
        'uq_model_name_version', 
        'model_version_detail', 
        ['model_id', 'version']
    )


def downgrade() -> None:
    """Drop all MLOps metadata tables."""
    op.drop_table('data_quality_metrics')
    op.drop_table('experiment_tracking')
    op.drop_table('model_version_detail')
    op.drop_table('model_registry')
    op.drop_table('feature_view')
    op.drop_table('dataset_catalog')
