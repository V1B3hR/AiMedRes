"""create_imaging_tables

Revision ID: b785ad6607eb
Revises: a4dc2932e8b1
Create Date: 2025-09-16 06:41:45.433064

Creates medical imaging database tables for subjects, studies, series, QC, and features.
Supports DICOM metadata, BIDS compliance tracking, and de-identification workflows.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = 'b785ad6607eb'
down_revision: Union[str, Sequence[str], None] = 'a4dc2932e8b1'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create imaging tables for medical imaging data management."""
    
    # Imaging Subjects Table - Patient-level imaging metadata
    op.create_table('imaging_subjects',
        sa.Column('id', sa.Integer, primary_key=True, autoincrement=True),
        sa.Column('subject_id', sa.String(100), nullable=False, unique=True),  # De-identified subject ID
        sa.Column('original_patient_id', sa.String(100)),  # Original ID (encrypted/hashed)
        sa.Column('age_at_scan', sa.Integer),  # Age in years at time of scan
        sa.Column('sex', sa.String(10)),  # M/F/O/Unknown
        sa.Column('diagnosis_primary', sa.String(255)),  # Primary diagnosis (ICD-10)
        sa.Column('diagnosis_secondary', postgresql.JSONB),  # Secondary diagnoses array
        sa.Column('demographics', postgresql.JSONB),  # Additional demographic data
        sa.Column('clinical_notes', sa.Text),  # De-identified clinical notes
        sa.Column('consent_imaging', sa.Boolean, default=False),  # Imaging consent flag
        sa.Column('consent_research', sa.Boolean, default=False),  # Research consent flag
        sa.Column('de_identification_date', sa.DateTime),  # When de-identification occurred
        sa.Column('created_at', sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime, server_default=sa.func.now()),
    )
    
    # Imaging Studies Table - Study-level DICOM metadata
    op.create_table('imaging_studies',
        sa.Column('id', sa.Integer, primary_key=True, autoincrement=True),
        sa.Column('study_instance_uid', sa.String(200), nullable=False, unique=True),
        sa.Column('subject_id', sa.String(100), sa.ForeignKey('imaging_subjects.subject_id'), nullable=False),
        sa.Column('study_id', sa.String(100)),  # Study identifier
        sa.Column('study_date', sa.Date),  # DICOM StudyDate
        sa.Column('study_time', sa.Time),  # DICOM StudyTime
        sa.Column('accession_number', sa.String(100)),  # DICOM AccessionNumber
        sa.Column('study_description', sa.String(500)),  # DICOM StudyDescription
        sa.Column('referring_physician', sa.String(255)),  # De-identified referring physician
        sa.Column('institution_name', sa.String(255)),  # DICOM InstitutionName
        sa.Column('manufacturer', sa.String(100)),  # DICOM Manufacturer
        sa.Column('model_name', sa.String(100)),  # DICOM ManufacturerModelName
        sa.Column('magnetic_field_strength', sa.Float),  # MRI field strength
        sa.Column('protocol_name', sa.String(255)),  # Scan protocol name
        sa.Column('dicom_metadata', postgresql.JSONB),  # Full DICOM header (de-identified)
        sa.Column('bids_compliant', sa.Boolean, default=False),  # BIDS validation status
        sa.Column('bids_validation_report', postgresql.JSONB),  # BIDS validation details
        sa.Column('phi_removed', sa.Boolean, default=False),  # PHI removal flag
        sa.Column('storage_location', sa.String(500)),  # File storage path/URI
        sa.Column('created_at', sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime, server_default=sa.func.now()),
    )
    
    # Imaging Series Table - Series-level DICOM metadata
    op.create_table('imaging_series',
        sa.Column('id', sa.Integer, primary_key=True, autoincrement=True),
        sa.Column('series_instance_uid', sa.String(200), nullable=False, unique=True),
        sa.Column('study_instance_uid', sa.String(200), sa.ForeignKey('imaging_studies.study_instance_uid'), nullable=False),
        sa.Column('series_number', sa.Integer),  # DICOM SeriesNumber
        sa.Column('series_date', sa.Date),  # DICOM SeriesDate
        sa.Column('series_time', sa.Time),  # DICOM SeriesTime
        sa.Column('series_description', sa.String(500)),  # DICOM SeriesDescription
        sa.Column('modality', sa.String(20), nullable=False),  # DICOM Modality (MR, CT, etc.)
        sa.Column('body_part', sa.String(100)),  # DICOM BodyPartExamined
        sa.Column('sequence_name', sa.String(255)),  # MRI sequence name
        sa.Column('slice_thickness', sa.Float),  # DICOM SliceThickness
        sa.Column('pixel_spacing', postgresql.ARRAY(sa.Float)),  # DICOM PixelSpacing [x, y]
        sa.Column('repetition_time', sa.Float),  # MRI TR (ms)
        sa.Column('echo_time', sa.Float),  # MRI TE (ms)
        sa.Column('flip_angle', sa.Float),  # MRI FlipAngle
        sa.Column('matrix_size', postgresql.ARRAY(sa.Integer)),  # Image matrix [rows, cols]
        sa.Column('image_count', sa.Integer),  # Number of images in series
        sa.Column('dicom_metadata', postgresql.JSONB),  # Full DICOM header (de-identified)
        sa.Column('nifti_path', sa.String(500)),  # Path to converted NIfTI file
        sa.Column('nifti_conversion_date', sa.DateTime),  # When DICOM→NIfTI conversion occurred
        sa.Column('bids_sidecar', postgresql.JSONB),  # BIDS JSON sidecar data
        sa.Column('storage_location', sa.String(500)),  # DICOM file storage path/URI
        sa.Column('created_at', sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime, server_default=sa.func.now()),
    )
    
    # Imaging QC Table - Quality control metrics and assessments
    op.create_table('imaging_qc',
        sa.Column('id', sa.Integer, primary_key=True, autoincrement=True),
        sa.Column('series_instance_uid', sa.String(200), sa.ForeignKey('imaging_series.series_instance_uid'), nullable=False),
        sa.Column('qc_date', sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column('qc_version', sa.String(50)),  # QC pipeline version
        sa.Column('qc_status', sa.String(20), nullable=False),  # 'pass', 'fail', 'warning', 'pending'
        sa.Column('overall_score', sa.Float),  # Overall QC score (0-1)
        sa.Column('snr_score', sa.Float),  # Signal-to-noise ratio
        sa.Column('contrast_score', sa.Float),  # Contrast quality
        sa.Column('motion_score', sa.Float),  # Motion artifact assessment
        sa.Column('artifact_score', sa.Float),  # General artifact assessment
        sa.Column('coverage_score', sa.Float),  # Anatomical coverage quality
        sa.Column('resolution_score', sa.Float),  # Spatial resolution assessment
        sa.Column('intensity_uniformity', sa.Float),  # Intensity uniformity metric
        sa.Column('geometric_distortion', sa.Float),  # Geometric distortion measure
        sa.Column('automated_flags', postgresql.JSONB),  # Automated QC flags
        sa.Column('manual_review_flags', postgresql.JSONB),  # Manual review flags
        sa.Column('reviewer_id', sa.String(100)),  # QC reviewer identifier
        sa.Column('review_notes', sa.Text),  # Manual review notes
        sa.Column('usable_for_analysis', sa.Boolean, default=True),  # Analysis inclusion flag
        sa.Column('qc_metrics', postgresql.JSONB),  # Detailed QC metrics
        sa.Column('created_at', sa.DateTime, nullable=False, server_default=sa.func.now()),
    )
    
    # Imaging Features Table - Extracted imaging features and measurements
    op.create_table('imaging_features',
        sa.Column('id', sa.Integer, primary_key=True, autoincrement=True),
        sa.Column('series_instance_uid', sa.String(200), sa.ForeignKey('imaging_series.series_instance_uid'), nullable=False),
        sa.Column('feature_extraction_date', sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column('extraction_pipeline', sa.String(100)),  # Feature extraction pipeline name
        sa.Column('pipeline_version', sa.String(50)),  # Pipeline version
        sa.Column('feature_set_name', sa.String(100)),  # Feature set identifier
        # Volumetric measurements
        sa.Column('total_brain_volume', sa.Float),  # Total brain volume (mm³)
        sa.Column('gray_matter_volume', sa.Float),  # Gray matter volume
        sa.Column('white_matter_volume', sa.Float),  # White matter volume
        sa.Column('csf_volume', sa.Float),  # CSF volume
        sa.Column('intracranial_volume', sa.Float),  # Total intracranial volume
        sa.Column('hippocampal_volume', sa.Float),  # Hippocampal volume
        sa.Column('ventricular_volume', sa.Float),  # Ventricular volume
        # Regional measurements
        sa.Column('cortical_thickness', postgresql.JSONB),  # Regional cortical thickness
        sa.Column('subcortical_volumes', postgresql.JSONB),  # Subcortical structure volumes
        sa.Column('white_matter_lesions', postgresql.JSONB),  # WM lesion measurements
        # Image intensity features
        sa.Column('intensity_statistics', postgresql.JSONB),  # Mean, std, percentiles
        sa.Column('texture_features', postgresql.JSONB),  # GLCM, GLRLM features
        sa.Column('shape_features', postgresql.JSONB),  # Shape and morphology features
        # Advanced features
        sa.Column('radiomics_features', postgresql.JSONB),  # PyRadiomics feature extraction
        sa.Column('deep_learning_features', postgresql.JSONB),  # DL-extracted features
        sa.Column('custom_measurements', postgresql.JSONB),  # Study-specific measurements
        # Normalization and standardization
        sa.Column('normalized_features', postgresql.JSONB),  # Z-score normalized features
        sa.Column('atlas_registered', sa.Boolean, default=False),  # Atlas registration flag
        sa.Column('atlas_name', sa.String(100)),  # Reference atlas used
        sa.Column('feature_hash', sa.String(32)),  # Feature schema hash for consistency
        sa.Column('processing_notes', sa.Text),  # Processing notes and comments
        sa.Column('created_at', sa.DateTime, nullable=False, server_default=sa.func.now()),
    )
    
    # Create indexes for performance optimization
    op.create_index('ix_imaging_subjects_subject_id', 'imaging_subjects', ['subject_id'])
    op.create_index('ix_imaging_subjects_demographics', 'imaging_subjects', ['demographics'], postgresql_using='gin')
    
    op.create_index('ix_imaging_studies_subject_id', 'imaging_studies', ['subject_id'])
    op.create_index('ix_imaging_studies_study_date', 'imaging_studies', ['study_date'])
    op.create_index('ix_imaging_studies_modality', 'imaging_studies', ['study_description'])
    op.create_index('ix_imaging_studies_bids', 'imaging_studies', ['bids_compliant'])
    op.create_index('ix_imaging_studies_metadata', 'imaging_studies', ['dicom_metadata'], postgresql_using='gin')
    
    op.create_index('ix_imaging_series_study_uid', 'imaging_series', ['study_instance_uid'])
    op.create_index('ix_imaging_series_modality', 'imaging_series', ['modality'])
    op.create_index('ix_imaging_series_sequence', 'imaging_series', ['sequence_name'])
    op.create_index('ix_imaging_series_nifti', 'imaging_series', ['nifti_path'])
    op.create_index('ix_imaging_series_metadata', 'imaging_series', ['dicom_metadata'], postgresql_using='gin')
    
    op.create_index('ix_imaging_qc_series_uid', 'imaging_qc', ['series_instance_uid'])
    op.create_index('ix_imaging_qc_status', 'imaging_qc', ['qc_status'])
    op.create_index('ix_imaging_qc_score', 'imaging_qc', ['overall_score'])
    op.create_index('ix_imaging_qc_usable', 'imaging_qc', ['usable_for_analysis'])
    op.create_index('ix_imaging_qc_date', 'imaging_qc', ['qc_date'])
    
    op.create_index('ix_imaging_features_series_uid', 'imaging_features', ['series_instance_uid'])
    op.create_index('ix_imaging_features_pipeline', 'imaging_features', ['extraction_pipeline'])
    op.create_index('ix_imaging_features_hash', 'imaging_features', ['feature_hash'])
    op.create_index('ix_imaging_features_date', 'imaging_features', ['feature_extraction_date'])
    op.create_index('ix_imaging_features_radiomics', 'imaging_features', ['radiomics_features'], postgresql_using='gin')
    
    # Create unique constraints to prevent duplicate entries
    op.create_unique_constraint('uq_imaging_qc_series_date', 'imaging_qc', 
                               ['series_instance_uid', 'qc_date', 'qc_version'])
    op.create_unique_constraint('uq_imaging_features_series_pipeline', 'imaging_features', 
                               ['series_instance_uid', 'extraction_pipeline', 'pipeline_version'])


def downgrade() -> None:
    """Remove imaging tables and related indexes."""
    # Drop tables in reverse dependency order
    op.drop_table('imaging_features')
    op.drop_table('imaging_qc')
    op.drop_table('imaging_series')
    op.drop_table('imaging_studies')
    op.drop_table('imaging_subjects')
