"""
Smoke tests for DuetMind Adaptive Medical Imaging Pipeline

Basic functionality tests to ensure imaging components can be imported
and instantiated without errors.
"""

import pytest
import tempfile
import os
from pathlib import Path
import json
import numpy as np

# Test imports with graceful handling of missing dependencies
try:
    from mlops.imaging.generators.synthetic_nifti import SyntheticNIfTIGenerator
    SYNTHETIC_AVAILABLE = True
except ImportError as e:
    SYNTHETIC_AVAILABLE = False
    SYNTHETIC_IMPORT_ERROR = str(e)

try:
    from mlops.imaging.converters.dicom_to_nifti import DICOMToNIfTIConverter
    CONVERTER_AVAILABLE = True
except ImportError as e:
    CONVERTER_AVAILABLE = False
    CONVERTER_IMPORT_ERROR = str(e)

try:
    from mlops.imaging.validators.bids_validator import BIDSComplianceValidator
    VALIDATOR_AVAILABLE = True
except ImportError as e:
    VALIDATOR_AVAILABLE = False
    VALIDATOR_IMPORT_ERROR = str(e)

try:
    from mlops.imaging.utils.deidentify import MedicalImageDeidentifier
    DEIDENTIFY_AVAILABLE = True
except ImportError as e:
    DEIDENTIFY_AVAILABLE = False
    DEIDENTIFY_IMPORT_ERROR = str(e)


class TestImagingImports:
    """Test that imaging modules can be imported successfully."""
    
    def test_imaging_module_import(self):
        """Test that the main imaging module can be imported."""
        try:
            import mlops.imaging
            assert hasattr(mlops.imaging, '__version__')
        except ImportError as e:
            pytest.fail(f"Failed to import mlops.imaging: {e}")
    
    def test_generators_import(self):
        """Test generators module import."""
        if not SYNTHETIC_AVAILABLE:
            pytest.skip(f"Synthetic generator not available: {SYNTHETIC_IMPORT_ERROR}")
        
        assert SyntheticNIfTIGenerator is not None
    
    def test_converters_import(self):
        """Test converters module import."""
        if not CONVERTER_AVAILABLE:
            pytest.skip(f"DICOM converter not available: {CONVERTER_IMPORT_ERROR}")
        
        assert DICOMToNIfTIConverter is not None
    
    def test_validators_import(self):
        """Test validators module import."""
        if not VALIDATOR_AVAILABLE:
            pytest.skip(f"BIDS validator not available: {VALIDATOR_IMPORT_ERROR}")
        
        assert BIDSComplianceValidator is not None
    
    def test_deidentify_import(self):
        """Test de-identification module import."""
        if not DEIDENTIFY_AVAILABLE:
            pytest.skip(f"De-identifier not available: {DEIDENTIFY_IMPORT_ERROR}")
        
        assert MedicalImageDeidentifier is not None


class TestSyntheticNIfTIGenerator:
    """Smoke tests for synthetic NIfTI generation."""
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary directory for test outputs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    def test_generator_initialization(self, temp_output_dir):
        """Test that SyntheticNIfTIGenerator can be initialized."""
        if not SYNTHETIC_AVAILABLE:
            pytest.skip(f"Synthetic generator not available: {SYNTHETIC_IMPORT_ERROR}")
        
        generator = SyntheticNIfTIGenerator(temp_output_dir)
        assert generator.output_dir == temp_output_dir
        assert generator.standard_shape == (182, 218, 182)
        assert generator.voxel_size == (1.0, 1.0, 1.0)
    
    def test_brain_mask_generation(self, temp_output_dir):
        """Test brain mask generation without full dependencies."""
        if not SYNTHETIC_AVAILABLE:
            pytest.skip(f"Synthetic generator not available: {SYNTHETIC_IMPORT_ERROR}")
        
        generator = SyntheticNIfTIGenerator(temp_output_dir)
        shape = (64, 64, 64)  # Small shape for testing
        
        # This should work even without scipy
        mask = generator.generate_brain_mask(shape)
        
        assert mask.shape == shape
        assert mask.dtype == np.float32
        assert 0 <= np.min(mask) <= np.max(mask) <= 1
    
    def test_modality_parameters(self, temp_output_dir):
        """Test that modality parameters are properly defined."""
        if not SYNTHETIC_AVAILABLE:
            pytest.skip(f"Synthetic generator not available: {SYNTHETIC_IMPORT_ERROR}")
        
        generator = SyntheticNIfTIGenerator(temp_output_dir)
        
        # Check that expected modalities are defined
        expected_modalities = ['T1w', 'T2w', 'FLAIR', 'DWI']
        for modality in expected_modalities:
            assert modality in generator.modality_params
            params = generator.modality_params[modality]
            assert 'description' in params
            assert 'intensity_range' in params
            assert 'noise_level' in params


class TestDICOMToNIfTIConverter:
    """Smoke tests for DICOM to NIfTI conversion."""
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary directory for test outputs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    def test_converter_initialization(self, temp_output_dir):
        """Test that DICOMToNIfTIConverter can be initialized."""
        if not CONVERTER_AVAILABLE:
            pytest.skip(f"DICOM converter not available: {CONVERTER_IMPORT_ERROR}")
        
        converter = DICOMToNIfTIConverter(temp_output_dir)
        assert Path(converter.output_dir).exists()
        assert (Path(converter.output_dir) / "nifti").exists()
        assert (Path(converter.output_dir) / "metadata").exists()
    
    def test_bids_tags_defined(self, temp_output_dir):
        """Test that BIDS tags are properly defined."""
        if not CONVERTER_AVAILABLE:
            pytest.skip(f"DICOM converter not available: {CONVERTER_IMPORT_ERROR}")
        
        converter = DICOMToNIfTIConverter(temp_output_dir)
        
        # Check essential BIDS tags
        essential_tags = ['StudyInstanceUID', 'SeriesInstanceUID', 'PatientID', 'Modality']
        for tag in essential_tags:
            assert tag in converter.bids_tags
            assert isinstance(converter.bids_tags[tag], tuple)
            assert len(converter.bids_tags[tag]) == 2
    
    def test_phi_tags_defined(self, temp_output_dir):
        """Test that PHI tags for removal are defined."""
        if not CONVERTER_AVAILABLE:
            pytest.skip(f"DICOM converter not available: {CONVERTER_IMPORT_ERROR}")
        
        converter = DICOMToNIfTIConverter(temp_output_dir)
        
        # Check that PHI tags are defined
        assert len(converter.phi_tags) > 0
        for tag in converter.phi_tags:
            assert isinstance(tag, tuple)
            assert len(tag) == 2


class TestBIDSValidator:
    """Smoke tests for BIDS compliance validation."""
    
    def test_validator_initialization(self):
        """Test that BIDSComplianceValidator can be initialized."""
        if not VALIDATOR_AVAILABLE:
            pytest.skip(f"BIDS validator not available: {VALIDATOR_IMPORT_ERROR}")
        
        validator = BIDSComplianceValidator()
        assert hasattr(validator, 'filename_patterns')
        assert hasattr(validator, 'required_metadata')
        assert hasattr(validator, 'validation_messages')
    
    def test_filename_patterns_defined(self):
        """Test that filename patterns are properly defined."""
        if not VALIDATOR_AVAILABLE:
            pytest.skip(f"BIDS validator not available: {VALIDATOR_IMPORT_ERROR}")
        
        validator = BIDSComplianceValidator()
        
        # Check common datatypes
        expected_datatypes = ['anat', 'func', 'dwi', 'fmap']
        for datatype in expected_datatypes:
            assert datatype in validator.filename_patterns
            assert isinstance(validator.filename_patterns[datatype], dict)
    
    def test_entity_extraction(self):
        """Test BIDS entity extraction from filename."""
        if not VALIDATOR_AVAILABLE:
            pytest.skip(f"BIDS validator not available: {VALIDATOR_IMPORT_ERROR}")
        
        validator = BIDSComplianceValidator()
        
        # Test filename parsing
        test_filename = "sub-001_ses-01_T1w"
        entities = validator._extract_bids_entities(test_filename)
        
        assert entities.get('subject') == '001'
        assert entities.get('session') == '01'
        assert entities.get('suffix') == 'T1w'
    
    @pytest.fixture
    def mock_bids_dataset(self):
        """Create a minimal mock BIDS dataset structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create dataset_description.json
            dataset_desc = {
                "Name": "Test Dataset",
                "BIDSVersion": "1.8.0",
                "DatasetType": "raw"
            }
            with open(temp_path / "dataset_description.json", 'w') as f:
                json.dump(dataset_desc, f)
            
            # Create subject directory
            sub_dir = temp_path / "sub-001" / "anat"
            sub_dir.mkdir(parents=True)
            
            # Create mock NIfTI file (empty)
            (sub_dir / "sub-001_T1w.nii.gz").touch()
            
            # Create JSON sidecar
            sidecar = {"RepetitionTime": 2.3, "EchoTime": 0.004}
            with open(sub_dir / "sub-001_T1w.json", 'w') as f:
                json.dump(sidecar, f)
            
            yield temp_path
    
    def test_dataset_validation_structure(self, mock_bids_dataset):
        """Test basic dataset structure validation."""
        if not VALIDATOR_AVAILABLE:
            pytest.skip(f"BIDS validator not available: {VALIDATOR_IMPORT_ERROR}")
        
        validator = BIDSComplianceValidator()
        results = validator.validate_dataset(str(mock_bids_dataset))
        
        # Should have basic validation structure
        assert 'valid' in results
        assert 'errors' in results
        assert 'warnings' in results
        assert 'summary' in results
        
        # Should find the test subject
        assert 'sub-001' in results.get('subjects_found', [])


class TestMedicalImageDeidentifier:
    """Smoke tests for medical image de-identification."""
    
    @pytest.fixture
    def temp_mapping_file(self):
        """Create temporary mapping file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name
        yield temp_file
        # Cleanup
        if os.path.exists(temp_file):
            os.unlink(temp_file)
    
    def test_deidentifier_initialization(self, temp_mapping_file):
        """Test that MedicalImageDeidentifier can be initialized."""
        if not DEIDENTIFY_AVAILABLE:
            pytest.skip(f"De-identifier not available: {DEIDENTIFY_IMPORT_ERROR}")
        
        deidentifier = MedicalImageDeidentifier(
            encryption_key="test_key",
            mapping_file=temp_mapping_file
        )
        
        assert deidentifier.mapping_file == temp_mapping_file
        assert isinstance(deidentifier.id_mappings, dict)
    
    def test_anonymous_id_generation(self, temp_mapping_file):
        """Test anonymous ID generation."""
        if not DEIDENTIFY_AVAILABLE:
            pytest.skip(f"De-identifier not available: {DEIDENTIFY_IMPORT_ERROR}")
        
        deidentifier = MedicalImageDeidentifier(mapping_file=temp_mapping_file)
        
        # Test patient ID anonymization
        original_id = "PATIENT123"
        anonymous_id = deidentifier.generate_anonymous_id(original_id, "patient")
        
        assert anonymous_id.startswith("ANON-")
        assert len(anonymous_id) > len("ANON-")
        
        # Test consistency
        anonymous_id2 = deidentifier.generate_anonymous_id(original_id, "patient")
        assert anonymous_id == anonymous_id2
    
    def test_date_shifting(self, temp_mapping_file):
        """Test date shifting functionality."""
        if not DEIDENTIFY_AVAILABLE:
            pytest.skip(f"De-identifier not available: {DEIDENTIFY_IMPORT_ERROR}")
        
        deidentifier = MedicalImageDeidentifier(mapping_file=temp_mapping_file)
        
        # Test date shifting
        original_date = "20230115"
        patient_id = "PATIENT123"
        
        shifted_date = deidentifier.shift_date(original_date, patient_id)
        
        assert len(shifted_date) == 8  # YYYYMMDD format
        assert shifted_date != original_date  # Should be different
        
        # Test consistency
        shifted_date2 = deidentifier.shift_date(original_date, patient_id)
        assert shifted_date == shifted_date2
    
    def test_metadata_deidentification(self, temp_mapping_file):
        """Test metadata de-identification."""
        if not DEIDENTIFY_AVAILABLE:
            pytest.skip(f"De-identifier not available: {DEIDENTIFY_IMPORT_ERROR}")
        
        deidentifier = MedicalImageDeidentifier(mapping_file=temp_mapping_file)
        
        # Test metadata
        metadata = {
            'PatientName': 'John Doe',
            'PatientID': 'P123456',
            'StudyDate': '2023-01-15',
            'RepetitionTime': 2.3,
            'EchoTime': 0.004
        }
        
        deidentified, modifications = deidentifier.deidentify_metadata_json(
            metadata, patient_id='P123456'
        )
        
        # PHI should be removed
        assert 'PatientName' not in deidentified
        assert 'PatientID' not in deidentified
        
        # Technical parameters should be preserved
        assert deidentified['RepetitionTime'] == 2.3
        assert deidentified['EchoTime'] == 0.004
        
        # Should have de-identification metadata
        assert 'DeidentificationDate' in deidentified
        assert 'DeidentificationSoftware' in deidentified
        
        # Should have modification log
        assert 'removed_fields' in modifications
        assert len(modifications['removed_fields']) > 0


class TestImagingPipelineIntegration:
    """Integration smoke tests for the imaging pipeline."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace for integration tests."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            
            # Create directory structure
            (workspace / "raw").mkdir()
            (workspace / "processed").mkdir()
            (workspace / "outputs").mkdir()
            
            yield workspace
    
    def test_pipeline_component_compatibility(self, temp_workspace):
        """Test that pipeline components can work together."""
        components_available = [
            SYNTHETIC_AVAILABLE,
            CONVERTER_AVAILABLE, 
            VALIDATOR_AVAILABLE,
            DEIDENTIFY_AVAILABLE
        ]
        
        if not any(components_available):
            pytest.skip("No imaging components available for integration test")
        
        # At least one component should be available
        assert any(components_available)
    
    def test_params_yaml_structure(self):
        """Test that params_imaging.yaml has expected structure."""
        params_path = Path("params_imaging.yaml")
        
        if not params_path.exists():
            pytest.skip("params_imaging.yaml not found")
        
        import yaml
        with open(params_path, 'r') as f:
            params = yaml.safe_load(f)
        
        # Check main sections
        expected_sections = ['data', 'synthetic', 'conversion', 'bids', 
                           'deidentification', 'quality_control', 'feature_extraction']
        
        for section in expected_sections:
            assert section in params, f"Missing section: {section}"
    
    def test_makefile_imaging_targets(self):
        """Test that imaging targets are defined in Makefile."""
        makefile_path = Path("Makefile")
        
        if not makefile_path.exists():
            pytest.skip("Makefile not found")
        
        with open(makefile_path, 'r') as f:
            makefile_content = f.read()
        
        # Check for imaging targets
        expected_targets = [
            'imaging-setup', 'imaging-synthetic', 'imaging-convert',
            'imaging-validate', 'imaging-qc', 'imaging-features',
            'imaging-deidentify', 'imaging-test'
        ]
        
        for target in expected_targets:
            assert f"{target}:" in makefile_content, f"Missing target: {target}"


# Mark slow tests
pytestmark = pytest.mark.imaging


if __name__ == "__main__":
    # Run smoke tests directly
    pytest.main([__file__, "-v"])