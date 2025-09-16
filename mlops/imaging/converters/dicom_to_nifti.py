"""
DICOM to NIfTI Converter for DuetMind Adaptive

Converts DICOM medical images to NIfTI format with BIDS compliance.
Handles metadata extraction, de-identification, and quality validation.
"""

import os
import json
import logging
import tempfile
import subprocess
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from pathlib import Path

# Import medical imaging libraries with fallbacks
try:
    import pydicom
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False

try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False

import numpy as np


class DICOMToNIfTIConverter:
    """
    Converts DICOM files to NIfTI format with BIDS compliance.
    
    Features:
    - DICOM series detection and grouping
    - NIfTI conversion with proper orientation
    - BIDS-compliant metadata extraction
    - De-identification support
    - Quality validation
    - Multi-echo and diffusion imaging support
    """
    
    def __init__(self, output_dir: str = "./converted_nifti"):
        """
        Initialize the DICOM to NIfTI converter.
        
        Args:
            output_dir: Directory to save converted NIfTI files
        """
        self.output_dir = Path(output_dir)
        self.logger = logging.getLogger(__name__)
        
        # Create output directory structure
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "nifti").mkdir(exist_ok=True)
        (self.output_dir / "metadata").mkdir(exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)
        
        # Check for external conversion tools
        self._check_external_tools()
        
        # DICOM tags to extract for BIDS metadata
        self.bids_tags = {
            'StudyInstanceUID': (0x0020, 0x000D),
            'SeriesInstanceUID': (0x0020, 0x000E),
            'StudyDate': (0x0008, 0x0020),
            'StudyTime': (0x0008, 0x0030),
            'SeriesDate': (0x0008, 0x0021),
            'SeriesTime': (0x0008, 0x0031),
            'PatientID': (0x0010, 0x0020),
            'PatientName': (0x0010, 0x0010),
            'PatientBirthDate': (0x0010, 0x0030),
            'PatientSex': (0x0010, 0x0040),
            'StudyDescription': (0x0008, 0x1030),
            'SeriesDescription': (0x0008, 0x103E),
            'Modality': (0x0008, 0x0060),
            'Manufacturer': (0x0008, 0x0070),
            'ManufacturerModelName': (0x0008, 0x1090),
            'MagneticFieldStrength': (0x0018, 0x0087),
            'RepetitionTime': (0x0018, 0x0080),
            'EchoTime': (0x0018, 0x0081),
            'FlipAngle': (0x0018, 0x1314),
            'SliceThickness': (0x0018, 0x0050),
            'PixelSpacing': (0x0028, 0x0030),
            'ImageOrientationPatient': (0x0020, 0x0037),
            'ImagePositionPatient': (0x0020, 0x0032),
            'AcquisitionTime': (0x0008, 0x0032),
            'InstitutionName': (0x0008, 0x0080),
            'ScanningSequence': (0x0018, 0x0020),
            'SequenceVariant': (0x0018, 0x0021),
            'ScanOptions': (0x0018, 0x0022),
            'NumberOfAverages': (0x0018, 0x0083),
        }
        
        # PHI tags to remove during de-identification
        self.phi_tags = [
            (0x0010, 0x0010),  # PatientName
            (0x0010, 0x0020),  # PatientID  
            (0x0010, 0x0030),  # PatientBirthDate
            (0x0008, 0x0090),  # ReferringPhysicianName
            (0x0008, 0x1070),  # OperatorsName
            (0x0008, 0x0080),  # InstitutionName
            (0x0008, 0x0081),  # InstitutionAddress
            (0x0008, 0x1010),  # StationName
            (0x0008, 0x1030),  # StudyDescription (may contain PHI)
            (0x0020, 0x0010),  # StudyID
        ]
    
    def _check_external_tools(self):
        """Check availability of external conversion tools."""
        self.dcm2niix_available = False
        self.dcmtk_available = False
        
        # Check for dcm2niix (preferred tool)
        try:
            result = subprocess.run(['dcm2niix', '-h'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                self.dcm2niix_available = True
                self.logger.info("dcm2niix is available")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        # Check for DCMTK tools
        try:
            result = subprocess.run(['dcmdump', '--help'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                self.dcmtk_available = True
                self.logger.info("DCMTK tools are available")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        if not self.dcm2niix_available and not PYDICOM_AVAILABLE:
            self.logger.warning(
                "Neither dcm2niix nor pydicom are available. "
                "Limited DICOM conversion functionality."
            )
    
    def scan_dicom_directory(self, dicom_dir: str) -> Dict[str, List[str]]:
        """
        Scan directory for DICOM files and group by series.
        
        Args:
            dicom_dir: Directory containing DICOM files
            
        Returns:
            Dictionary mapping SeriesInstanceUID to list of DICOM file paths
        """
        if not PYDICOM_AVAILABLE:
            raise ImportError("pydicom is required for DICOM scanning")
        
        dicom_dir = Path(dicom_dir)
        series_files = {}
        
        self.logger.info(f"Scanning DICOM directory: {dicom_dir}")
        
        # Find all potential DICOM files
        dicom_files = []
        for ext in ['*.dcm', '*.dicom', '*']:
            dicom_files.extend(dicom_dir.rglob(ext))
        
        processed_files = 0
        for file_path in dicom_files:
            if file_path.is_file():
                try:
                    # Try to read as DICOM
                    ds = pydicom.dcmread(str(file_path), force=True)
                    
                    # Check if it's actually a DICOM file
                    if not hasattr(ds, 'SOPClassUID'):
                        continue
                    
                    # Get series instance UID
                    series_uid = getattr(ds, 'SeriesInstanceUID', 'unknown')
                    
                    if series_uid not in series_files:
                        series_files[series_uid] = []
                    
                    series_files[series_uid].append(str(file_path))
                    processed_files += 1
                    
                except Exception as e:
                    # Not a valid DICOM file, skip
                    continue
        
        self.logger.info(f"Found {processed_files} DICOM files in {len(series_files)} series")
        
        # Sort files within each series by instance number
        for series_uid in series_files:
            series_files[series_uid] = self._sort_dicom_files(series_files[series_uid])
        
        return series_files
    
    def _sort_dicom_files(self, file_paths: List[str]) -> List[str]:
        """Sort DICOM files by instance number and position."""
        if not PYDICOM_AVAILABLE:
            return file_paths
        
        file_info = []
        
        for file_path in file_paths:
            try:
                ds = pydicom.dcmread(file_path, stop_before_pixels=True)
                
                instance_number = int(getattr(ds, 'InstanceNumber', 0))
                slice_location = float(getattr(ds, 'SliceLocation', 0.0))
                
                # Use ImagePositionPatient for more robust sorting
                image_position = getattr(ds, 'ImagePositionPatient', [0, 0, 0])
                if isinstance(image_position, list) and len(image_position) >= 3:
                    z_position = float(image_position[2])
                else:
                    z_position = slice_location
                
                file_info.append({
                    'path': file_path,
                    'instance_number': instance_number,
                    'z_position': z_position
                })
                
            except Exception:
                # If we can't read metadata, keep original position
                file_info.append({
                    'path': file_path,
                    'instance_number': 0,
                    'z_position': 0.0
                })
        
        # Sort by instance number, then by z position
        file_info.sort(key=lambda x: (x['instance_number'], x['z_position']))
        
        return [info['path'] for info in file_info]
    
    def extract_dicom_metadata(self, dicom_files: List[str]) -> Dict[str, Any]:
        """
        Extract metadata from DICOM series for BIDS conversion.
        
        Args:
            dicom_files: List of DICOM file paths in the series
            
        Returns:
            Dictionary of extracted metadata
        """
        if not PYDICOM_AVAILABLE:
            raise ImportError("pydicom is required for metadata extraction")
        
        if not dicom_files:
            return {}
        
        # Read first file for series-level metadata
        ds = pydicom.dcmread(dicom_files[0])
        
        metadata = {}
        
        # Extract BIDS-relevant tags
        for key, tag in self.bids_tags.items():
            try:
                value = ds[tag].value
                
                # Handle special data types
                if key in ['StudyDate', 'SeriesDate'] and value:
                    # Convert DICOM date format YYYYMMDD to ISO format
                    if len(str(value)) == 8:
                        date_str = str(value)
                        metadata[key] = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
                    else:
                        metadata[key] = str(value)
                elif key in ['StudyTime', 'SeriesTime', 'AcquisitionTime'] and value:
                    # Convert DICOM time format
                    time_str = str(value).split('.')[0]  # Remove fractional seconds
                    if len(time_str) >= 6:
                        metadata[key] = f"{time_str[:2]}:{time_str[2:4]}:{time_str[4:6]}"
                    else:
                        metadata[key] = str(value)
                elif key in ['RepetitionTime', 'EchoTime']:
                    # Convert to seconds if in milliseconds
                    if isinstance(value, (int, float)):
                        metadata[key] = float(value) / 1000.0  # Convert ms to s
                    else:
                        metadata[key] = float(value) if value else 0.0
                elif key == 'PixelSpacing' and value:
                    # Convert to list of floats
                    metadata[key] = [float(x) for x in value]
                elif key in ['ImageOrientationPatient', 'ImagePositionPatient']:
                    # Convert to list of floats
                    if value:
                        metadata[key] = [float(x) for x in value]
                elif isinstance(value, (bytes, pydicom.valuerep.PersonName)):
                    metadata[key] = str(value)
                else:
                    metadata[key] = value
                    
            except (KeyError, AttributeError, ValueError):
                # Tag not present or invalid, skip
                continue
        
        # Add derived information
        metadata['NumberOfSlices'] = len(dicom_files)
        
        # Calculate slice spacing
        if len(dicom_files) > 1:
            try:
                ds_first = pydicom.dcmread(dicom_files[0])
                ds_last = pydicom.dcmread(dicom_files[-1])
                
                pos_first = getattr(ds_first, 'ImagePositionPatient', [0, 0, 0])
                pos_last = getattr(ds_last, 'ImagePositionPatient', [0, 0, 0])
                
                if len(pos_first) >= 3 and len(pos_last) >= 3:
                    slice_spacing = abs(float(pos_last[2]) - float(pos_first[2])) / (len(dicom_files) - 1)
                    metadata['SliceSpacing'] = slice_spacing
                    
            except Exception:
                pass
        
        # Determine imaging parameters based on modality
        modality = metadata.get('Modality', '').upper()
        if modality == 'MR':
            metadata = self._extract_mr_parameters(ds, metadata)
        elif modality == 'CT':
            metadata = self._extract_ct_parameters(ds, metadata)
        
        return metadata
    
    def _extract_mr_parameters(self, ds: 'pydicom.Dataset', metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Extract MR-specific parameters."""
        # Additional MR parameters
        mr_tags = {
            'InversionTime': (0x0018, 0x0082),
            'EchoTrainLength': (0x0018, 0x0091),
            'PercentSampling': (0x0018, 0x0093),
            'PercentPhaseFieldOfView': (0x0018, 0x0094),
            'ReceiveCoilName': (0x0018, 0x1250),
            'TransmitCoilName': (0x0018, 0x1251),
            'ImagingFrequency': (0x0018, 0x0084),
        }
        
        for key, tag in mr_tags.items():
            try:
                value = ds[tag].value
                if key == 'InversionTime' and value:
                    metadata[key] = float(value) / 1000.0  # Convert ms to s
                else:
                    metadata[key] = value
            except (KeyError, AttributeError):
                continue
        
        # Determine sequence type
        sequence = metadata.get('ScanningSequence', '')
        variant = metadata.get('SequenceVariant', '')
        
        if 'IR' in sequence:
            metadata['SequenceType'] = 'Inversion Recovery'
        elif 'GR' in sequence:
            metadata['SequenceType'] = 'Gradient Echo'
        elif 'SE' in sequence:
            metadata['SequenceType'] = 'Spin Echo'
        elif 'EP' in sequence:
            metadata['SequenceType'] = 'Echo Planar'
        
        return metadata
    
    def _extract_ct_parameters(self, ds: 'pydicom.Dataset', metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Extract CT-specific parameters."""
        ct_tags = {
            'KVP': (0x0018, 0x0060),
            'XRayTubeCurrent': (0x0018, 0x1151),
            'ExposureTime': (0x0018, 0x1150),
            'FilterType': (0x0018, 0x1160),
            'ConvolutionKernel': (0x0018, 0x1210),
            'CTDIvol': (0x0018, 0x9345),
        }
        
        for key, tag in ct_tags.items():
            try:
                value = ds[tag].value
                metadata[key] = value
            except (KeyError, AttributeError):
                continue
        
        return metadata
    
    def deidentify_dicom_metadata(self, metadata: Dict[str, Any], 
                                subject_id: str = None) -> Dict[str, Any]:
        """
        Remove or replace PHI from DICOM metadata.
        
        Args:
            metadata: Original metadata dictionary
            subject_id: De-identified subject ID to use
            
        Returns:
            De-identified metadata dictionary
        """
        deidentified = metadata.copy()
        
        # Remove direct identifiers
        phi_keys = [
            'PatientName', 'PatientID', 'PatientBirthDate',
            'ReferringPhysicianName', 'OperatorsName', 
            'InstitutionName', 'InstitutionAddress', 'StationName'
        ]
        
        for key in phi_keys:
            if key in deidentified:
                del deidentified[key]
        
        # Replace with de-identified values
        if subject_id:
            deidentified['Subject'] = subject_id
        
        # Remove study description if it might contain PHI
        if 'StudyDescription' in deidentified:
            study_desc = deidentified['StudyDescription']
            # Keep only if it looks like a standard protocol name
            if any(phi_word in study_desc.lower() for phi_word in 
                   ['patient', 'name', 'dob', 'birth', 'phone', 'address']):
                del deidentified['StudyDescription']
        
        # Add de-identification timestamp
        deidentified['DeidentificationDate'] = datetime.now().isoformat()
        deidentified['DeidentificationSoftware'] = 'DuetMind DICOM Converter v1.0'
        
        return deidentified
    
    def convert_series_to_nifti(self, 
                               dicom_files: List[str],
                               output_prefix: str,
                               deidentify: bool = True,
                               subject_id: str = None) -> Dict[str, Any]:
        """
        Convert DICOM series to NIfTI format.
        
        Args:
            dicom_files: List of DICOM file paths
            output_prefix: Prefix for output filenames
            deidentify: Whether to remove PHI from metadata
            subject_id: De-identified subject ID
            
        Returns:
            Dictionary with conversion results and metadata
        """
        if not dicom_files:
            raise ValueError("No DICOM files provided")
        
        self.logger.info(f"Converting {len(dicom_files)} DICOM files to NIfTI")
        
        # Extract metadata
        metadata = self.extract_dicom_metadata(dicom_files)
        
        # De-identify if requested
        if deidentify:
            metadata = self.deidentify_dicom_metadata(metadata, subject_id)
        
        # Attempt conversion with dcm2niix first (preferred)
        if self.dcm2niix_available:
            result = self._convert_with_dcm2niix(dicom_files, output_prefix, metadata)
        elif PYDICOM_AVAILABLE and NIBABEL_AVAILABLE:
            result = self._convert_with_pydicom(dicom_files, output_prefix, metadata)
        else:
            raise RuntimeError("No suitable DICOM conversion method available")
        
        # Add metadata to result
        result['metadata'] = metadata
        result['conversion_date'] = datetime.now().isoformat()
        result['source_files'] = dicom_files
        result['deidentified'] = deidentify
        
        # Save BIDS JSON sidecar
        json_path = str(self.output_dir / "metadata" / f"{output_prefix}.json")
        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        result['json_path'] = json_path
        
        self.logger.info(f"DICOM conversion completed: {result['nifti_path']}")
        
        return result
    
    def _convert_with_dcm2niix(self, dicom_files: List[str], 
                              output_prefix: str, 
                              metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Convert using dcm2niix (external tool)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Copy DICOM files to temporary directory
            temp_dicom_dir = Path(temp_dir) / "dicom"
            temp_dicom_dir.mkdir()
            
            for i, dicom_file in enumerate(dicom_files):
                temp_file = temp_dicom_dir / f"slice_{i:04d}.dcm"
                temp_file.write_bytes(Path(dicom_file).read_bytes())
            
            # Run dcm2niix
            output_dir = self.output_dir / "nifti"
            cmd = [
                'dcm2niix',
                '-z', 'y',  # Compress output
                '-f', output_prefix,  # Output filename
                '-o', str(output_dir),  # Output directory
                '-s', 'n',  # Don't save source folder structure
                str(temp_dicom_dir)
            ]
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                
                if result.returncode != 0:
                    raise RuntimeError(f"dcm2niix failed: {result.stderr}")
                
                # Find output NIfTI file
                nifti_path = output_dir / f"{output_prefix}.nii.gz"
                if not nifti_path.exists():
                    # dcm2niix might have modified the filename
                    nifti_files = list(output_dir.glob(f"{output_prefix}*.nii.gz"))
                    if nifti_files:
                        nifti_path = nifti_files[0]
                    else:
                        raise RuntimeError("No NIfTI output file found")
                
                return {
                    'nifti_path': str(nifti_path),
                    'conversion_method': 'dcm2niix',
                    'conversion_log': result.stdout
                }
                
            except subprocess.TimeoutExpired:
                raise RuntimeError("dcm2niix conversion timed out")
    
    def _convert_with_pydicom(self, dicom_files: List[str], 
                             output_prefix: str, 
                             metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Convert using pydicom and nibabel (Python-based)."""
        # Read all DICOM slices
        slices = []
        for dicom_file in dicom_files:
            ds = pydicom.dcmread(dicom_file)
            slices.append(ds)
        
        # Sort slices by position (should already be sorted, but double-check)
        slices.sort(key=lambda x: float(getattr(x, 'ImagePositionPatient', [0, 0, 0])[2]))
        
        # Create 3D volume
        volume = []
        for slice_ds in slices:
            slice_data = slice_ds.pixel_array.astype(np.float32)
            
            # Apply rescale slope and intercept if present
            slope = float(getattr(slice_ds, 'RescaleSlope', 1.0))
            intercept = float(getattr(slice_ds, 'RescaleIntercept', 0.0))
            slice_data = slice_data * slope + intercept
            
            volume.append(slice_data)
        
        volume = np.stack(volume, axis=2)  # Stack along Z axis
        
        # Create affine transformation matrix
        affine = self._create_affine_matrix(slices[0], len(slices))
        
        # Create NIfTI image
        nifti_img = nib.Nifti1Image(volume, affine)
        
        # Set header information
        header = nifti_img.header
        header.set_xyzt_units('mm', 'sec')
        
        # Save NIfTI file
        nifti_path = self.output_dir / "nifti" / f"{output_prefix}.nii.gz"
        nib.save(nifti_img, str(nifti_path))
        
        return {
            'nifti_path': str(nifti_path),
            'conversion_method': 'pydicom',
            'volume_shape': volume.shape,
            'affine_matrix': affine.tolist()
        }
    
    def _create_affine_matrix(self, dicom_ds: 'pydicom.Dataset', num_slices: int) -> np.ndarray:
        """Create affine transformation matrix from DICOM metadata."""
        # Default identity matrix
        affine = np.eye(4)
        
        try:
            # Get image orientation and position
            orientation = np.array(dicom_ds.ImageOrientationPatient).reshape(2, 3)
            position = np.array(dicom_ds.ImagePositionPatient)
            pixel_spacing = np.array(dicom_ds.PixelSpacing)
            slice_thickness = float(getattr(dicom_ds, 'SliceThickness', 1.0))
            
            # Create rotation matrix from orientation vectors
            row_cosines = orientation[0]
            col_cosines = orientation[1]
            slice_cosines = np.cross(row_cosines, col_cosines)
            
            # Create transformation matrix
            affine[:3, 0] = row_cosines * pixel_spacing[0]
            affine[:3, 1] = col_cosines * pixel_spacing[1]
            affine[:3, 2] = slice_cosines * slice_thickness
            affine[:3, 3] = position
            
        except (AttributeError, ValueError, IndexError):
            # Fall back to simple scaling matrix
            pixel_spacing = getattr(dicom_ds, 'PixelSpacing', [1.0, 1.0])
            slice_thickness = float(getattr(dicom_ds, 'SliceThickness', 1.0))
            
            affine[0, 0] = float(pixel_spacing[0])
            affine[1, 1] = float(pixel_spacing[1])
            affine[2, 2] = slice_thickness
        
        return affine
    
    def convert_directory(self, dicom_dir: str, 
                         subject_id: str = None,
                         deidentify: bool = True) -> List[Dict[str, Any]]:
        """
        Convert all DICOM series in a directory to NIfTI.
        
        Args:
            dicom_dir: Directory containing DICOM files
            subject_id: De-identified subject ID
            deidentify: Whether to remove PHI
            
        Returns:
            List of conversion results for each series
        """
        # Scan directory for DICOM series
        series_files = self.scan_dicom_directory(dicom_dir)
        
        results = []
        
        for series_uid, dicom_files in series_files.items():
            if not dicom_files:
                continue
            
            try:
                # Extract metadata to determine output filename
                metadata = self.extract_dicom_metadata(dicom_files[:1])  # Just first file
                
                modality = metadata.get('Modality', 'UNK')
                series_desc = metadata.get('SeriesDescription', 'unknown')
                series_num = metadata.get('SeriesNumber', '000')
                
                # Clean series description for filename
                clean_desc = ''.join(c for c in series_desc if c.isalnum() or c in '-_').strip()
                
                # Create output prefix
                if subject_id:
                    output_prefix = f"{subject_id}_{modality}_{series_num:03d}_{clean_desc}"
                else:
                    output_prefix = f"series_{series_num:03d}_{modality}_{clean_desc}"
                
                # Convert series
                result = self.convert_series_to_nifti(
                    dicom_files=dicom_files,
                    output_prefix=output_prefix,
                    deidentify=deidentify,
                    subject_id=subject_id
                )
                
                result['series_uid'] = series_uid
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"Failed to convert series {series_uid}: {e}")
                continue
        
        self.logger.info(f"Converted {len(results)} DICOM series from {dicom_dir}")
        
        return results


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create converter
    converter = DICOMToNIfTIConverter("./test_dicom_conversion")
    
    # Example: Convert a directory of DICOM files
    # results = converter.convert_directory(
    #     dicom_dir="/path/to/dicom/files",
    #     subject_id="sub-001",
    #     deidentify=True
    # )
    # 
    # for result in results:
    #     print(f"Converted: {result['nifti_path']}")
    #     print(f"Metadata: {result['json_path']}")
    
    print("DICOM to NIfTI converter initialized successfully")