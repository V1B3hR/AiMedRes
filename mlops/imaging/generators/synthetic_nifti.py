"""
Synthetic NIfTI Generator for DuetMind Adaptive

Generates synthetic medical imaging data for testing, development, and pipeline validation.
Supports various imaging modalities and realistic anatomical structures.
"""

import numpy as np
import os
from typing import Optional, Dict, Any, Tuple, List
from datetime import datetime
import json
import logging

# Import medical imaging libraries (with fallbacks for basic functionality)
try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False
    logging.warning("nibabel not available. Limited functionality for NIfTI generation.")

try:
    from scipy import ndimage
    from scipy.stats import multivariate_normal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class SyntheticNIfTIGenerator:
    """
    Generates synthetic NIfTI files for medical imaging pipeline testing.
    
    Features:
    - Multiple imaging modalities (T1, T2, FLAIR, DWI, DTI)
    - Realistic brain anatomy simulation
    - Customizable noise and artifacts
    - BIDS-compatible metadata generation
    - Quality control metrics
    """
    
    def __init__(self, output_dir: str = "./synthetic_imaging_data"):
        """
        Initialize the synthetic NIfTI generator.
        
        Args:
            output_dir: Directory to save generated files
        """
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)
        
        # Create output directory structure
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "nifti"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "metadata"), exist_ok=True)
        
        # Standard brain dimensions and properties
        self.standard_shape = (182, 218, 182)  # MNI152 standard space
        self.voxel_size = (1.0, 1.0, 1.0)  # 1mm isotropic
        
        # Imaging parameters for different modalities
        self.modality_params = {
            'T1w': {
                'description': 'T1-weighted structural MRI',
                'tr': 2.3,  # Repetition time (seconds)
                'te': 0.0045,  # Echo time (seconds)
                'flip_angle': 9,  # degrees
                'intensity_range': (0, 4095),
                'noise_level': 0.02
            },
            'T2w': {
                'description': 'T2-weighted structural MRI',
                'tr': 3.0,
                'te': 0.08,
                'flip_angle': 90,
                'intensity_range': (0, 4095),
                'noise_level': 0.03
            },
            'FLAIR': {
                'description': 'Fluid-attenuated inversion recovery',
                'tr': 9.0,
                'te': 0.125,
                'ti': 2.5,  # Inversion time
                'intensity_range': (0, 4095),
                'noise_level': 0.025
            },
            'DWI': {
                'description': 'Diffusion-weighted imaging',
                'tr': 4.0,
                'te': 0.08,
                'b_value': 1000,  # s/mm²
                'directions': 32,
                'intensity_range': (0, 2047),
                'noise_level': 0.05
            }
        }
    
    def generate_brain_mask(self, shape: Tuple[int, int, int]) -> np.ndarray:
        """Generate a realistic brain mask."""
        if not SCIPY_AVAILABLE:
            # Simple ellipsoid mask if scipy not available
            z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
            center = (shape[0]//2, shape[1]//2, shape[2]//2)
            radii = (shape[0]//3, shape[1]//3, shape[2]//3)
            
            mask = ((z - center[0])**2 / radii[0]**2 + 
                   (y - center[1])**2 / radii[1]**2 + 
                   (x - center[2])**2 / radii[2]**2) <= 1
            return mask.astype(np.float32)
        
        # More realistic brain mask using multiple Gaussian components
        brain_mask = np.zeros(shape)
        
        # Central brain mass
        center = (shape[0]//2, shape[1]//2, shape[2]//2)
        cov = np.diag([40**2, 45**2, 35**2])  # Covariance for brain shape
        
        z, y, x = np.meshgrid(range(shape[0]), range(shape[1]), range(shape[2]), indexing='ij')
        pos = np.stack([z.ravel(), y.ravel(), x.ravel()], axis=1)
        
        # Main brain volume
        brain_prob = multivariate_normal.pdf(pos, mean=center, cov=cov)
        brain_prob = brain_prob.reshape(shape)
        brain_mask = (brain_prob > np.percentile(brain_prob, 75)).astype(np.float32)
        
        # Add cerebellum
        cerebellum_center = (shape[0]//2 + 20, shape[1]//2 - 30, shape[2]//2)
        cerebellum_cov = np.diag([15**2, 20**2, 15**2])
        cerebellum_prob = multivariate_normal.pdf(pos, mean=cerebellum_center, cov=cerebellum_cov)
        cerebellum_prob = cerebellum_prob.reshape(shape)
        cerebellum_mask = (cerebellum_prob > np.percentile(cerebellum_prob, 85)).astype(np.float32)
        
        brain_mask = np.maximum(brain_mask, cerebellum_mask)
        
        # Smooth the mask
        brain_mask = ndimage.gaussian_filter(brain_mask, sigma=2.0)
        brain_mask = (brain_mask > 0.3).astype(np.float32)
        
        return brain_mask
    
    def generate_tissue_segmentation(self, brain_mask: np.ndarray) -> Dict[str, np.ndarray]:
        """Generate tissue segmentation maps (GM, WM, CSF)."""
        shape = brain_mask.shape
        
        if not SCIPY_AVAILABLE:
            # Simple tissue segmentation
            gray_matter = brain_mask * np.random.uniform(0.4, 0.8, shape)
            white_matter = brain_mask * np.random.uniform(0.2, 0.6, shape)
            csf = brain_mask * np.random.uniform(0.1, 0.3, shape)
        else:
            # More realistic tissue distribution
            # Gray matter (cortical ribbon and subcortical structures)
            gray_matter = np.zeros(shape)
            
            # Cortical gray matter (outer layer)
            eroded_mask = ndimage.binary_erosion(brain_mask, iterations=3)
            cortical_gm = brain_mask - eroded_mask
            gray_matter += cortical_gm * 0.8
            
            # Subcortical gray matter structures
            center = (shape[0]//2, shape[1]//2, shape[2]//2)
            
            # Thalamus
            thalamus_center = (center[0], center[1], center[2])
            thalamus_mask = self._generate_subcortical_structure(shape, thalamus_center, (8, 12, 8))
            gray_matter += thalamus_mask * 0.7
            
            # Caudate and putamen
            caudate_center = (center[0] - 10, center[1] + 15, center[2] - 5)
            caudate_mask = self._generate_subcortical_structure(shape, caudate_center, (6, 8, 6))
            gray_matter += caudate_mask * 0.7
            
            # White matter (central regions)
            white_matter = eroded_mask.astype(np.float32)
            white_matter = ndimage.binary_erosion(white_matter, iterations=2).astype(np.float32)
            white_matter *= 0.9
            
            # CSF (ventricles and external CSF)
            csf = np.zeros(shape)
            
            # Lateral ventricles
            ventricle_center = (center[0], center[1] + 10, center[2])
            ventricle_mask = self._generate_subcortical_structure(shape, ventricle_center, (4, 6, 4))
            csf += ventricle_mask * 0.95
            
            # Add some external CSF
            external_csf = brain_mask - ndimage.binary_erosion(brain_mask, iterations=1)
            csf += external_csf * 0.3
        
        return {
            'gray_matter': gray_matter,
            'white_matter': white_matter,
            'csf': csf
        }
    
    def _generate_subcortical_structure(self, shape: Tuple[int, int, int], 
                                      center: Tuple[int, int, int], 
                                      radii: Tuple[int, int, int]) -> np.ndarray:
        """Generate a subcortical structure using ellipsoidal shape."""
        z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
        
        mask = ((z - center[0])**2 / radii[0]**2 + 
               (y - center[1])**2 / radii[1]**2 + 
               (x - center[2])**2 / radii[2]**2) <= 1
        
        if SCIPY_AVAILABLE:
            mask = ndimage.gaussian_filter(mask.astype(np.float32), sigma=1.0)
            mask = (mask > 0.3).astype(np.float32)
        
        return mask.astype(np.float32)
    
    def add_noise_and_artifacts(self, image: np.ndarray, modality: str) -> np.ndarray:
        """Add realistic noise and imaging artifacts."""
        params = self.modality_params[modality]
        noise_level = params['noise_level']
        
        # Gaussian noise
        noise = np.random.normal(0, noise_level * np.max(image), image.shape)
        noisy_image = image + noise
        
        if SCIPY_AVAILABLE:
            # Motion artifacts (slight blur)
            if np.random.random() < 0.3:  # 30% chance of motion
                motion_kernel = np.random.uniform(0.5, 1.5)
                noisy_image = ndimage.gaussian_filter(noisy_image, sigma=motion_kernel)
            
            # Intensity non-uniformity (bias field)
            if np.random.random() < 0.4:  # 40% chance of bias field
                bias_field = self._generate_bias_field(image.shape)
                noisy_image = noisy_image * bias_field
        
        # Ensure non-negative values
        noisy_image = np.maximum(noisy_image, 0)
        
        return noisy_image
    
    def _generate_bias_field(self, shape: Tuple[int, int, int]) -> np.ndarray:
        """Generate smooth bias field for intensity non-uniformity."""
        # Create low-frequency spatial variation
        bias_field = np.ones(shape)
        
        # Add polynomial bias
        z, y, x = np.meshgrid(
            np.linspace(-1, 1, shape[0]),
            np.linspace(-1, 1, shape[1]), 
            np.linspace(-1, 1, shape[2]),
            indexing='ij'
        )
        
        # Quadratic bias field
        bias_field *= (1 + 0.1 * (x**2 + y**2) + 0.05 * z**2)
        
        # Add some random smooth variation
        if SCIPY_AVAILABLE:
            random_field = np.random.normal(0, 0.05, [s//4 for s in shape])
            random_field = ndimage.zoom(random_field, 4, order=1)
            
            # Crop/pad to match original shape
            for i in range(3):
                if random_field.shape[i] > shape[i]:
                    start = (random_field.shape[i] - shape[i]) // 2
                    slices = [slice(None)] * 3
                    slices[i] = slice(start, start + shape[i])
                    random_field = random_field[tuple(slices)]
                elif random_field.shape[i] < shape[i]:
                    pad_width = [(0, 0)] * 3
                    pad_width[i] = (0, shape[i] - random_field.shape[i])
                    random_field = np.pad(random_field, pad_width, mode='edge')
            
            bias_field *= (1 + random_field)
        
        return bias_field
    
    def generate_synthetic_nifti(self, 
                                subject_id: str,
                                modality: str = 'T1w',
                                shape: Optional[Tuple[int, int, int]] = None,
                                add_pathology: bool = False) -> Dict[str, Any]:
        """
        Generate a synthetic NIfTI file with metadata.
        
        Args:
            subject_id: Subject identifier (e.g., 'sub-001')
            modality: Imaging modality ('T1w', 'T2w', 'FLAIR', 'DWI')
            shape: Image dimensions (defaults to standard_shape)
            add_pathology: Whether to add simulated pathology
            
        Returns:
            Dictionary with file paths and metadata
        """
        if not NIBABEL_AVAILABLE:
            raise ImportError("nibabel is required for NIfTI generation. Install with: pip install nibabel")
        
        if shape is None:
            shape = self.standard_shape
        
        if modality not in self.modality_params:
            raise ValueError(f"Unsupported modality: {modality}")
        
        params = self.modality_params[modality]
        
        # Generate brain mask and tissue segmentation
        self.logger.info(f"Generating brain anatomy for {subject_id}...")
        brain_mask = self.generate_brain_mask(shape)
        tissues = self.generate_tissue_segmentation(brain_mask)
        
        # Create intensity image based on modality
        image = np.zeros(shape)
        
        if modality == 'T1w':
            # T1: WM bright, GM intermediate, CSF dark
            image += tissues['white_matter'] * 3000
            image += tissues['gray_matter'] * 2000
            image += tissues['csf'] * 500
        elif modality == 'T2w':
            # T2: CSF bright, GM intermediate, WM dark
            image += tissues['csf'] * 3500
            image += tissues['gray_matter'] * 2500
            image += tissues['white_matter'] * 1000
        elif modality == 'FLAIR':
            # FLAIR: WM bright, GM intermediate, CSF suppressed
            image += tissues['white_matter'] * 3000
            image += tissues['gray_matter'] * 2000
            image += tissues['csf'] * 200
        elif modality == 'DWI':
            # DWI: Reduced diffusion appears bright
            image += tissues['white_matter'] * 1500
            image += tissues['gray_matter'] * 1200
            image += tissues['csf'] * 400
        
        # Add pathology if requested
        if add_pathology:
            image = self._add_simulated_pathology(image, brain_mask, modality)
        
        # Add noise and artifacts
        image = self.add_noise_and_artifacts(image, modality)
        
        # Create NIfTI image
        affine = np.eye(4)
        affine[:3, :3] *= self.voxel_size
        affine[:3, 3] = [-shape[0]//2, -shape[1]//2, -shape[2]//2]  # Center origin
        
        nifti_img = nib.Nifti1Image(image.astype(np.float32), affine)
        
        # Set header information
        header = nifti_img.header
        header.set_xyzt_units('mm', 'sec')
        header['pixdim'][1:4] = self.voxel_size
        
        # Generate file paths
        nifti_filename = f"{subject_id}_{modality}.nii.gz"
        nifti_path = os.path.join(self.output_dir, "nifti", nifti_filename)
        
        # Save NIfTI file
        nib.save(nifti_img, nifti_path)
        
        # Generate BIDS JSON sidecar
        json_metadata = self._generate_bids_metadata(subject_id, modality, params)
        json_filename = f"{subject_id}_{modality}.json"
        json_path = os.path.join(self.output_dir, "metadata", json_filename)
        
        with open(json_path, 'w') as f:
            json.dump(json_metadata, f, indent=2)
        
        # Calculate basic QC metrics
        qc_metrics = self._calculate_qc_metrics(image, brain_mask)
        
        self.logger.info(f"Generated synthetic {modality} for {subject_id}")
        
        return {
            'nifti_path': nifti_path,
            'json_path': json_path,
            'metadata': json_metadata,
            'qc_metrics': qc_metrics,
            'subject_id': subject_id,
            'modality': modality,
            'shape': shape,
            'voxel_size': self.voxel_size
        }
    
    def _add_simulated_pathology(self, image: np.ndarray, brain_mask: np.ndarray, modality: str) -> np.ndarray:
        """Add simulated pathological changes to the image."""
        # Simulate lesions (e.g., MS lesions, stroke)
        num_lesions = np.random.randint(1, 5)
        
        for _ in range(num_lesions):
            # Random lesion location within brain
            brain_indices = np.where(brain_mask > 0.5)
            if len(brain_indices[0]) == 0:
                continue
                
            idx = np.random.randint(len(brain_indices[0]))
            center = (brain_indices[0][idx], brain_indices[1][idx], brain_indices[2][idx])
            
            # Lesion size
            radius = np.random.uniform(2, 8)  # 2-8 voxels
            
            # Create lesion mask
            z, y, x = np.ogrid[:image.shape[0], :image.shape[1], :image.shape[2]]
            lesion_mask = ((z - center[0])**2 + (y - center[1])**2 + (x - center[2])**2) <= radius**2
            
            # Modify intensity based on modality
            if modality in ['T2w', 'FLAIR']:
                # Hyperintense lesions
                image[lesion_mask] *= 1.5
            elif modality == 'T1w':
                # Hypointense lesions
                image[lesion_mask] *= 0.7
        
        return image
    
    def _generate_bids_metadata(self, subject_id: str, modality: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate BIDS-compliant JSON metadata."""
        metadata = {
            'Subject': subject_id,
            'Modality': modality,
            'MagneticFieldStrength': 3.0,
            'Manufacturer': 'Synthetic',
            'ManufacturersModelName': 'DuetMind Simulator',
            'SoftwareVersions': 'DuetMind v1.0.0',
            'RepetitionTime': params.get('tr', 1.0),
            'EchoTime': params.get('te', 0.01),
            'FlipAngle': params.get('flip_angle', 90),
            'SliceThickness': self.voxel_size[2],
            'PixelSpacing': list(self.voxel_size[:2]),
            'ImageOrientationPatientDICOM': [1, 0, 0, 0, 1, 0],
            'ImagePositionPatientDICOM': [0, 0, 0],
            'InPlanePhaseEncodingDirectionDICOM': 'ROW',
            'ConversionSoftware': 'DuetMind Synthetic Generator',
            'ConversionSoftwareVersion': '1.0.0',
            'ScanningSequence': 'GR' if modality != 'DWI' else 'EP',
            'SequenceVariant': 'SP',
            'ScanOptions': 'SYNTH',
            'AcquisitionTime': datetime.now().strftime('%H:%M:%S'),
            'AcquisitionDate': datetime.now().strftime('%Y-%m-%d'),
            'SeriesDescription': params['description'],
            'ProtocolName': f'Synthetic {modality}',
            'GeneratedBy': 'DuetMind Adaptive Synthetic NIfTI Generator',
            'GenerationDate': datetime.now().isoformat()
        }
        
        # Add modality-specific parameters
        if modality == 'FLAIR':
            metadata['InversionTime'] = params.get('ti', 2.5)
        elif modality == 'DWI':
            metadata['BValue'] = params.get('b_value', 1000)
            metadata['NumberOfDirections'] = params.get('directions', 32)
        
        return metadata
    
    def _calculate_qc_metrics(self, image: np.ndarray, brain_mask: np.ndarray) -> Dict[str, float]:
        """Calculate basic quality control metrics."""
        brain_voxels = image[brain_mask > 0.5]
        
        if len(brain_voxels) == 0:
            return {'error': 'No brain voxels found'}
        
        # Background (outside brain) for noise estimation
        background_voxels = image[brain_mask <= 0.1]
        
        metrics = {
            'brain_volume_voxels': int(np.sum(brain_mask > 0.5)),
            'mean_intensity': float(np.mean(brain_voxels)),
            'std_intensity': float(np.std(brain_voxels)),
            'min_intensity': float(np.min(brain_voxels)),
            'max_intensity': float(np.max(brain_voxels)),
            'intensity_range': float(np.max(brain_voxels) - np.min(brain_voxels)),
        }
        
        # Signal-to-noise ratio estimation
        if len(background_voxels) > 0:
            noise_std = np.std(background_voxels)
            if noise_std > 0:
                metrics['snr_estimate'] = float(metrics['mean_intensity'] / noise_std)
            else:
                metrics['snr_estimate'] = float('inf')
        
        # Contrast measures
        percentiles = np.percentile(brain_voxels, [5, 25, 50, 75, 95])
        metrics.update({
            'p05': float(percentiles[0]),
            'p25': float(percentiles[1]),
            'p50': float(percentiles[2]),
            'p75': float(percentiles[3]),
            'p95': float(percentiles[4]),
            'iqr': float(percentiles[3] - percentiles[1])
        })
        
        return metrics
    
    def generate_dataset(self, 
                        num_subjects: int = 10,
                        modalities: List[str] = ['T1w'],
                        pathology_rate: float = 0.3) -> List[Dict[str, Any]]:
        """
        Generate a complete synthetic dataset.
        
        Args:
            num_subjects: Number of subjects to generate
            modalities: List of modalities to generate for each subject
            pathology_rate: Fraction of subjects with simulated pathology
            
        Returns:
            List of generated file information dictionaries
        """
        dataset_info = []
        
        self.logger.info(f"Generating synthetic dataset: {num_subjects} subjects, modalities: {modalities}")
        
        for i in range(num_subjects):
            subject_id = f"sub-{i+1:03d}"
            
            # Decide if this subject has pathology
            has_pathology = np.random.random() < pathology_rate
            
            subject_info = {
                'subject_id': subject_id,
                'has_pathology': has_pathology,
                'modalities': {}
            }
            
            for modality in modalities:
                try:
                    result = self.generate_synthetic_nifti(
                        subject_id=subject_id,
                        modality=modality,
                        add_pathology=has_pathology
                    )
                    subject_info['modalities'][modality] = result
                    dataset_info.append(result)
                    
                except Exception as e:
                    self.logger.error(f"Failed to generate {modality} for {subject_id}: {e}")
        
        # Save dataset summary
        summary_path = os.path.join(self.output_dir, "dataset_summary.json")
        with open(summary_path, 'w') as f:
            json.dump({
                'dataset_info': dataset_info,
                'generation_date': datetime.now().isoformat(),
                'num_subjects': num_subjects,
                'modalities': modalities,
                'pathology_rate': pathology_rate
            }, f, indent=2)
        
        self.logger.info(f"Dataset generation complete. Summary saved to {summary_path}")
        
        return dataset_info


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create generator
    generator = SyntheticNIfTIGenerator("./test_synthetic_data")
    
    # Generate a single T1-weighted image
    result = generator.generate_synthetic_nifti(
        subject_id="sub-test001",
        modality="T1w",
        add_pathology=True
    )
    
    print("Generated files:")
    print(f"NIfTI: {result['nifti_path']}")
    print(f"JSON: {result['json_path']}")
    print(f"QC Metrics: {result['qc_metrics']}")
    
    # Generate a small dataset
    dataset = generator.generate_dataset(
        num_subjects=3,
        modalities=['T1w', 'T2w'],
        pathology_rate=0.5
    )
    
    print(f"\nGenerated dataset with {len(dataset)} images")