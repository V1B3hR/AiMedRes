"""
Volumetric feature extraction for medical images.

This module provides volumetric measurements and morphometric analysis
for brain MRI images, including region-based volume calculations.
"""

import logging
import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Optional, Union, Dict, Any, List, Tuple
from scipy import ndimage
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)


class VolumetricFeatureExtractor:
    """
    Extract volumetric features from medical images.
    
    Provides comprehensive volumetric analysis including tissue segmentation,
    region-based measurements, and morphometric features.
    """
    
    def __init__(self, atlas: str = "AAL3", **kwargs):
        """
        Initialize volumetric feature extractor.
        
        Args:
            atlas: Atlas to use for region definition ('AAL3', 'Harvard-Oxford', 'custom')
            **kwargs: Additional parameters
        """
        self.atlas = atlas
        self.params = kwargs
        self.logger = logging.getLogger(__name__)
        
    def extract_features(
        self, 
        image_path: Union[str, Path],
        mask_path: Optional[Union[str, Path]] = None,
        atlas_path: Optional[Union[str, Path]] = None
    ) -> Dict[str, float]:
        """
        Extract volumetric features from image.
        
        Args:
            image_path: Path to input NIfTI image
            mask_path: Path to brain mask (optional)
            atlas_path: Path to atlas image (optional)
            
        Returns:
            Dictionary of extracted volumetric features
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Input image not found: {image_path}")
            
        self.logger.info(f"Extracting volumetric features from {image_path}")
        
        # Load image
        img = nib.load(image_path)
        data = img.get_fdata()
        voxel_size = np.prod(img.header.get_zooms()[:3])  # mm³
        
        # Load mask if provided
        if mask_path and Path(mask_path).exists():
            mask_img = nib.load(mask_path)
            mask = mask_img.get_fdata() > 0
        else:
            # Create simple brain mask
            mask = self._create_brain_mask(data)
        
        # Extract features
        features = {}
        
        # Basic volumetric measurements
        features.update(self._extract_basic_volumes(data, mask, voxel_size))
        
        # Tissue segmentation volumes
        features.update(self._extract_tissue_volumes(data, mask, voxel_size))
        
        # Morphometric features
        features.update(self._extract_morphometric_features(data, mask, voxel_size))
        
        # Regional volumes (if atlas provided)
        if atlas_path and Path(atlas_path).exists():
            features.update(self._extract_regional_volumes(data, atlas_path, voxel_size))
        else:
            # Use synthetic regions
            features.update(self._extract_synthetic_regional_volumes(data, mask, voxel_size))
        
        self.logger.info(f"Extracted {len(features)} volumetric features")
        return features
    
    def _extract_basic_volumes(self, data: np.ndarray, mask: np.ndarray, voxel_size: float) -> Dict[str, float]:
        """Extract basic volumetric measurements."""
        features = {}
        
        # Total brain volume
        brain_voxels = np.sum(mask)
        features['total_brain_volume_mm3'] = float(brain_voxels * voxel_size)
        features['total_brain_volume_voxels'] = float(brain_voxels)
        
        # Intracranial volume (approximate)
        icv_mask = self._estimate_icv_mask(data)
        icv_voxels = np.sum(icv_mask)
        features['intracranial_volume_mm3'] = float(icv_voxels * voxel_size)
        
        # Brain volume fraction
        if icv_voxels > 0:
            features['brain_volume_fraction'] = float(brain_voxels / icv_voxels)
        else:
            features['brain_volume_fraction'] = 0.0
        
        # Image dimensions
        features['image_volume_mm3'] = float(np.prod(data.shape) * voxel_size)
        features['voxel_size_mm3'] = float(voxel_size)
        
        return features
    
    def _extract_tissue_volumes(self, data: np.ndarray, mask: np.ndarray, voxel_size: float) -> Dict[str, float]:
        """Extract tissue-specific volumes using segmentation."""
        features = {}
        
        # Apply mask to focus on brain tissue
        brain_data = data * mask
        brain_intensities = brain_data[mask > 0]
        
        if len(brain_intensities) == 0:
            return features
        
        # Simple 3-class tissue segmentation using K-means
        # Classes: CSF, Gray Matter, White Matter
        try:
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            tissue_labels = kmeans.fit_predict(brain_intensities.reshape(-1, 1))
            
            # Map labels to tissue types based on intensity
            cluster_means = kmeans.cluster_centers_.flatten()
            sorted_indices = np.argsort(cluster_means)
            
            # Create full tissue map
            tissue_map = np.zeros(data.shape, dtype=int)
            tissue_map[mask > 0] = tissue_labels
            
            # Count voxels for each tissue type
            csf_voxels = np.sum(tissue_map == sorted_indices[0])  # Lowest intensity
            gm_voxels = np.sum(tissue_map == sorted_indices[1])   # Medium intensity  
            wm_voxels = np.sum(tissue_map == sorted_indices[2])   # Highest intensity
            
            features['csf_volume_mm3'] = float(csf_voxels * voxel_size)
            features['gray_matter_volume_mm3'] = float(gm_voxels * voxel_size)
            features['white_matter_volume_mm3'] = float(wm_voxels * voxel_size)
            
            # Tissue fractions
            total_tissue = csf_voxels + gm_voxels + wm_voxels
            if total_tissue > 0:
                features['csf_fraction'] = float(csf_voxels / total_tissue)
                features['gray_matter_fraction'] = float(gm_voxels / total_tissue)
                features['white_matter_fraction'] = float(wm_voxels / total_tissue)
            
        except Exception as e:
            self.logger.warning(f"Tissue segmentation failed: {e}")
            # Fallback to intensity-based segmentation
            features.update(self._simple_tissue_segmentation(brain_data, mask, voxel_size))
        
        return features
    
    def _extract_morphometric_features(self, data: np.ndarray, mask: np.ndarray, voxel_size: float) -> Dict[str, float]:
        """Extract morphometric shape features."""
        features = {}
        
        if not np.any(mask):
            return features
        
        # Surface area approximation
        surface_area = self._estimate_surface_area(mask, voxel_size)
        features['brain_surface_area_mm2'] = float(surface_area)
        
        # Sphericity (compactness measure)
        volume = np.sum(mask) * voxel_size
        if surface_area > 0:
            sphericity = (np.pi ** (1/3)) * ((6 * volume) ** (2/3)) / surface_area
            features['sphericity'] = float(sphericity)
        
        # Convex hull volume
        convex_volume = self._estimate_convex_hull_volume(mask, voxel_size)
        features['convex_hull_volume_mm3'] = float(convex_volume)
        
        # Convexity (volume / convex hull volume)
        if convex_volume > 0:
            features['convexity'] = float(volume / convex_volume)
        
        # Elongation measures
        features.update(self._compute_elongation_features(mask))
        
        # Centroid location (normalized by image dimensions)
        centroid = ndimage.center_of_mass(mask.astype(float))
        features['centroid_x_normalized'] = float(centroid[0] / data.shape[0])
        features['centroid_y_normalized'] = float(centroid[1] / data.shape[1])
        features['centroid_z_normalized'] = float(centroid[2] / data.shape[2])
        
        return features
    
    def _extract_regional_volumes(self, data: np.ndarray, atlas_path: Path, voxel_size: float) -> Dict[str, float]:
        """Extract volumes for atlas-defined regions."""
        features = {}
        
        try:
            # Load atlas
            atlas_img = nib.load(atlas_path)
            atlas_data = atlas_img.get_fdata()
            
            # Get unique regions
            regions = np.unique(atlas_data[atlas_data > 0])
            
            for region_id in regions:
                region_mask = atlas_data == region_id
                region_volume = np.sum(region_mask) * voxel_size
                
                # Calculate mean intensity in region
                if np.any(region_mask):
                    mean_intensity = np.mean(data[region_mask])
                    features[f'region_{int(region_id)}_volume_mm3'] = float(region_volume)
                    features[f'region_{int(region_id)}_mean_intensity'] = float(mean_intensity)
                    
        except Exception as e:
            self.logger.warning(f"Atlas-based regional extraction failed: {e}")
        
        return features
    
    def _extract_synthetic_regional_volumes(self, data: np.ndarray, mask: np.ndarray, voxel_size: float) -> Dict[str, float]:
        """Extract volumes for synthetic brain regions."""
        features = {}
        
        if not np.any(mask):
            return features
        
        # Create synthetic regions based on anatomical location
        shape = data.shape
        center = np.array(shape) // 2
        
        # Define synthetic regions
        regions = {
            'anterior': lambda x, y, z: x < center[0],
            'posterior': lambda x, y, z: x >= center[0],
            'left': lambda x, y, z: y < center[1],
            'right': lambda x, y, z: y >= center[1],
            'superior': lambda x, y, z: z >= center[2],
            'inferior': lambda x, y, z: z < center[2]
        }
        
        # Extract volumes for each region
        coords = np.mgrid[:shape[0], :shape[1], :shape[2]]
        
        for region_name, region_func in regions.items():
            region_mask = region_func(coords[0], coords[1], coords[2]) & mask
            region_volume = np.sum(region_mask) * voxel_size
            
            features[f'{region_name}_volume_mm3'] = float(region_volume)
            
            # Mean intensity in region
            if np.any(region_mask):
                mean_intensity = np.mean(data[region_mask])
                features[f'{region_name}_mean_intensity'] = float(mean_intensity)
        
        return features
    
    def _create_brain_mask(self, data: np.ndarray, threshold_percentile: float = 15) -> np.ndarray:
        """Create a simple brain mask using thresholding."""
        # Calculate threshold
        threshold = np.percentile(data[data > 0], threshold_percentile)
        
        # Create initial mask
        mask = data > threshold
        
        # Clean up mask with morphological operations
        mask = ndimage.binary_closing(mask, structure=np.ones((3, 3, 3)), iterations=2)
        mask = ndimage.binary_opening(mask, structure=np.ones((3, 3, 3)), iterations=1)
        
        # Keep largest connected component
        labeled_mask, num_labels = ndimage.label(mask)
        if num_labels > 1:
            sizes = ndimage.sum(mask, labeled_mask, range(1, num_labels + 1))
            max_label = np.argmax(sizes) + 1
            mask = labeled_mask == max_label
        
        return mask
    
    def _estimate_icv_mask(self, data: np.ndarray) -> np.ndarray:
        """Estimate intracranial volume mask."""
        # Simple approach: dilate brain mask to approximate skull boundary
        brain_mask = self._create_brain_mask(data)
        
        # Dilate to approximate intracranial space
        structure = np.ones((5, 5, 5))
        icv_mask = ndimage.binary_dilation(brain_mask, structure=structure, iterations=5)
        
        return icv_mask
    
    def _simple_tissue_segmentation(self, brain_data: np.ndarray, mask: np.ndarray, voxel_size: float) -> Dict[str, float]:
        """Simple intensity-based tissue segmentation fallback."""
        features = {}
        
        brain_intensities = brain_data[mask > 0]
        if len(brain_intensities) == 0:
            return features
        
        # Use intensity quartiles for segmentation
        q25 = np.percentile(brain_intensities, 25)
        q75 = np.percentile(brain_intensities, 75)
        
        # Create tissue maps
        csf_mask = (brain_data <= q25) & mask
        wm_mask = (brain_data >= q75) & mask
        gm_mask = (brain_data > q25) & (brain_data < q75) & mask
        
        # Calculate volumes
        features['csf_volume_mm3'] = float(np.sum(csf_mask) * voxel_size)
        features['gray_matter_volume_mm3'] = float(np.sum(gm_mask) * voxel_size)
        features['white_matter_volume_mm3'] = float(np.sum(wm_mask) * voxel_size)
        
        # Tissue fractions
        total_voxels = np.sum(mask)
        if total_voxels > 0:
            features['csf_fraction'] = float(np.sum(csf_mask) / total_voxels)
            features['gray_matter_fraction'] = float(np.sum(gm_mask) / total_voxels)
            features['white_matter_fraction'] = float(np.sum(wm_mask) / total_voxels)
        
        return features
    
    def _estimate_surface_area(self, mask: np.ndarray, voxel_size: float) -> float:
        """Estimate surface area using gradient magnitude."""
        # Calculate gradient magnitude at mask boundary
        gradient = np.gradient(mask.astype(float))
        gradient_magnitude = np.sqrt(sum(g**2 for g in gradient))
        
        # Surface area approximation
        surface_area = np.sum(gradient_magnitude > 0.1) * (voxel_size ** (2/3))
        
        return surface_area
    
    def _estimate_convex_hull_volume(self, mask: np.ndarray, voxel_size: float) -> float:
        """Estimate convex hull volume (simplified)."""
        # Find bounding box as approximation
        coords = np.where(mask)
        if len(coords[0]) == 0:
            return 0.0
        
        min_coords = [np.min(coords[i]) for i in range(3)]
        max_coords = [np.max(coords[i]) for i in range(3)]
        
        # Bounding box volume as convex hull approximation
        bbox_volume = np.prod([max_coords[i] - min_coords[i] + 1 for i in range(3)]) * voxel_size
        
        return bbox_volume
    
    def _compute_elongation_features(self, mask: np.ndarray) -> Dict[str, float]:
        """Compute elongation and shape features using moments."""
        features = {}
        
        # Calculate moments
        coords = np.where(mask)
        if len(coords[0]) == 0:
            return features
        
        points = np.column_stack(coords)
        
        # Center the points
        centroid = np.mean(points, axis=0)
        centered_points = points - centroid
        
        # Calculate covariance matrix
        cov_matrix = np.cov(centered_points.T)
        
        # Eigenvalues represent principal axes lengths
        eigenvalues = np.linalg.eigvals(cov_matrix)
        eigenvalues = np.sort(eigenvalues)[::-1]  # Sort descending
        
        if len(eigenvalues) >= 3 and eigenvalues[2] > 0:
            # Elongation ratios
            features['elongation_ratio_1'] = float(eigenvalues[0] / eigenvalues[1])
            features['elongation_ratio_2'] = float(eigenvalues[1] / eigenvalues[2])
            features['flatness_ratio'] = float(eigenvalues[2] / eigenvalues[0])
        
        return features
    
    def validate_inputs(self, image_path: Union[str, Path]) -> bool:
        """Validate input file exists and is a valid NIfTI image."""
        try:
            image_path = Path(image_path)
            if not image_path.exists():
                return False
                
            # Try to load as NIfTI
            nib.load(image_path)
            return True
            
        except Exception as e:
            self.logger.error(f"Input validation failed: {e}")
            return False
    
    def get_feature_summary(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Get summary statistics of extracted features."""
        if not features:
            return {}
        
        values = list(features.values())
        
        return {
            "total_features": len(features),
            "feature_mean": float(np.mean(values)),
            "feature_std": float(np.std(values)),
            "feature_min": float(np.min(values)),
            "feature_max": float(np.max(values)),
            "volume_features": len([k for k in features.keys() if "volume" in k.lower()]),
            "intensity_features": len([k for k in features.keys() if "intensity" in k.lower()]),
            "morphometric_features": len([k for k in features.keys() if any(x in k.lower() for x in ["sphericity", "convexity", "elongation"])])
        }