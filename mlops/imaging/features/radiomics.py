"""
Basic radiomics feature extraction for medical images.

This module provides basic radiomics features including first-order statistics,
shape features, and texture analysis. It's a simplified implementation since
pyradiomics had installation issues.
"""

import logging
import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Optional, Union, Dict, Any
from scipy import ndimage
from sklearn.feature_extraction import image as sk_image

logger = logging.getLogger(__name__)


class RadiomicsExtractor:
    """
    Basic radiomics feature extraction.
    
    Provides simplified radiomics features including:
    - First-order statistics
    - Shape features  
    - Basic texture features
    """
    
    def __init__(self, enabled: bool = True, **kwargs):
        """
        Initialize radiomics extractor.
        
        Args:
            enabled: Whether radiomics extraction is enabled
            **kwargs: Additional parameters
        """
        self.enabled = enabled
        self.params = kwargs
        self.logger = logging.getLogger(__name__)
        
    def extract_features(
        self, 
        image_path: Union[str, Path],
        mask_path: Optional[Union[str, Path]] = None
    ) -> Dict[str, float]:
        """
        Extract radiomics features from image.
        
        Args:
            image_path: Path to input NIfTI image
            mask_path: Path to region mask (optional)
            
        Returns:
            Dictionary of radiomics features
        """
        if not self.enabled:
            self.logger.info("Radiomics extraction is disabled")
            return {}
            
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Input image not found: {image_path}")
            
        self.logger.info(f"Extracting radiomics features from {image_path}")
        
        # Load image
        img = nib.load(image_path)
        data = img.get_fdata()
        
        # Load mask if provided
        if mask_path and Path(mask_path).exists():
            mask_img = nib.load(mask_path)
            mask = mask_img.get_fdata() > 0
        else:
            # Create simple mask
            mask = self._create_roi_mask(data)
        
        # Extract features
        features = {}
        
        try:
            # First-order statistics
            features.update(self._extract_first_order_features(data, mask))
            
            # Shape features
            features.update(self._extract_shape_features(mask, img.header.get_zooms()[:3]))
            
            # Texture features
            features.update(self._extract_texture_features(data, mask))
            
            self.logger.info(f"Extracted {len(features)} radiomics features")
            
        except Exception as e:
            self.logger.error(f"Radiomics extraction failed: {e}")
            features['radiomics_extraction_failed'] = 1.0
        
        return features
    
    def _extract_first_order_features(self, data: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
        """Extract first-order statistical features."""
        features = {}
        
        # Apply mask
        roi_data = data[mask]
        
        if len(roi_data) == 0:
            return features
        
        # Basic statistics
        features['radiomics_mean'] = float(np.mean(roi_data))
        features['radiomics_std'] = float(np.std(roi_data))
        features['radiomics_median'] = float(np.median(roi_data))
        features['radiomics_min'] = float(np.min(roi_data))
        features['radiomics_max'] = float(np.max(roi_data))
        features['radiomics_range'] = float(np.max(roi_data) - np.min(roi_data))
        
        # Percentiles
        percentiles = [10, 25, 75, 90]
        for p in percentiles:
            features[f'radiomics_percentile_{p}'] = float(np.percentile(roi_data, p))
        
        # Interquartile range
        features['radiomics_iqr'] = float(np.percentile(roi_data, 75) - np.percentile(roi_data, 25))
        
        # Variance and coefficient of variation
        features['radiomics_variance'] = float(np.var(roi_data))
        if np.mean(roi_data) != 0:
            features['radiomics_cv'] = float(np.std(roi_data) / np.mean(roi_data))
        
        # Skewness and kurtosis (simplified)
        mean_val = np.mean(roi_data)
        std_val = np.std(roi_data)
        
        if std_val > 0:
            standardized = (roi_data - mean_val) / std_val
            features['radiomics_skewness'] = float(np.mean(standardized ** 3))
            features['radiomics_kurtosis'] = float(np.mean(standardized ** 4) - 3)
        
        # Energy and entropy
        features['radiomics_energy'] = float(np.sum(roi_data ** 2))
        
        # Histogram-based entropy
        hist, _ = np.histogram(roi_data, bins=50, density=True)
        hist = hist[hist > 0]  # Remove zero bins
        if len(hist) > 0:
            features['radiomics_entropy'] = float(-np.sum(hist * np.log2(hist + 1e-10)))
        
        # Root mean square
        features['radiomics_rms'] = float(np.sqrt(np.mean(roi_data ** 2)))
        
        return features
    
    def _extract_shape_features(self, mask: np.ndarray, voxel_spacing: tuple) -> Dict[str, float]:
        """Extract shape-based features."""
        features = {}
        
        if not np.any(mask):
            return features
        
        voxel_volume = np.prod(voxel_spacing)  # mm³
        
        # Volume
        volume_voxels = np.sum(mask)
        features['radiomics_volume_voxels'] = float(volume_voxels)
        features['radiomics_volume_mm3'] = float(volume_voxels * voxel_volume)
        
        # Surface area approximation
        surface_area = self._estimate_surface_area(mask, voxel_spacing)
        features['radiomics_surface_area_mm2'] = float(surface_area)
        
        # Sphericity
        if surface_area > 0:
            volume_mm3 = volume_voxels * voxel_volume
            sphericity = (np.pi ** (1/3)) * ((6 * volume_mm3) ** (2/3)) / surface_area
            features['radiomics_sphericity'] = float(sphericity)
        
        # Compactness (inverse of sphericity)
        if 'radiomics_sphericity' in features and features['radiomics_sphericity'] > 0:
            features['radiomics_compactness'] = float(1.0 / features['radiomics_sphericity'])
        
        # Elongation and flatness
        elongation_features = self._calculate_elongation_features(mask)
        features.update(elongation_features)
        
        return features
    
    def _extract_texture_features(self, data: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
        """Extract basic texture features."""
        features = {}
        
        roi_data = data[mask]
        
        if len(roi_data) == 0:
            return features
        
        # GLCM-like features (simplified)
        try:
            glcm_features = self._calculate_glcm_features(data, mask)
            features.update(glcm_features)
        except Exception as e:
            self.logger.warning(f"GLCM feature calculation failed: {e}")
        
        # Local binary patterns (simplified)
        try:
            lbp_features = self._calculate_lbp_features(data, mask)
            features.update(lbp_features)
        except Exception as e:
            self.logger.warning(f"LBP feature calculation failed: {e}")
        
        # Gradient-based texture
        try:
            gradient_features = self._calculate_gradient_texture_features(data, mask)
            features.update(gradient_features)
        except Exception as e:
            self.logger.warning(f"Gradient texture calculation failed: {e}")
        
        return features
    
    def _create_roi_mask(self, data: np.ndarray, threshold_percentile: float = 25) -> np.ndarray:
        """Create ROI mask using automatic thresholding."""
        # Calculate threshold
        threshold = np.percentile(data[data > 0], threshold_percentile)
        
        # Create mask
        mask = data > threshold
        
        # Clean up with morphological operations
        mask = ndimage.binary_closing(mask, structure=np.ones((3, 3, 3)), iterations=1)
        mask = ndimage.binary_opening(mask, structure=np.ones((3, 3, 3)), iterations=1)
        
        # Keep largest connected component
        labeled_mask, num_labels = ndimage.label(mask)
        if num_labels > 1:
            sizes = ndimage.sum(mask, labeled_mask, range(1, num_labels + 1))
            max_label = np.argmax(sizes) + 1
            mask = labeled_mask == max_label
        
        return mask
    
    def _estimate_surface_area(self, mask: np.ndarray, voxel_spacing: tuple) -> float:
        """Estimate surface area using gradient magnitude."""
        # Calculate gradient magnitude at mask boundary
        gradient = np.gradient(mask.astype(float))
        gradient_magnitude = np.sqrt(sum(g**2 for g in gradient))
        
        # Scale by voxel spacing
        surface_voxels = np.sum(gradient_magnitude > 0.1)
        surface_area = surface_voxels * np.mean(voxel_spacing[:2])**2  # Approximate
        
        return surface_area
    
    def _calculate_elongation_features(self, mask: np.ndarray) -> Dict[str, float]:
        """Calculate elongation features using PCA."""
        features = {}
        
        # Get coordinates of mask
        coords = np.where(mask)
        if len(coords[0]) == 0:
            return features
        
        points = np.column_stack(coords)
        
        # Center the points
        centroid = np.mean(points, axis=0)
        centered_points = points - centroid
        
        # Calculate covariance matrix and eigenvalues
        try:
            cov_matrix = np.cov(centered_points.T)
            eigenvalues = np.linalg.eigvals(cov_matrix)
            eigenvalues = np.sort(eigenvalues)[::-1]  # Sort descending
            
            if len(eigenvalues) >= 3 and eigenvalues[2] > 0:
                features['radiomics_elongation'] = float(eigenvalues[1] / eigenvalues[0])
                features['radiomics_flatness'] = float(eigenvalues[2] / eigenvalues[0])
                
        except np.linalg.LinAlgError:
            self.logger.warning("Eigenvalue calculation failed for elongation features")
        
        return features
    
    def _calculate_glcm_features(self, data: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
        """Calculate simplified GLCM features."""
        features = {}
        
        # Get ROI data
        roi_data = data[mask]
        
        if len(roi_data) < 4:  # Need minimum points
            return features
        
        # Quantize intensities
        n_levels = min(16, len(np.unique(roi_data)))
        if n_levels < 2:
            return features
            
        quantized = np.digitize(roi_data, bins=np.linspace(np.min(roi_data), np.max(roi_data), n_levels))
        
        # Create simplified co-occurrence matrix
        glcm = np.zeros((n_levels, n_levels))
        
        # Look at neighboring voxels (simplified)
        coords = np.where(mask)
        for i in range(len(coords[0]) - 1):
            val1 = quantized[i]
            val2 = quantized[i + 1]
            if 0 < val1 <= n_levels and 0 < val2 <= n_levels:
                glcm[val1-1, val2-1] += 1
        
        # Normalize
        if np.sum(glcm) > 0:
            glcm = glcm / np.sum(glcm)
            
            # Calculate GLCM features
            # Energy (ASM)
            features['radiomics_glcm_energy'] = float(np.sum(glcm ** 2))
            
            # Contrast
            contrast = 0
            for i in range(n_levels):
                for j in range(n_levels):
                    contrast += ((i - j) ** 2) * glcm[i, j]
            features['radiomics_glcm_contrast'] = float(contrast)
            
            # Homogeneity
            homogeneity = 0
            for i in range(n_levels):
                for j in range(n_levels):
                    homogeneity += glcm[i, j] / (1 + abs(i - j))
            features['radiomics_glcm_homogeneity'] = float(homogeneity)
            
            # Entropy
            entropy = -np.sum(glcm * np.log2(glcm + 1e-10))
            features['radiomics_glcm_entropy'] = float(entropy)
        
        return features
    
    def _calculate_lbp_features(self, data: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
        """Calculate simplified Local Binary Pattern features."""
        features = {}
        
        # Get 2D slice with most mask voxels for LBP calculation
        mask_counts = [np.sum(mask[:, :, z]) for z in range(mask.shape[2])]
        best_slice = np.argmax(mask_counts)
        
        slice_data = data[:, :, best_slice]
        slice_mask = mask[:, :, best_slice]
        
        if not np.any(slice_mask):
            return features
        
        # Simple LBP approximation
        try:
            # Use scikit-image-like approach but simplified
            from sklearn.feature_extraction.image import extract_patches_2d
            
            # Extract patches and calculate local patterns
            patches = extract_patches_2d(slice_data, (3, 3), max_patches=100)
            
            lbp_values = []
            for patch in patches:
                center = patch[1, 1]
                neighbors = [patch[0,0], patch[0,1], patch[0,2], patch[1,2], 
                           patch[2,2], patch[2,1], patch[2,0], patch[1,0]]
                
                binary_pattern = sum([(neighbor >= center) * (2**i) for i, neighbor in enumerate(neighbors)])
                lbp_values.append(binary_pattern)
            
            if lbp_values:
                features['radiomics_lbp_mean'] = float(np.mean(lbp_values))
                features['radiomics_lbp_std'] = float(np.std(lbp_values))
                
                # Uniformity (histogram-based)
                hist, _ = np.histogram(lbp_values, bins=16, density=True)
                uniformity = np.sum(hist ** 2)
                features['radiomics_lbp_uniformity'] = float(uniformity)
                
        except Exception as e:
            self.logger.warning(f"LBP calculation failed: {e}")
        
        return features
    
    def _calculate_gradient_texture_features(self, data: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
        """Calculate gradient-based texture features."""
        features = {}
        
        # Calculate gradients
        gradients = np.gradient(data)
        gradient_magnitude = np.sqrt(sum(g**2 for g in gradients))
        
        # Focus on ROI
        roi_gradients = gradient_magnitude[mask]
        
        if len(roi_gradients) > 0:
            features['radiomics_gradient_mean'] = float(np.mean(roi_gradients))
            features['radiomics_gradient_std'] = float(np.std(roi_gradients))
            features['radiomics_gradient_max'] = float(np.max(roi_gradients))
            
            # Gradient direction features
            for i, grad in enumerate(gradients):
                roi_grad = grad[mask]
                if len(roi_grad) > 0:
                    features[f'radiomics_gradient_{["x","y","z"][i]}_mean'] = float(np.mean(roi_grad))
                    features[f'radiomics_gradient_{["x","y","z"][i]}_std'] = float(np.std(roi_grad))
        
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
    
    def get_feature_categories(self, features: Dict[str, float]) -> Dict[str, int]:
        """Categorize extracted radiomics features."""
        categories = {
            'first_order': 0,
            'shape': 0,
            'glcm': 0,
            'lbp': 0,
            'gradient': 0
        }
        
        for feature_name in features.keys():
            if any(x in feature_name for x in ['mean', 'std', 'median', 'min', 'max', 'percentile', 'entropy', 'energy']):
                if 'glcm' not in feature_name and 'lbp' not in feature_name:
                    categories['first_order'] += 1
            elif any(x in feature_name for x in ['volume', 'surface', 'sphericity', 'elongation', 'flatness']):
                categories['shape'] += 1
            elif 'glcm' in feature_name:
                categories['glcm'] += 1
            elif 'lbp' in feature_name:
                categories['lbp'] += 1
            elif 'gradient' in feature_name:
                categories['gradient'] += 1
        
        return categories