"""
Quality control metrics for medical images.

This module provides comprehensive quality assessment metrics for medical images,
including signal-to-noise ratio, contrast measures, motion artifacts, and more.
"""

import logging
import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Optional, Union, Dict, Any, List, Tuple
from scipy import ndimage, stats
from sklearn.metrics import mutual_info_score

logger = logging.getLogger(__name__)


class QualityControlMetrics:
    """
    Quality control assessment for medical images.
    
    Provides comprehensive QC metrics including SNR, contrast,
    motion detection, artifact assessment, and coverage analysis.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize quality control metrics calculator.
        
        Args:
            **kwargs: Additional parameters for QC calculations
        """
        self.params = kwargs
        self.logger = logging.getLogger(__name__)
        
    def calculate_qc_metrics(
        self, 
        image_path: Union[str, Path],
        mask_path: Optional[Union[str, Path]] = None,
        background_mask_path: Optional[Union[str, Path]] = None
    ) -> Dict[str, float]:
        """
        Calculate comprehensive quality control metrics.
        
        Args:
            image_path: Path to input NIfTI image
            mask_path: Path to foreground/brain mask (optional)
            background_mask_path: Path to background mask (optional)
            
        Returns:
            Dictionary of QC metrics
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Input image not found: {image_path}")
            
        self.logger.info(f"Calculating QC metrics for {image_path}")
        
        # Load image
        img = nib.load(image_path)
        data = img.get_fdata()
        
        # Load masks
        if mask_path and Path(mask_path).exists():
            mask_img = nib.load(mask_path)
            fg_mask = mask_img.get_fdata() > 0
        else:
            fg_mask = self._create_foreground_mask(data)
        
        if background_mask_path and Path(background_mask_path).exists():
            bg_mask_img = nib.load(background_mask_path)
            bg_mask = bg_mask_img.get_fdata() > 0
        else:
            bg_mask = self._create_background_mask(data, fg_mask)
        
        # Calculate QC metrics
        qc_metrics = {}
        
        # Signal-to-noise ratio metrics
        qc_metrics.update(self._calculate_snr_metrics(data, fg_mask, bg_mask))
        
        # Contrast metrics
        qc_metrics.update(self._calculate_contrast_metrics(data, fg_mask))
        
        # Motion and artifact metrics
        qc_metrics.update(self._calculate_motion_metrics(data, fg_mask))
        
        # Coverage and field-of-view metrics
        qc_metrics.update(self._calculate_coverage_metrics(data, fg_mask))
        
        # Intensity distribution metrics
        qc_metrics.update(self._calculate_intensity_metrics(data, fg_mask))
        
        # Spatial coherence metrics
        qc_metrics.update(self._calculate_spatial_metrics(data, fg_mask))
        
        self.logger.info(f"Calculated {len(qc_metrics)} QC metrics")
        return qc_metrics
    
    def _calculate_snr_metrics(self, data: np.ndarray, fg_mask: np.ndarray, bg_mask: np.ndarray) -> Dict[str, float]:
        """Calculate signal-to-noise ratio metrics."""
        metrics = {}
        
        # Extract foreground and background intensities
        fg_intensities = data[fg_mask]
        bg_intensities = data[bg_mask]
        
        if len(fg_intensities) == 0 or len(bg_intensities) == 0:
            return metrics
        
        # Basic SNR (mean signal / std noise)
        signal_mean = np.mean(fg_intensities)
        noise_std = np.std(bg_intensities)
        
        if noise_std > 0:
            metrics['snr_basic'] = float(signal_mean / noise_std)
        else:
            metrics['snr_basic'] = float('inf')
        
        # CNR (Contrast-to-Noise Ratio)
        noise_mean = np.mean(bg_intensities)
        if noise_std > 0:
            metrics['cnr'] = float((signal_mean - noise_mean) / noise_std)
        else:
            metrics['cnr'] = float('inf')
        
        # SNR using robust statistics (median-based)
        signal_median = np.median(fg_intensities)
        noise_mad = stats.median_abs_deviation(bg_intensities)
        
        if noise_mad > 0:
            metrics['snr_robust'] = float(signal_median / noise_mad)
        else:
            metrics['snr_robust'] = float('inf')
        
        # Signal variability
        signal_std = np.std(fg_intensities)
        if signal_mean > 0:
            metrics['signal_cv'] = float(signal_std / signal_mean)
        else:
            metrics['signal_cv'] = float('inf')
        
        # Background uniformity
        if noise_mean > 0:
            metrics['background_cv'] = float(noise_std / noise_mean)
        else:
            metrics['background_cv'] = float('inf')
        
        return metrics
    
    def _calculate_contrast_metrics(self, data: np.ndarray, fg_mask: np.ndarray) -> Dict[str, float]:
        """Calculate image contrast metrics."""
        metrics = {}
        
        fg_intensities = data[fg_mask]
        
        if len(fg_intensities) == 0:
            return metrics
        
        # RMS contrast
        mean_intensity = np.mean(fg_intensities)
        rms_contrast = np.sqrt(np.mean((fg_intensities - mean_intensity) ** 2))
        if mean_intensity > 0:
            metrics['rms_contrast'] = float(rms_contrast / mean_intensity)
        else:
            metrics['rms_contrast'] = 0.0
        
        # Michelson contrast (for periodic patterns)
        max_intensity = np.max(fg_intensities)
        min_intensity = np.min(fg_intensities)
        if max_intensity + min_intensity > 0:
            metrics['michelson_contrast'] = float((max_intensity - min_intensity) / (max_intensity + min_intensity))
        else:
            metrics['michelson_contrast'] = 0.0
        
        # Weber contrast
        if mean_intensity > 0:
            metrics['weber_contrast'] = float((max_intensity - mean_intensity) / mean_intensity)
        else:
            metrics['weber_contrast'] = 0.0
        
        # Local contrast variation
        local_contrasts = self._calculate_local_contrasts(data, fg_mask)
        if len(local_contrasts) > 0:
            metrics['local_contrast_mean'] = float(np.mean(local_contrasts))
            metrics['local_contrast_std'] = float(np.std(local_contrasts))
        
        return metrics
    
    def _calculate_motion_metrics(self, data: np.ndarray, fg_mask: np.ndarray) -> Dict[str, float]:
        """Calculate motion and artifact metrics."""
        metrics = {}
        
        # Edge sharpness (motion blur indicator)
        edge_sharpness = self._calculate_edge_sharpness(data, fg_mask)
        metrics['edge_sharpness'] = float(edge_sharpness)
        
        # Ghost artifact detection (simplified)
        ghost_score = self._detect_ghost_artifacts(data, fg_mask)
        metrics['ghost_artifact_score'] = float(ghost_score)
        
        # Ringing artifact detection
        ringing_score = self._detect_ringing_artifacts(data, fg_mask)
        metrics['ringing_artifact_score'] = float(ringing_score)
        
        # Motion score based on intensity variations
        motion_score = self._calculate_motion_score(data, fg_mask)
        metrics['motion_score'] = float(motion_score)
        
        return metrics
    
    def _calculate_coverage_metrics(self, data: np.ndarray, fg_mask: np.ndarray) -> Dict[str, float]:
        """Calculate field-of-view and coverage metrics."""
        metrics = {}
        
        # Calculate coverage ratios
        total_voxels = np.prod(data.shape)
        fg_voxels = np.sum(fg_mask)
        
        metrics['coverage_ratio'] = float(fg_voxels / total_voxels)
        
        # Bounding box analysis
        if np.any(fg_mask):
            coords = np.where(fg_mask)
            
            # Bounding box dimensions (normalized)
            bbox_dims = []
            for i in range(3):
                min_coord = np.min(coords[i])
                max_coord = np.max(coords[i])
                bbox_dims.append((max_coord - min_coord + 1) / data.shape[i])
            
            metrics['bbox_x_ratio'] = float(bbox_dims[0])
            metrics['bbox_y_ratio'] = float(bbox_dims[1])
            metrics['bbox_z_ratio'] = float(bbox_dims[2])
            
            # Centroid position (normalized)
            centroid = ndimage.center_of_mass(fg_mask.astype(float))
            metrics['centroid_x_normalized'] = float(centroid[0] / data.shape[0])
            metrics['centroid_y_normalized'] = float(centroid[1] / data.shape[1])
            metrics['centroid_z_normalized'] = float(centroid[2] / data.shape[2])
            
            # Coverage uniformity (how well the object fills the bounding box)
            bbox_volume = np.prod(bbox_dims) * total_voxels
            if bbox_volume > 0:
                metrics['coverage_uniformity'] = float(fg_voxels / bbox_volume)
        
        return metrics
    
    def _calculate_intensity_metrics(self, data: np.ndarray, fg_mask: np.ndarray) -> Dict[str, float]:
        """Calculate intensity distribution metrics."""
        metrics = {}
        
        fg_intensities = data[fg_mask]
        
        if len(fg_intensities) == 0:
            return metrics
        
        # Basic statistics
        metrics['intensity_mean'] = float(np.mean(fg_intensities))
        metrics['intensity_std'] = float(np.std(fg_intensities))
        metrics['intensity_median'] = float(np.median(fg_intensities))
        metrics['intensity_mad'] = float(stats.median_abs_deviation(fg_intensities))
        
        # Percentiles
        percentiles = [5, 25, 75, 95]
        for p in percentiles:
            metrics[f'intensity_p{p}'] = float(np.percentile(fg_intensities, p))
        
        # Distribution shape
        metrics['intensity_skewness'] = float(stats.skew(fg_intensities))
        metrics['intensity_kurtosis'] = float(stats.kurtosis(fg_intensities))
        
        # Dynamic range
        min_intensity = np.min(fg_intensities)
        max_intensity = np.max(fg_intensities)
        metrics['dynamic_range'] = float(max_intensity - min_intensity)
        
        if np.mean(fg_intensities) > 0:
            metrics['dynamic_range_ratio'] = float((max_intensity - min_intensity) / np.mean(fg_intensities))
        
        return metrics
    
    def _calculate_spatial_metrics(self, data: np.ndarray, fg_mask: np.ndarray) -> Dict[str, float]:
        """Calculate spatial coherence and smoothness metrics."""
        metrics = {}
        
        # Spatial smoothness using local variance
        smoothness = self._calculate_spatial_smoothness(data, fg_mask)
        metrics['spatial_smoothness'] = float(smoothness)
        
        # Texture analysis using GLCM approximation
        texture_metrics = self._calculate_texture_metrics(data, fg_mask)
        metrics.update(texture_metrics)
        
        # Gradient magnitude statistics
        gradient_stats = self._calculate_gradient_stats(data, fg_mask)
        metrics.update(gradient_stats)
        
        return metrics
    
    def _create_foreground_mask(self, data: np.ndarray, threshold_percentile: float = 15) -> np.ndarray:
        """Create foreground mask using automatic thresholding."""
        # Calculate threshold
        threshold = np.percentile(data[data > 0], threshold_percentile)
        
        # Create mask
        mask = data > threshold
        
        # Clean up with morphological operations
        mask = ndimage.binary_closing(mask, structure=np.ones((3, 3, 3)), iterations=2)
        mask = ndimage.binary_opening(mask, structure=np.ones((3, 3, 3)), iterations=1)
        
        # Keep largest connected component
        labeled_mask, num_labels = ndimage.label(mask)
        if num_labels > 1:
            sizes = ndimage.sum(mask, labeled_mask, range(1, num_labels + 1))
            max_label = np.argmax(sizes) + 1
            mask = labeled_mask == max_label
        
        return mask
    
    def _create_background_mask(self, data: np.ndarray, fg_mask: np.ndarray) -> np.ndarray:
        """Create background mask for noise estimation."""
        # Background is low-intensity areas outside foreground
        low_intensity_threshold = np.percentile(data[data > 0], 5)
        bg_candidate = (data <= low_intensity_threshold) & (data > 0)
        
        # Remove areas too close to foreground
        dilated_fg = ndimage.binary_dilation(fg_mask, structure=np.ones((5, 5, 5)))
        bg_mask = bg_candidate & ~dilated_fg
        
        return bg_mask
    
    def _calculate_local_contrasts(self, data: np.ndarray, fg_mask: np.ndarray, window_size: int = 5) -> np.ndarray:
        """Calculate local contrast measures."""
        local_contrasts = []
        
        # Use sliding window to calculate local contrasts
        coords = np.where(fg_mask)
        
        if len(coords[0]) == 0:
            return np.array([])
        
        for i in range(0, len(coords[0]), window_size):
            end_idx = min(i + window_size, len(coords[0]))
            local_coords = tuple([coords[j][i:end_idx] for j in range(3)])
            
            if len(local_coords[0]) > 1:
                try:
                    local_intensities = data[local_coords]
                    if len(local_intensities) > 0:
                        local_std = np.std(local_intensities)
                        local_mean = np.mean(local_intensities)
                        if local_mean > 0:
                            local_contrasts.append(local_std / local_mean)
                except IndexError:
                    # Skip invalid coordinates
                    continue
        
        return np.array(local_contrasts)
    
    def _calculate_edge_sharpness(self, data: np.ndarray, fg_mask: np.ndarray) -> float:
        """Calculate edge sharpness as motion blur indicator."""
        # Calculate gradient magnitude
        gradients = np.gradient(data)
        gradient_magnitude = np.sqrt(sum(g**2 for g in gradients))
        
        # Focus on foreground edges
        edge_gradients = gradient_magnitude[fg_mask]
        
        if len(edge_gradients) > 0:
            # Use 90th percentile as sharpness measure
            sharpness = np.percentile(edge_gradients, 90)
        else:
            sharpness = 0.0
        
        return sharpness
    
    def _detect_ghost_artifacts(self, data: np.ndarray, fg_mask: np.ndarray) -> float:
        """Detect ghost artifacts (simplified)."""
        # Look for periodic patterns in the background
        bg_mask = ~fg_mask & (data > 0)
        
        if not np.any(bg_mask):
            return 0.0
        
        bg_intensities = data[bg_mask]
        
        # Calculate variance in background (ghosts increase background variance)
        bg_variance = np.var(bg_intensities)
        bg_mean = np.mean(bg_intensities)
        
        if bg_mean > 0:
            ghost_score = bg_variance / (bg_mean ** 2)
        else:
            ghost_score = 0.0
        
        return min(ghost_score, 1.0)  # Normalize to [0, 1]
    
    def _detect_ringing_artifacts(self, data: np.ndarray, fg_mask: np.ndarray) -> float:
        """Detect ringing artifacts near edges."""
        # Calculate second derivative (Laplacian) to detect ringing
        laplacian = ndimage.laplace(data)
        
        # Focus on areas near foreground edges
        dilated_fg = ndimage.binary_dilation(fg_mask, structure=np.ones((3, 3, 3)))
        edge_region = dilated_fg & ~fg_mask
        
        if not np.any(edge_region):
            return 0.0
        
        edge_laplacian = laplacian[edge_region]
        ringing_score = np.std(edge_laplacian) / (np.std(data[fg_mask]) + 1e-10)
        
        return min(ringing_score, 1.0)  # Normalize to [0, 1]
    
    def _calculate_motion_score(self, data: np.ndarray, fg_mask: np.ndarray) -> float:
        """Calculate overall motion score."""
        # Motion affects image sharpness and introduces artifacts
        edge_sharpness = self._calculate_edge_sharpness(data, fg_mask)
        ghost_score = self._detect_ghost_artifacts(data, fg_mask)
        
        # Combine measures (lower sharpness and higher ghosts indicate more motion)
        max_sharpness = np.percentile(data[fg_mask], 95) if np.any(fg_mask) else 1.0
        normalized_sharpness = 1.0 - (edge_sharpness / (max_sharpness + 1e-10))
        
        motion_score = 0.6 * normalized_sharpness + 0.4 * ghost_score
        
        return min(motion_score, 1.0)
    
    def _calculate_spatial_smoothness(self, data: np.ndarray, fg_mask: np.ndarray) -> float:
        """Calculate spatial smoothness using local variance."""
        # Calculate local variance using sliding window
        structure = np.ones((3, 3, 3))
        local_mean = ndimage.uniform_filter(data, size=3)
        local_variance = ndimage.uniform_filter(data**2, size=3) - local_mean**2
        
        # Focus on foreground
        fg_variance = local_variance[fg_mask]
        
        if len(fg_variance) > 0:
            smoothness = 1.0 / (1.0 + np.mean(fg_variance))
        else:
            smoothness = 0.0
        
        return smoothness
    
    def _calculate_texture_metrics(self, data: np.ndarray, fg_mask: np.ndarray) -> Dict[str, float]:
        """Calculate texture metrics (simplified GLCM)."""
        metrics = {}
        
        fg_data = data[fg_mask]
        
        if len(fg_data) == 0:
            return metrics
        
        # Quantize intensities for texture analysis
        quantized = np.digitize(fg_data, bins=np.linspace(np.min(fg_data), np.max(fg_data), 16))
        
        # Calculate basic texture measures
        metrics['texture_variance'] = float(np.var(quantized))
        metrics['texture_entropy'] = float(stats.entropy(np.bincount(quantized)))
        
        return metrics
    
    def _calculate_gradient_stats(self, data: np.ndarray, fg_mask: np.ndarray) -> Dict[str, float]:
        """Calculate gradient-based spatial statistics."""
        metrics = {}
        
        # Calculate gradients
        gradients = np.gradient(data)
        gradient_magnitude = np.sqrt(sum(g**2 for g in gradients))
        
        # Focus on foreground
        fg_gradients = gradient_magnitude[fg_mask]
        
        if len(fg_gradients) > 0:
            metrics['gradient_mean'] = float(np.mean(fg_gradients))
            metrics['gradient_std'] = float(np.std(fg_gradients))
            metrics['gradient_p95'] = float(np.percentile(fg_gradients, 95))
        
        return metrics
    
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
    
    def classify_quality(self, qc_metrics: Dict[str, float]) -> Dict[str, str]:
        """Classify image quality based on QC metrics."""
        classifications = {}
        
        # SNR classification
        snr = qc_metrics.get('snr_basic', 0)
        if snr > 15:
            classifications['snr_quality'] = 'excellent'
        elif snr > 10:
            classifications['snr_quality'] = 'good'
        elif snr > 5:
            classifications['snr_quality'] = 'fair'
        else:
            classifications['snr_quality'] = 'poor'
        
        # Motion classification
        motion_score = qc_metrics.get('motion_score', 0)
        if motion_score < 0.2:
            classifications['motion_quality'] = 'excellent'
        elif motion_score < 0.4:
            classifications['motion_quality'] = 'good'
        elif motion_score < 0.6:
            classifications['motion_quality'] = 'fair'
        else:
            classifications['motion_quality'] = 'poor'
        
        # Coverage classification
        coverage = qc_metrics.get('coverage_ratio', 0)
        if coverage > 0.15:
            classifications['coverage_quality'] = 'excellent'
        elif coverage > 0.10:
            classifications['coverage_quality'] = 'good'
        elif coverage > 0.05:
            classifications['coverage_quality'] = 'fair'
        else:
            classifications['coverage_quality'] = 'poor'
        
        # Overall quality (simple voting)
        quality_scores = [classifications.get(k, 'poor') for k in ['snr_quality', 'motion_quality', 'coverage_quality']]
        quality_counts = {q: quality_scores.count(q) for q in ['excellent', 'good', 'fair', 'poor']}
        overall_quality = max(quality_counts, key=quality_counts.get)
        classifications['overall_quality'] = overall_quality
        
        return classifications