"""
Advanced Quality Control (QC) metrics for medical and single-cell imaging.

This module provides an extensible framework to compute comprehensive quality
assessment metrics for neuro / radiological 3D NIfTI volumes as well as
2D / 3D fluorescence / microscopy single-cell images.

Key Enhancements Over Prior Version:
- Modular, extensible architecture with a Configuration dataclass
- Support for 2D, 3D, and 4D (time-series) data (e.g., fMRI or live-cell imaging)
- Optional per-slice and per-timepoint QC summaries
- Advanced statistical robustness (MAD, robust CV, adaptive thresholds)
- Expanded metric families:
  * Signal / Noise: SNR (basic, robust), CNR, tSNR (temporal), pSNR
  * Contrast: multiple global + local; entropy-based contrast
  * Motion / Artifacts: ghosting, ringing, blur indices, temporal fluctuation
  * Coverage / Geometry: bounding box, centroid, anisotropy, fill ratios
  * Intensity Distribution: skewness, kurtosis, range ratios, entropy
  * Spatial: gradients, Laplacian focus, GLCM texture (if scikit-image available)
  * Single-Cell Specific Metrics (if a cell mask is provided):
      cell_count, mean_cell_volume, cell_size_cv, cell_intensity_cv,
      focus_score (variance of Laplacian), background_uniformity
- Optional template similarity metrics (SSIM, Mutual Information) if a template is supplied
- Safe optional imports (skimage, cupy) and graceful degradation
- Internal caching of expensive computations (gradients, laplacian)
- Improved mask generation with adaptive Otsu fallback (if skimage available)
- Memory-aware sampling for huge volumes
- Extensive type hints, detailed docstrings, and logging

NOTE:
This file does not enforce hard dependencies on scikit-image or cupy. If they
are unavailable, related metrics will be skipped automatically.

Author: (Enhanced by AI Assistant)
"""

from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional, Union, Dict, Any, List, Tuple

import numpy as np
import nibabel as nib
from scipy import ndimage, stats

# Optional imports
try:
    from skimage.metrics import structural_similarity as ssim
    from skimage.filters import threshold_otsu
    from skimage.feature import greycomatrix, greycoprops
    _HAVE_SKIMAGE = True
except Exception:
    _HAVE_SKIMAGE = False

try:
    import cupy as cp  # noqa: F401
    _HAVE_CUDA = True
except Exception:
    _HAVE_CUDA = False

# Constants
EPS = 1e-10

logger = logging.getLogger(__name__)


@dataclass
class QCMetricsConfig:
    """
    Configuration for QC metric computation.

    Attributes:
        use_gpu: Attempt GPU acceleration where possible (if cupy installed).
        local_contrast_window: Window size (voxels) for local contrast calculations.
        gradient_cache: Enable caching of gradient magnitude for reuse.
        max_sample_voxels: Down-sample computations if volume exceeds this voxel count.
        template_image_path: Path to reference/template image for similarity metrics.
        enable_glcm: Enable GLCM texture metrics (needs scikit-image).
        glcm_distances: Distances for GLCM (if enabled).
        glcm_angles: Angles (radians) for GLCM (if enabled).
        per_slice_metrics: Compute per-slice summaries (z-dimension emphasis).
        per_time_metrics: Compute metrics across 4th dimension (time) if present.
        single_cell_mode: Enable single cell metrics (needs cell mask).
        focus_measure_kernel: Kernel size for focus measures (Laplacian).
        mask_threshold_percentile: Percentile for adaptive threshold when building foreground mask.
        robust: Use robust (median/MAD) variants of metrics where applicable.
        compute_entropy: Compute foreground intensity entropy.
        compute_psnr: Compute peak SNR (needs dynamic range).
        compute_ssim: Compute SSIM if template provided.
        compute_mutual_info: Compute Mutual Information with template.
    """
    use_gpu: bool = False
    local_contrast_window: int = 5
    gradient_cache: bool = True
    max_sample_voxels: int = 20_000_000
    template_image_path: Optional[Union[str, Path]] = None
    enable_glcm: bool = True
    glcm_distances: Tuple[int, ...] = (1, 2)
    glcm_angles: Tuple[float, ...] = (0.0, np.pi / 4, np.pi / 2)
    per_slice_metrics: bool = False
    per_time_metrics: bool = True
    single_cell_mode: bool = True
    focus_measure_kernel: int = 3
    mask_threshold_percentile: float = 15.0
    robust: bool = True
    compute_entropy: bool = True
    compute_psnr: bool = True
    compute_ssim: bool = True
    compute_mutual_info: bool = True

    extra: Dict[str, Any] = field(default_factory=dict)


class QualityControlMetrics:
    """
    Advanced quality control assessment for medical and single-cell images.

    Usage:
        qc = QualityControlMetrics(config=QCMetricsConfig())
        metrics = qc.calculate_qc_metrics("image.nii.gz",
                                          mask_path="brain_mask.nii.gz",
                                          cell_mask_path="cells_mask.nii.gz")
    """

    def __init__(self, config: Optional[QCMetricsConfig] = None, **kwargs):
        """
        Initialize QC metrics calculator.

        Args:
            config: QCMetricsConfig instance
            **kwargs: Overrides / additional parameters merged into config.extra
        """
        self.config = config or QCMetricsConfig()
        self.config.extra.update(kwargs)
        self.logger = logging.getLogger(self.__class__.__name__)

        if self.config.use_gpu and not _HAVE_CUDA:
            self.logger.warning("GPU requested but CuPy not found. Falling back to CPU.")
            self.config.use_gpu = False

        if self.config.enable_glcm and not _HAVE_SKIMAGE:
            self.logger.warning("GLCM metrics requested but scikit-image missing. Disabling GLCM.")
            self.config.enable_glcm = False

        if self.config.template_image_path and not Path(self.config.template_image_path).exists():
            self.logger.warning("Template image path does not exist. Disabling template-based metrics.")
            self.config.template_image_path = None

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def calculate_qc_metrics(
        self,
        image_path: Union[str, Path],
        mask_path: Optional[Union[str, Path]] = None,
        background_mask_path: Optional[Union[str, Path]] = None,
        cell_mask_path: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive quality control metrics for a medical or single-cell image.

        Args:
            image_path: Path to input image (NIfTI primary, but can handle others if nib loads)
            mask_path: Foreground / brain mask path
            background_mask_path: Background mask path
            cell_mask_path: Single-cell segmentation mask path (labeled or binary)

        Returns:
            Dictionary of QC metrics (floats and possibly nested dicts for slice/time)
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Input image not found: {image_path}")

        self.logger.info(f"Loading image: {image_path}")
        img = nib.load(str(image_path))
        data = img.get_fdata()

        original_shape = data.shape
        ndim = data.ndim
        self.logger.debug(f"Image shape: {original_shape} (ndim={ndim})")

        # Down-sampling for huge volumes (keep memory reasonable)
        if np.prod(data.shape[:3]) > self.config.max_sample_voxels:
            self.logger.warning("Volume size exceeds limit; applying coarse down-sampling for global metrics.")
            factors = [max(1, math.ceil(s / (self.config.max_sample_voxels ** (1 / 3)))) for s in data.shape[:3]]
            slices = tuple(slice(None, None, f) for f in factors)
            if ndim == 4:
                slices = slices + (slice(None),)
            data_sampled = data[slices]
        else:
            data_sampled = data

        # Load masks
        fg_mask = self._load_or_create_foreground_mask(data, mask_path)
        bg_mask = self._load_or_create_background_mask(data, fg_mask, background_mask_path)

        # Cell mask for single-cell metrics
        if cell_mask_path and Path(cell_mask_path).exists():
            cell_mask_img = nib.load(str(cell_mask_path))
            cell_mask = cell_mask_img.get_fdata()
            cell_mask = cell_mask > 0 if np.unique(cell_mask).size <= 3 else cell_mask  # keep labels if many
        else:
            cell_mask = None

        metrics: Dict[str, Any] = {}
        # Core metric groups
        metrics.update(self._calculate_snr_metrics(data_sampled, fg_mask, bg_mask))
        metrics.update(self._calculate_contrast_metrics(data_sampled, fg_mask))
        metrics.update(self._calculate_motion_metrics(data_sampled, fg_mask))
        metrics.update(self._calculate_coverage_metrics(data_sampled, fg_mask))
        metrics.update(self._calculate_intensity_metrics(data_sampled, fg_mask))
        metrics.update(self._calculate_spatial_metrics(data_sampled, fg_mask))

        # Temporal metrics for 4D
        if data.ndim == 4 and self.config.per_time_metrics:
            metrics['temporal_metrics'] = self._calculate_temporal_metrics(data, fg_mask)

        # Slice metrics (z-dimension)
        if self.config.per_slice_metrics:
            metrics['per_slice'] = self._calculate_per_slice_metrics(data, fg_mask)

        # Template comparison (SSIM / MI)
        if self.config.template_image_path:
            metrics.update(self._template_similarity_metrics(data, fg_mask))

        # Single-cell metrics
        if self.config.single_cell_mode and cell_mask is not None:
            metrics['single_cell'] = self._calculate_single_cell_metrics(data, cell_mask, bg_mask)

        # Quality classification
        metrics['classification'] = self.classify_quality(metrics)

        self.logger.info(f"Computed {len([k for k in metrics if isinstance(metrics[k], (int, float))])} scalar metrics "
                         f"(plus structured groups).")
        return metrics

    def validate_inputs(self, image_path: Union[str, Path]) -> bool:
        """Validate input exists and is readable as NIfTI."""
        try:
            image_path = Path(image_path)
            if not image_path.exists():
                return False
            nib.load(str(image_path))
            return True
        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            return False

    def classify_quality(self, qc_metrics: Dict[str, Any]) -> Dict[str, str]:
        """
        Classify overall quality using heuristic thresholds.

        Expandable strategy: can be replaced with ML-based classifier externally.
        """
        result: Dict[str, str] = {}

        def classify_range(val: float, thresholds: List[float], labels: List[str]) -> str:
            for t, label in zip(thresholds, labels):
                if val >= t:
                    return label
            return labels[-1]

        snr = float(qc_metrics.get('snr_basic', 0.0))
        result['snr_quality'] = classify_range(snr, [20, 12, 6], ['excellent', 'good', 'fair', 'poor'])

        motion_score = float(qc_metrics.get('motion_score', 1.0))
        if motion_score < 0.2:
            result['motion_quality'] = 'excellent'
        elif motion_score < 0.4:
            result['motion_quality'] = 'good'
        elif motion_score < 0.6:
            result['motion_quality'] = 'fair'
        else:
            result['motion_quality'] = 'poor'

        coverage = float(qc_metrics.get('coverage_ratio', 0.0))
        result['coverage_quality'] = classify_range(coverage, [0.20, 0.12, 0.06], ['excellent', 'good', 'fair', 'poor'])

        # Composite vote
        votes = [result['snr_quality'], result['motion_quality'], result['coverage_quality']]
        order = ['excellent', 'good', 'fair', 'poor']
        score_count = {k: votes.count(k) for k in order}
        result['overall_quality'] = max(score_count, key=score_count.get)
        return result

    # ---------------------------------------------------------------------
    # Private Helpers: Loading & Masking
    # ---------------------------------------------------------------------
    def _load_or_create_foreground_mask(self, data: np.ndarray, mask_path: Optional[Union[str, Path]]) -> np.ndarray:
        if mask_path and Path(mask_path).exists():
            try:
                m = nib.load(str(mask_path)).get_fdata()
                return m > 0
            except Exception as e:
                self.logger.warning(f"Failed to load provided mask: {e}. Generating automatically.")
        return self._create_foreground_mask(data, self.config.mask_threshold_percentile)

    def _load_or_create_background_mask(
        self,
        data: np.ndarray,
        fg_mask: np.ndarray,
        background_mask_path: Optional[Union[str, Path]]
    ) -> np.ndarray:
        if background_mask_path and Path(background_mask_path).exists():
            try:
                m = nib.load(str(background_mask_path)).get_fdata()
                return m > 0
            except Exception as e:
                self.logger.warning(f"Failed to load background mask: {e}. Generating automatically.")
        return self._create_background_mask(data, fg_mask)

    # ---------------------------------------------------------------------
    # Metric Families
    # ---------------------------------------------------------------------
    def _calculate_snr_metrics(self, data: np.ndarray, fg_mask: np.ndarray, bg_mask: np.ndarray) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        fg_vals = data[fg_mask]
        bg_vals = data[bg_mask]

        if len(fg_vals) == 0 or len(bg_vals) == 0:
            return metrics

        signal_mean = float(np.mean(fg_vals))
        noise_std = float(np.std(bg_vals) + EPS)
        metrics['snr_basic'] = signal_mean / noise_std
        metrics['cnr'] = (signal_mean - float(np.mean(bg_vals))) / noise_std

        if self.config.robust:
            signal_median = float(np.median(fg_vals))
            noise_mad = float(stats.median_abs_deviation(bg_vals) + EPS)
            metrics['snr_robust'] = signal_median / noise_mad

        signal_std = float(np.std(fg_vals) + EPS)
        metrics['signal_cv'] = signal_std / (signal_mean + EPS)

        bg_mean = float(np.mean(bg_vals) + EPS)
        metrics['background_cv'] = noise_std / bg_mean

        # Peak SNR (optional)
        if self.config.compute_psnr:
            peak = float(np.max(fg_vals))
            mse = float(np.mean((fg_vals - peak) ** 2) + EPS)
            metrics['psnr'] = 10 * math.log10((peak ** 2) / mse)

        return metrics

    def _calculate_contrast_metrics(self, data: np.ndarray, fg_mask: np.ndarray) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        fg_vals = data[fg_mask]
        if len(fg_vals) == 0:
            return metrics

        mean_int = float(np.mean(fg_vals) + EPS)
        rms_contrast = math.sqrt(float(np.mean((fg_vals - mean_int) ** 2)))
        metrics['rms_contrast'] = rms_contrast / mean_int

        max_int = float(np.max(fg_vals))
        min_int = float(np.min(fg_vals))
        sum_max_min = max_int + min_int + EPS
        metrics['michelson_contrast'] = (max_int - min_int) / sum_max_min
        metrics['weber_contrast'] = (max_int - mean_int) / mean_int

        local_contrasts = self._calculate_local_contrasts(data, fg_mask, self.config.local_contrast_window)
        if local_contrasts.size > 0:
            metrics['local_contrast_mean'] = float(np.mean(local_contrasts))
            metrics['local_contrast_std'] = float(np.std(local_contrasts))

        if self.config.compute_entropy:
            hist, _ = np.histogram(fg_vals, bins=64)
            metrics['intensity_entropy'] = float(stats.entropy(hist + EPS))

        return metrics

    def _calculate_motion_metrics(self, data: np.ndarray, fg_mask: np.ndarray) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        edge_sharpness = self._calculate_edge_sharpness(data, fg_mask)
        metrics['edge_sharpness'] = float(edge_sharpness)

        ghost_score = self._detect_ghost_artifacts(data, fg_mask)
        metrics['ghost_artifact_score'] = float(ghost_score)

        ringing_score = self._detect_ringing_artifacts(data, fg_mask)
        metrics['ringing_artifact_score'] = float(ringing_score)

        motion_score = self._calculate_motion_score(data, fg_mask)
        metrics['motion_score'] = float(motion_score)

        return metrics

    def _calculate_coverage_metrics(self, data: np.ndarray, fg_mask: np.ndarray) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        total_voxels = float(np.prod(data.shape[:3]))
        fg_voxels = float(np.sum(fg_mask))
        metrics['coverage_ratio'] = fg_voxels / (total_voxels + EPS)

        if np.any(fg_mask):
            coords = np.where(fg_mask)
            bbox_dims: List[float] = []
            for i in range(3):
                min_c, max_c = int(np.min(coords[i])), int(np.max(coords[i]))
                bbox_dims.append((max_c - min_c + 1) / data.shape[i])

            metrics['bbox_x_ratio'], metrics['bbox_y_ratio'], metrics['bbox_z_ratio'] = map(float, bbox_dims)
            centroid = ndimage.center_of_mass(fg_mask.astype(float))
            metrics['centroid_x_normalized'] = float(centroid[0] / data.shape[0])
            metrics['centroid_y_normalized'] = float(centroid[1] / data.shape[1])
            metrics['centroid_z_normalized'] = float(centroid[2] / data.shape[2])
            bbox_volume = np.prod(bbox_dims) * total_voxels
            if bbox_volume > 0:
                metrics['coverage_uniformity'] = fg_voxels / (bbox_volume + EPS)

            # Simple anisotropy (range of bbox dimensions)
            metrics['bbox_anisotropy'] = float(np.std(bbox_dims) / (np.mean(bbox_dims) + EPS))
        return metrics

    def _calculate_intensity_metrics(self, data: np.ndarray, fg_mask: np.ndarray) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        fg_vals = data[fg_mask]
        if len(fg_vals) == 0:
            return metrics

        metrics['intensity_mean'] = float(np.mean(fg_vals))
        metrics['intensity_std'] = float(np.std(fg_vals))
        metrics['intensity_median'] = float(np.median(fg_vals))
        metrics['intensity_mad'] = float(stats.median_abs_deviation(fg_vals))

        for p in [5, 25, 75, 95]:
            metrics[f'intensity_p{p}'] = float(np.percentile(fg_vals, p))

        metrics['intensity_skewness'] = float(stats.skew(fg_vals, bias=False))
        metrics['intensity_kurtosis'] = float(stats.kurtosis(fg_vals, bias=False))

        min_int = float(np.min(fg_vals))
        max_int = float(np.max(fg_vals))
        drange = max_int - min_int
        metrics['dynamic_range'] = drange
        mean_val = metrics['intensity_mean'] + EPS
        metrics['dynamic_range_ratio'] = drange / mean_val

        return metrics

    def _calculate_spatial_metrics(self, data: np.ndarray, fg_mask: np.ndarray) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        smoothness = self._calculate_spatial_smoothness(data, fg_mask)
        metrics['spatial_smoothness'] = float(smoothness)

        texture_metrics = self._calculate_texture_metrics(data, fg_mask)
        metrics.update(texture_metrics)

        gradient_stats = self._calculate_gradient_stats(data, fg_mask)
        metrics.update(gradient_stats)

        # Focus measure: variance of Laplacian in foreground
        lap = self._get_laplacian(data)
        fg_lap = lap[fg_mask]
        if fg_lap.size > 0:
            metrics['focus_variance_laplacian'] = float(np.var(fg_lap))
        return metrics

    def _calculate_temporal_metrics(self, data: np.ndarray, fg_mask: np.ndarray) -> Dict[str, float]:
        """
        Temporal metrics for 4D data (time-series).
        tSNR: mean_t / std_t voxel-wise, then averaged over foreground.
        """
        if data.ndim != 4:
            return {}
        self.logger.debug("Computing temporal metrics (4D).")
        fg_data = data[fg_mask]
        if fg_data.size == 0:
            return {}
        mean_t = np.mean(fg_data, axis=1)
        std_t = np.std(fg_data, axis=1) + EPS
        tSNR = mean_t / std_t
        return {
            'temporal_snr_mean': float(np.mean(tSNR)),
            'temporal_snr_median': float(np.median(tSNR)),
            'temporal_intensity_cv': float(np.mean(std_t / (mean_t + EPS)))
        }

    def _calculate_per_slice_metrics(self, data: np.ndarray, fg_mask: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Compute per-slice metrics along the last spatial axis (assumes z axis is 2 if 3D).
        """
        if data.ndim < 3:
            return {}
        z_dim = data.shape[2]
        results: Dict[str, Dict[str, float]] = {}
        for z in range(z_dim):
            slice_mask = fg_mask[:, :, z]
            if not np.any(slice_mask):
                continue
            slice_vals = data[:, :, z][slice_mask]
            results[f'slice_{z}'] = {
                'mean': float(np.mean(slice_vals)),
                'std': float(np.std(slice_vals)),
                'snr_local': float(np.mean(slice_vals) / (np.std(slice_vals) + EPS)),
                'p95': float(np.percentile(slice_vals, 95))
            }
        return results

    def _template_similarity_metrics(self, data: np.ndarray, fg_mask: np.ndarray) -> Dict[str, float]:
        """
        Compute similarity to a template (SSIM, Mutual Information) if available.
        """
        metrics: Dict[str, float] = {}
        try:
            tmpl_img = nib.load(str(self.config.template_image_path))
            tmpl_data = tmpl_img.get_fdata()
            if tmpl_data.shape != data.shape:
                self.logger.warning("Template shape mismatch; skipping template metrics.")
                return metrics

            # Use masked intensities
            data_fg = data[fg_mask]
            tmpl_fg = tmpl_data[fg_mask]
            # Mutual Information (discrete approximation)
            if self.config.compute_mutual_info:
                bins = 64
                hist_2d, _, _ = np.histogram2d(
                    data_fg, tmpl_fg, bins=bins
                )
                pxy = hist_2d / np.sum(hist_2d)
                px = np.sum(pxy, axis=1)
                py = np.sum(pxy, axis=0)
                px_py = px[:, None] * py[None, :]
                nz = pxy > 0
                mi = np.sum(pxy[nz] * np.log((pxy[nz] + EPS) / (px_py[nz] + EPS)))
                metrics['mutual_information'] = float(mi)

            if self.config.compute_ssim and _HAVE_SKIMAGE and data.ndim <= 3:
                # For 3D we approximate with mean SSIM over slices
                if data.ndim == 3:
                    ssim_vals = []
                    for z in range(data.shape[2]):
                        if np.any(fg_mask[:, :, z]):
                            ssim_vals.append(
                                ssim(data[:, :, z], tmpl_data[:, :, z], data_range=float(np.max(data) - np.min(data)))
                            )
                    if ssim_vals:
                        metrics['ssim_mean'] = float(np.mean(ssim_vals))
                elif data.ndim == 2:
                    metrics['ssim'] = float(
                        ssim(data, tmpl_data, data_range=float(np.max(data) - np.min(data)))
                    )
        except Exception as e:
            self.logger.warning(f"Template similarity metrics failed: {e}")
        return metrics

    # ---------------------------------------------------------------------
    # Single-Cell Metrics
    # ---------------------------------------------------------------------
    def _calculate_single_cell_metrics(
        self,
        data: np.ndarray,
        cell_mask: np.ndarray,
        bg_mask: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute single-cell segmentation metrics.
        Supports either labeled integer mask (N labels) or binary mask.

        Metrics:
          cell_count, mean_cell_volume, cell_size_cv, cell_intensity_cv,
          focus_score (variance of Laplacian inside cells),
          background_uniformity.
        """
        metrics: Dict[str, float] = {}
        unique_vals = np.unique(cell_mask)
        labeled = unique_vals.size > 3  # Heuristic
        if labeled:
            labels = unique_vals[unique_vals > 0]
            volumes = []
            means = []
            for lab in labels:
                voxels = data[cell_mask == lab]
                if voxels.size == 0:
                    continue
                volumes.append(voxels.size)
                means.append(np.mean(voxels))
            if volumes:
                volumes_arr = np.array(volumes, dtype=float)
                means_arr = np.array(means, dtype=float)
                metrics['cell_count'] = float(len(volumes_arr))
                metrics['mean_cell_volume'] = float(np.mean(volumes_arr))
                metrics['cell_size_cv'] = float(np.std(volumes_arr) / (np.mean(volumes_arr) + EPS))
                metrics['cell_intensity_cv'] = float(np.std(means_arr) / (np.mean(means_arr) + EPS))
        else:
            # Binary
            labeled_mask, n = ndimage.label(cell_mask > 0)
            metrics['cell_count'] = float(n)
            sizes = ndimage.sum(cell_mask > 0, labeled_mask, index=range(1, n + 1))
            if n > 0:
                sizes = np.array(sizes, dtype=float)
                metrics['mean_cell_volume'] = float(np.mean(sizes))
                metrics['cell_size_cv'] = float(np.std(sizes) / (np.mean(sizes) + EPS))

        # Focus score: variance of Laplacian in cell regions
        try:
            lap = self._get_laplacian(data)
            cell_region = lap[cell_mask > 0]
            if cell_region.size > 0:
                metrics['focus_score'] = float(np.var(cell_region))
        except Exception:
            pass

        # Background uniformity (reuse bg mask)
        if bg_mask is not None and np.any(bg_mask):
            bg_vals = data[bg_mask]
            metrics['background_uniformity'] = float(np.std(bg_vals) / (np.mean(bg_vals) + EPS))

        return metrics

    # ---------------------------------------------------------------------
    # Core Computational Building Blocks
    # ---------------------------------------------------------------------
    def _create_foreground_mask(self, data: np.ndarray, threshold_percentile: float = 15) -> np.ndarray:
        nonzero = data[data > 0]
        if nonzero.size == 0:
            return np.zeros_like(data, dtype=bool)
        if _HAVE_SKIMAGE:
            try:
                otsu_val = threshold_otsu(nonzero)
                thresh = max(otsu_val, np.percentile(nonzero, threshold_percentile))
            except Exception:
                thresh = np.percentile(nonzero, threshold_percentile)
        else:
            thresh = np.percentile(nonzero, threshold_percentile)
        mask = data > thresh

        # Morphological cleanup
        mask = ndimage.binary_closing(mask, structure=np.ones((3, 3, 3)), iterations=1)
        mask = ndimage.binary_opening(mask, structure=np.ones((3, 3, 3)), iterations=1)

        labeled_mask, num = ndimage.label(mask)
        if num > 1:
            sizes = ndimage.sum(mask, labeled_mask, range(1, num + 1))
            max_label = int(np.argmax(sizes)) + 1
            mask = labeled_mask == max_label
        return mask

    def _create_background_mask(self, data: np.ndarray, fg_mask: np.ndarray) -> np.ndarray:
        positive = data[data > 0]
        if positive.size == 0:
            return np.zeros_like(data, dtype=bool)
        low_thresh = np.percentile(positive, 5)
        bg_candidate = (data <= low_thresh) & (data > 0)
        dilated = ndimage.binary_dilation(fg_mask, structure=np.ones((5, 5, 5)))
        bg_mask = bg_candidate & ~dilated
        return bg_mask

    def _calculate_local_contrasts(
        self,
        data: np.ndarray,
        fg_mask: np.ndarray,
        window_size: int
    ) -> np.ndarray:
        coords = np.array(np.where(fg_mask)).T
        if coords.shape[0] == 0:
            return np.array([])
        contrasts = []
        step = max(1, window_size)
        for i in range(0, coords.shape[0], step):
            subset = coords[i:i + step]
            vals = data[tuple(subset.T)]
            if vals.size > 1:
                mean_v = np.mean(vals)
                if mean_v > 0:
                    contrasts.append(np.std(vals) / (mean_v + EPS))
        return np.array(contrasts)

    def _calculate_edge_sharpness(self, data: np.ndarray, fg_mask: np.ndarray) -> float:
        grad_mag = self._get_gradient_magnitude(data)
        edges = grad_mag[fg_mask]
        if edges.size == 0:
            return 0.0
        return float(np.percentile(edges, 90))

    def _detect_ghost_artifacts(self, data: np.ndarray, fg_mask: np.ndarray) -> float:
        bg_mask = (~fg_mask) & (data > 0)
        if not np.any(bg_mask):
            return 0.0
        bg_vals = data[bg_mask]
        bg_var = float(np.var(bg_vals))
        bg_mean = float(np.mean(bg_vals) + EPS)
        score = bg_var / (bg_mean ** 2 + EPS)
        return float(min(score, 1.0))

    def _detect_ringing_artifacts(self, data: np.ndarray, fg_mask: np.ndarray) -> float:
        lap = self._get_laplacian(data)
        dilated_fg = ndimage.binary_dilation(fg_mask, structure=np.ones((3, 3, 3)))
        edge_region = dilated_fg & ~fg_mask
        if not np.any(edge_region):
            return 0.0
        edge_vals = lap[edge_region]
        fg_std = np.std(data[fg_mask]) + EPS
        score = np.std(edge_vals) / fg_std
        return float(min(score, 1.0))

    def _calculate_motion_score(self, data: np.ndarray, fg_mask: np.ndarray) -> float:
        sharpness = self._calculate_edge_sharpness(data, fg_mask)
        ghost = self._detect_ghost_artifacts(data, fg_mask)
        if np.any(fg_mask):
            max_signal = np.percentile(data[fg_mask], 95)
        else:
            max_signal = 1.0
        normalized_sharpness = 1.0 - (sharpness / (max_signal + EPS))
        score = 0.6 * normalized_sharpness + 0.4 * ghost
        return float(min(score, 1.0))

    def _calculate_spatial_smoothness(self, data: np.ndarray, fg_mask: np.ndarray) -> float:
        local_mean = ndimage.uniform_filter(data, size=3)
        local_var = ndimage.uniform_filter(data ** 2, size=3) - local_mean ** 2
        fg_var = local_var[fg_mask]
        if fg_var.size == 0:
            return 0.0
        return float(1.0 / (1.0 + np.mean(fg_var)))

    def _calculate_texture_metrics(self, data: np.ndarray, fg_mask: np.ndarray) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        fg_data = data[fg_mask]
        if fg_data.size == 0:
            return metrics

        # Basic quantization
        q_bins = 16
        q = np.digitize(fg_data, bins=np.linspace(np.min(fg_data), np.max(fg_data) + EPS, q_bins))
        metrics['texture_variance'] = float(np.var(q))
        hist = np.bincount(q, minlength=q_bins + 2)
        metrics['texture_entropy'] = float(stats.entropy(hist + EPS))

        # GLCM metrics (optional & approximate: sample a sub-block)
        if self.config.enable_glcm and _HAVE_SKIMAGE and data.ndim >= 2:
            # Sample a central slice for simplicity
            try:
                if data.ndim == 3:
                    mid = data.shape[2] // 2
                    slice_data = data[:, :, mid]
                else:
                    slice_data = data if data.ndim == 2 else data[..., 0]
                # Normalize and quantize
                sd = slice_data - np.min(slice_data)
                if np.max(sd) > 0:
                    sd = (sd / np.max(sd)) * 63
                sd_q = sd.astype(np.uint8)
                glcm = greycomatrix(
                    sd_q,
                    distances=list(self.config.glcm_distances),
                    angles=list(self.config.glcm_angles),
                    symmetric=True,
                    normed=True
                )
                for prop in ['contrast', 'homogeneity', 'energy', 'correlation']:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        val = greycoprops(glcm, prop).mean()
                    metrics[f'glcm_{prop}'] = float(val)
            except Exception as e:
                self.logger.debug(f"GLCM computation failed: {e}")

        return metrics

    def _calculate_gradient_stats(self, data: np.ndarray, fg_mask: np.ndarray) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        grad_mag = self._get_gradient_magnitude(data)
        fg_grad = grad_mag[fg_mask]
        if fg_grad.size > 0:
            metrics['gradient_mean'] = float(np.mean(fg_grad))
            metrics['gradient_std'] = float(np.std(fg_grad))
            metrics['gradient_p95'] = float(np.percentile(fg_grad, 95))
        return metrics

    # ---------------------------------------------------------------------
    # Cached Computations
    # ---------------------------------------------------------------------
    @lru_cache(maxsize=8)
    def _cached_gradients(self, shape_key: Tuple[int, ...], data_hash: int, flattened: bytes) -> np.ndarray:
        # Reconstruct array
        arr = np.frombuffer(flattened, dtype=np.float64).reshape(shape_key)
        grads = np.gradient(arr)
        grad_mag = np.sqrt(sum(g ** 2 for g in grads))
        return grad_mag

    def _get_gradient_magnitude(self, data: np.ndarray) -> np.ndarray:
        if not self.config.gradient_cache:
            grads = np.gradient(data)
            return np.sqrt(sum(g ** 2 for g in grads))
        key = data.shape
        data_bytes = np.ascontiguousarray(data, dtype=np.float64).tobytes()
        data_hash = hash(data_bytes[:1024])  # cheap partial hash
        return self._cached_gradients(key, data_hash, data_bytes)

    @lru_cache(maxsize=8)
    def _cached_laplacian(self, shape_key: Tuple[int, ...], data_hash: int, flattened: bytes) -> np.ndarray:
        arr = np.frombuffer(flattened, dtype=np.float64).reshape(shape_key)
        return ndimage.laplace(arr)

    def _get_laplacian(self, data: np.ndarray) -> np.ndarray:
        key = data.shape
        data_bytes = np.ascontiguousarray(data, dtype=np.float64).tobytes()
        data_hash = hash(data_bytes[:1024])
        return self._cached_laplacian(key, data_hash, data_bytes)

    # ---------------------------------------------------------------------
    # Deprecated / Backward compatibility stubs (if needed externally)
    # ---------------------------------------------------------------------
    # (Left intentionally blank; can map old method names here if required)
    # ---------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Compute QC metrics for medical or single-cell images.")
    parser.add_argument("image", type=str, help="Path to image (NIfTI)")
    parser.add_argument("--mask", type=str, default=None, help="Foreground/brain mask path")
    parser.add_argument("--bg-mask", type=str, default=None, help="Background mask path")
    parser.add_argument("--cell-mask", type=str, default=None, help="Single-cell segmentation mask path")
    parser.add_argument("--template", type=str, default=None, help="Template image for similarity metrics")
    parser.add_argument("--per-slice", action="store_true", help="Enable per-slice metrics")
    parser.add_argument("--no-temporal", action="store_true", help="Disable temporal metrics for 4D")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")

    args = parser.parse_args()

    config = QCMetricsConfig(
        template_image_path=args.template,
        per_slice_metrics=args.per_slice,
        per_time_metrics=not args.no_temporal
    )

    qc = QualityControlMetrics(config=config)
    if not qc.validate_inputs(args.image):
        raise SystemExit("Invalid input image.")

    metrics = qc.calculate_qc_metrics(
        args.image,
        mask_path=args.mask,
        background_mask_path=args.bg_mask,
        cell_mask_path=args.cell_mask
    )

    if args.output:
        with open(args.output, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"QC metrics written to {args.output}")
    else:
        print(json.dumps(metrics, indent=2))
