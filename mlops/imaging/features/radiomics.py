"""
Advanced Radiomics Feature Extraction (Lightweight, Dependency-Tolerant)

This module provides a pragmatic, dependency-light alternative to full-featured
radiomics libraries (e.g., pyradiomics) for scenarios where installation,
portability, or environment constraints are limiting.

Key capabilities:
- First-order intensity statistics (robust + classical)
- Shape / morphology metrics (volume, surface, sphericity, elongation, axes, compactness variants)
- Texture features:
  * Simplified GLCM (with optional scikit-image acceleration if available)
  * Local Binary Pattern (LBP) slice-based approximation
  * Gradient-based texture
  * Simple gray level run-length statistics (GLRLM-lite)
- Fractal dimension (box-counting) estimation (coarse)
- Configurable masking with morphological cleanup
- Optional intensity normalization
- Graceful degradation if optional dependencies are unavailable

Intended Use:
- Research prototyping
- Lightweight pipelines
- MLOps compliance where deterministic & minimal external deps matter

Not Intended For:
- Clinical-grade radiomics (use validated libraries instead)
- Regulatory submissions without validation

Author: Auto-generated enhanced version
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from collections import OrderedDict
from typing import Optional, Union, Dict, Any, Iterable, Tuple, List

import numpy as np
import nibabel as nib
from scipy import ndimage

try:
    # Optional imports
    from skimage.feature import greycomatrix, greycoprops  # type: ignore
    _HAS_SKIMAGE = True
except Exception:
    _HAS_SKIMAGE = False

try:
    from sklearn.feature_extraction.image import extract_patches_2d  # type: ignore
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False


__all__ = [
    "RadiomicsConfig",
    "RadiomicsExtractor"
]

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------------------
# Configuration Dataclass
# --------------------------------------------------------------------------------------
@dataclass
class RadiomicsConfig:
    enabled: bool = True
    normalize: bool = False
    normalization_strategy: str = "zscore"  # zscore | minmax | robust
    histogram_bins: int = 50
    gray_levels: int = 32  # For GLCM quantization
    glcm_offsets_3d: Optional[List[Tuple[int, int, int]]] = field(default_factory=lambda: [
        (1, 0, 0), (0, 1, 0), (0, 0, 1),
        (1, 1, 0), (1, 0, 1), (0, 1, 1),
        (1, -1, 0), (1, 0, -1), (0, 1, -1)
    ])
    glcm_symmetric: bool = True
    glcm_levels_cap: int = 64  # Prevent explosion
    lbp_use: bool = True
    lbp_max_patches: int = 300
    lbp_slice_strategy: str = "largest_mask"  # largest_mask | middle | all_average
    mask_threshold_percentile: float = 25.0
    keep_largest_component: bool = True
    min_component_size: int = 50
    morphological_closing_iters: int = 1
    morphological_opening_iters: int = 1
    compute_fractal: bool = True
    fractal_min_box: int = 2
    fractal_levels: int = 5
    compute_run_length: bool = True
    gradient_sample_cap: int = 250_000
    roi_voxel_cap: int = 2_000_000  # Safety
    cache_images: bool = True
    time_features: bool = True
    fail_silently: bool = True  # If False, propagate exceptions
    random_state: Optional[int] = 42

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------------------
# Utility Functions
# --------------------------------------------------------------------------------------
def _safe_log2(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.log2(np.clip(x, eps, None))


def _coerce_path(p: Union[str, Path]) -> Path:
    return p if isinstance(p, Path) else Path(p)


def _quantize_intensities(arr: np.ndarray, n_levels: int) -> np.ndarray:
    if n_levels < 2:
        return np.zeros_like(arr, dtype=np.uint8)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr, dtype=np.uint8)
    lo, hi = np.percentile(finite, [1, 99])
    if hi <= lo:
        hi = finite.max()
        lo = finite.min()
    if hi == lo:
        return np.zeros_like(arr, dtype=np.uint8)
    scaled = (arr - lo) / (hi - lo)
    scaled = np.clip(scaled, 0, 1)
    return (scaled * (n_levels - 1)).astype(np.uint8)


def _box_count_fractal(mask: np.ndarray,
                       min_box: int = 2,
                       levels: int = 5) -> Optional[float]:
    if not np.any(mask):
        return None
    sizes = []
    counts = []
    max_dim = max(mask.shape)
    for i in range(levels):
        box_size = min_box * (2 ** i)
        if box_size > max_dim:
            break
        # Count occupied boxes
        steps = [range(0, s, box_size) for s in mask.shape]
        occupied = 0
        for x in steps[0]:
            for y in steps[1]:
                for z in steps[2]:
                    sub = mask[x:x + box_size, y:y + box_size, z:z + box_size]
                    if np.any(sub):
                        occupied += 1
        if occupied > 0:
            sizes.append(1.0 / box_size)
            counts.append(occupied)
    if len(sizes) < 2:
        return None
    log_sizes = np.log(sizes)
    log_counts = np.log(counts)
    # Linear fit slope => fractal dimension estimate
    slope, _ = np.polyfit(log_sizes, log_counts, 1)
    return float(slope)


# --------------------------------------------------------------------------------------
# Main Extractor Class
# --------------------------------------------------------------------------------------
class RadiomicsExtractor:
    """
    Advanced lightweight radiomics feature extractor with configurable components.

    Usage:
        extractor = RadiomicsExtractor()
        features = extractor.extract_features("image.nii.gz", "mask.nii.gz")

    Notes:
        - All feature names are prefixed with radiomics_*
        - Designed for robustness over completeness
        - Fallbacks used when optional libraries are missing
    """

    def __init__(self, config: Optional[RadiomicsConfig] = None, **overrides):
        self.config = config or RadiomicsConfig()
        # Apply overrides
        for k, v in overrides.items():
            if hasattr(self.config, k):
                setattr(self.config, k, v)
        self.logger = logging.getLogger(self.__class__.__name__)
        self._image_cache: Dict[str, nib.Nifti1Image] = {}
        if self.config.random_state is not None:
            np.random.seed(self.config.random_state)

    # ----------------------------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------------------------
    def extract_features(
        self,
        image_path: Union[str, Path],
        mask_path: Optional[Union[str, Path]] = None,
        include_config_meta: bool = True
    ) -> Dict[str, float]:
        """
        Extract radiomics features from an image (and optional mask).

        Args:
            image_path: NIfTI image path
            mask_path: Optional mask (binary or integer labels; >0 treated as ROI)
            include_config_meta: Append meta metrics (counts, timings, config summary)

        Returns:
            Dictionary of feature_name -> value
        """
        if not self.config.enabled:
            self.logger.info("Radiomics extraction is disabled in configuration.")
            return {}

        start_time = time.time()
        image_path = _coerce_path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Input image not found: {image_path}")

        try:
            img = self._load_nifti(image_path)
            data = img.get_fdata().astype(np.float32, copy=False)

            # Mask
            mask = self._load_or_generate_mask(data, mask_path)

            if np.sum(mask) == 0:
                self.logger.warning("Mask is empty; returning empty feature set.")
                return {"radiomics_empty_mask": 1.0}

            # Optional normalization
            if self.config.normalize:
                data = self._normalize_intensity(data, mask)

            # Safety cap
            roi_voxels = int(np.sum(mask))
            if roi_voxels > self.config.roi_voxel_cap:
                self.logger.warning(
                    f"ROI voxel count {roi_voxels} exceeds cap {self.config.roi_voxel_cap}; "
                    "downsampling mask for texture-heavy features."
                )
                mask = self._subsample_mask(mask, target=self.config.roi_voxel_cap)

            voxel_spacing = img.header.get_zooms()[:3]

            # Feature assembly
            features: "OrderedDict[str, float]" = OrderedDict()

            # First-order
            features.update(self._first_order(data, mask))

            # Shape / morphology
            features.update(self._shape_features(mask, voxel_spacing))

            # Texture
            features.update(self._texture_features(data, mask))

            # Fractal
            if self.config.compute_fractal:
                fractal_dim = _box_count_fractal(
                    mask,
                    min_box=self.config.fractal_min_box,
                    levels=self.config.fractal_levels
                )
                if fractal_dim is not None:
                    features["radiomics_fractal_dimension"] = fractal_dim

            # Run-length (simplified)
            if self.config.compute_run_length:
                features.update(self._run_length_features(data, mask))

            # Gradient stats (some already in texture; augment)
            features.update(self._gradient_directionality(data, mask))

            # Meta info
            if include_config_meta:
                features["radiomics_feature_count"] = float(len(features))
                features["radiomics_roi_voxels"] = float(roi_voxels)
                if self.config.time_features:
                    features["radiomics_elapsed_sec"] = float(time.time() - start_time)
                # Config digest (small selection)
                features["radiomics_gray_levels_cfg"] = float(self.config.gray_levels)
                features["radiomics_histogram_bins_cfg"] = float(self.config.histogram_bins)
                features["radiomics_use_lbp_cfg"] = float(int(self.config.lbp_use))
                features["radiomics_has_skimage"] = float(int(_HAS_SKIMAGE))
                features["radiomics_has_sklearn"] = float(int(_HAS_SKLEARN))

            self.logger.info(f"Extracted {len(features)} radiomics features.")
            return dict(features)

        except Exception as e:
            self.logger.error(f"Radiomics extraction failed: {e}", exc_info=not self.config.fail_silently)
            if self.config.fail_silently:
                return {"radiomics_extraction_failed": 1.0}
            raise

    def validate_inputs(self, image_path: Union[str, Path]) -> bool:
        """Validate existence & readability of a NIfTI image."""
        try:
            image_path = _coerce_path(image_path)
            if not image_path.exists():
                return False
            nib.load(str(image_path))
            return True
        except Exception:
            return False

    def get_feature_categories(self, features: Dict[str, float]) -> Dict[str, int]:
        """Categorize features for coarse grouping."""
        categories = {
            'first_order': 0,
            'shape': 0,
            'glcm': 0,
            'lbp': 0,
            'gradient': 0,
            'run_length': 0,
            'fractal': 0,
            'other': 0
        }
        for k in features:
            if k.startswith("radiomics_"):
                if any(token in k for token in [
                    "mean", "std", "median", "min", "max", "percentile",
                    "entropy", "energy", "kurtosis", "skewness", "mad", "cv", "rms", "iqr"
                ]) and "glcm" not in k and "lbp" not in k and "gradient" not in k:
                    categories['first_order'] += 1
                elif any(token in k for token in [
                    "volume", "surface", "sphericity", "elongation",
                    "flatness", "compactness", "axis", "bbox", "svr"
                ]):
                    categories['shape'] += 1
                elif "glcm" in k:
                    categories['glcm'] += 1
                elif "lbp" in k:
                    categories['lbp'] += 1
                elif "gradient" in k:
                    categories['gradient'] += 1
                elif "run_length" in k or "glrlm" in k:
                    categories['run_length'] += 1
                elif "fractal" in k:
                    categories['fractal'] += 1
                else:
                    categories['other'] += 1
        return categories

    # ----------------------------------------------------------------------------------
    # Internal: Image / Mask Handling
    # ----------------------------------------------------------------------------------
    def _load_nifti(self, path: Path) -> nib.Nifti1Image:
        if self.config.cache_images and path.as_posix() in self._image_cache:
            return self._image_cache[path.as_posix()]
        img = nib.load(str(path))
        if self.config.cache_images:
            self._image_cache[path.as_posix()] = img
        return img

    def _load_or_generate_mask(self, data: np.ndarray, mask_path: Optional[Union[str, Path]]) -> np.ndarray:
        if mask_path:
            p = _coerce_path(mask_path)
            if p.exists():
                try:
                    m = nib.load(str(p)).get_fdata()
                    mask = m > 0
                    mask = self._postprocess_mask(mask)
                    return mask
                except Exception as e:
                    self.logger.warning(f"Failed to load provided mask: {e}; generating automatic mask.")
        return self._create_roi_mask(data)

    def _postprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        mask = mask.astype(bool)
        if self.config.keep_largest_component:
            labeled, n = ndimage.label(mask)
            if n > 1:
                sizes = ndimage.sum(mask, labeled, range(1, n + 1))
                main_label = np.argmax(sizes) + 1
                mask = labeled == main_label
        if self.config.min_component_size > 0:
            if np.sum(mask) < self.config.min_component_size:
                self.logger.warning("Mask below minimum component size threshold.")
        return mask

    def _create_roi_mask(self, data: np.ndarray) -> np.ndarray:
        positive = data[data > 0]
        if positive.size == 0:
            return np.zeros_like(data, dtype=bool)
        threshold = np.percentile(positive, self.config.mask_threshold_percentile)
        mask = data > threshold

        # Morphology
        if self.config.morphological_closing_iters > 0:
            mask = ndimage.binary_closing(
                mask,
                structure=np.ones((3, 3, 3)),
                iterations=self.config.morphological_closing_iters
            )
        if self.config.morphological_opening_iters > 0:
            mask = ndimage.binary_opening(
                mask,
                structure=np.ones((3, 3, 3)),
                iterations=self.config.morphological_opening_iters
            )

        # Largest component
        if self.config.keep_largest_component:
            labeled, n = ndimage.label(mask)
            if n > 1:
                sizes = ndimage.sum(mask, labeled, range(1, n + 1))
                max_label = np.argmax(sizes) + 1
                mask = labeled == max_label
        return mask

    def _subsample_mask(self, mask: np.ndarray, target: int) -> np.ndarray:
        coords = np.column_stack(np.where(mask))
        if coords.shape[0] <= target:
            return mask
        idx = np.random.choice(coords.shape[0], size=target, replace=False)
        sampled_coords = coords[idx]
        new_mask = np.zeros_like(mask, dtype=bool)
        new_mask[tuple(sampled_coords.T)] = True
        return new_mask

    # ----------------------------------------------------------------------------------
    # Internal: Intensity Normalization
    # ----------------------------------------------------------------------------------
    def _normalize_intensity(self, data: np.ndarray, mask: np.ndarray) -> np.ndarray:
        roi = data[mask]
        if roi.size == 0:
            return data
        strat = self.config.normalization_strategy.lower()
        if strat == "zscore":
            mean = float(np.mean(roi))
            std = float(np.std(roi)) or 1.0
            norm = (data - mean) / std
        elif strat == "minmax":
            lo, hi = np.min(roi), np.max(roi)
            if hi == lo:
                norm = np.zeros_like(data)
            else:
                norm = (data - lo) / (hi - lo)
        elif strat == "robust":
            q1, q3 = np.percentile(roi, [25, 75])
            iqr = q3 - q1
            scale = iqr if iqr > 1e-6 else 1.0
            norm = (data - q1) / scale
        else:
            self.logger.warning(f"Unknown normalization strategy: {strat}. Skipping.")
            return data
        return norm.astype(np.float32, copy=False)

    # ----------------------------------------------------------------------------------
    # First-Order Features
    # ----------------------------------------------------------------------------------
    def _first_order(self, data: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
        roi = data[mask]
        features: Dict[str, float] = {}
        if roi.size == 0:
            return features

        # Basic
        mean = np.mean(roi)
        std = np.std(roi)
        features["radiomics_mean"] = float(mean)
        features["radiomics_std"] = float(std)
        features["radiomics_median"] = float(np.median(roi))
        features["radiomics_min"] = float(np.min(roi))
        features["radiomics_max"] = float(np.max(roi))
        features["radiomics_range"] = float(np.max(roi) - np.min(roi))
        features["radiomics_variance"] = float(np.var(roi))

        # Percentiles
        for p in (1, 5, 10, 25, 50, 75, 90, 95, 99):
            features[f"radiomics_percentile_{p}"] = float(np.percentile(roi, p))
        features["radiomics_iqr"] = float(np.percentile(roi, 75) - np.percentile(roi, 25))

        if mean != 0:
            features["radiomics_cv"] = float(std / mean)

        # Skewness & kurtosis (Fisher)
        if std > 0:
            standardized = (roi - mean) / std
            features["radiomics_skewness"] = float(np.mean(standardized ** 3))
            features["radiomics_kurtosis"] = float(np.mean(standardized ** 4) - 3)

        # MAD (Median Absolute Deviation)
        med = features["radiomics_median"]
        features["radiomics_mad"] = float(np.median(np.abs(roi - med)))

        # Robust mean (trimmed)
        trimmed = np.sort(roi)
        trim_n = max(1, int(0.05 * trimmed.size))
        if trim_n * 2 < trimmed.size:
            robust_mean = np.mean(trimmed[trim_n:-trim_n])
            features["radiomics_trimmed_mean_5pct"] = float(robust_mean)

        # Energy, RMS, Entropy
        features["radiomics_energy"] = float(np.sum(roi ** 2))
        features["radiomics_rms"] = float(np.sqrt(np.mean(roi ** 2)))

        hist, _ = np.histogram(roi, bins=self.config.histogram_bins, density=True)
        hist = hist[hist > 0]
        if hist.size > 0:
            ent = -np.sum(hist * _safe_log2(hist))
            features["radiomics_entropy"] = float(ent)
            features["radiomics_uniformity"] = float(np.sum(hist ** 2))

        return features

    # ----------------------------------------------------------------------------------
    # Shape / Morphology Features
    # ----------------------------------------------------------------------------------
    def _shape_features(self, mask: np.ndarray, voxel_spacing: Tuple[float, float, float]) -> Dict[str, float]:
        features: Dict[str, float] = {}
        if not np.any(mask):
            return features

        voxel_vol = float(np.prod(voxel_spacing))
        volume_voxels = int(np.sum(mask))
        volume_mm3 = volume_voxels * voxel_vol

        features["radiomics_volume_voxels"] = float(volume_voxels)
        features["radiomics_volume_mm3"] = float(volume_mm3)

        # Surface area (approx via marching cubes gradient surrogate)
        surface_area = self._estimate_surface_area(mask, voxel_spacing)
        features["radiomics_surface_area_mm2"] = float(surface_area)
        if surface_area > 0:
            features["radiomics_surface_volume_ratio"] = float(surface_area / volume_mm3)
            # Classic sphericity
            sphericity = (math.pi ** (1 / 3.0)) * ((6.0 * volume_mm3) ** (2 / 3.0)) / surface_area
            features["radiomics_sphericity"] = float(sphericity)
            features["radiomics_compactness"] = float(1.0 / sphericity) if sphericity != 0 else 0.0
            features["radiomics_compactness2"] = float((36 * math.pi * volume_mm3 ** 2) / (surface_area ** 3))

        # Bounding box
        coords = np.array(np.where(mask)).T
        mins = coords.min(axis=0)
        maxs = coords.max(axis=0)
        dims_vox = (maxs - mins + 1).astype(float)
        dims_mm = dims_vox * np.array(voxel_spacing)
        features["radiomics_bbox_x_vox"] = float(dims_vox[0])
        features["radiomics_bbox_y_vox"] = float(dims_vox[1])
        features["radiomics_bbox_z_vox"] = float(dims_vox[2])
        features["radiomics_bbox_volume_vox"] = float(np.prod(dims_vox))
        features["radiomics_bbox_fill_ratio"] = float(volume_voxels / np.prod(dims_vox))
        features["radiomics_bbox_x_mm"] = float(dims_mm[0])
        features["radiomics_bbox_y_mm"] = float(dims_mm[1])
        features["radiomics_bbox_z_mm"] = float(dims_mm[2])

        # Principal axes (PCA)
        centered = coords - coords.mean(axis=0, keepdims=True)
        cov = np.cov(centered, rowvar=False)
        try:
            eigenvals, eigenvecs = np.linalg.eigh(cov)
            order = np.argsort(eigenvals)[::-1]
            eigenvals = eigenvals[order]
            # Axis lengths (2 * sqrt(lambda)) in voxel units
            axis_lengths = 2.0 * np.sqrt(np.maximum(eigenvals, 1e-12))
            # Convert to mm (approx)
            axis_lengths_mm = axis_lengths * np.array(voxel_spacing)
            features["radiomics_major_axis_length_mm"] = float(axis_lengths_mm[0])
            features["radiomics_minor_axis_length_mm"] = float(axis_lengths_mm[1] if len(axis_lengths_mm) > 1 else 0.0)
            features["radiomics_least_axis_length_mm"] = float(axis_lengths_mm[2] if len(axis_lengths_mm) > 2 else 0.0)
            if eigenvals[0] > 0:
                features["radiomics_elongation"] = float(eigenvals[1] / eigenvals[0]) if len(eigenvals) > 1 else 0.0
                features["radiomics_flatness"] = float(eigenvals[2] / eigenvals[0]) if len(eigenvals) > 2 else 0.0
        except np.linalg.LinAlgError:
            self.logger.warning("PCA eigen decomposition failed for shape features.")

        return features

    def _estimate_surface_area(self, mask: np.ndarray, voxel_spacing: Tuple[float, float, float]) -> float:
        gradient = np.gradient(mask.astype(float))
        grad_mag = np.sqrt(sum(g ** 2 for g in gradient))
        boundary = grad_mag > 0.1
        # Approximate each boundary voxel as rectangle with average face area
        mean_face = (voxel_spacing[0] * voxel_spacing[1] +
                     voxel_spacing[1] * voxel_spacing[2] +
                     voxel_spacing[0] * voxel_spacing[2]) / 3.0
        surface_area = float(np.sum(boundary) * mean_face)
        return surface_area

    # ----------------------------------------------------------------------------------
    # Texture Features
    # ----------------------------------------------------------------------------------
    def _texture_features(self, data: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
        feats: Dict[str, float] = {}
        roi = data[mask]
        if roi.size == 0:
            return feats

        # GLCM
        feats.update(self._glcm_features(data, mask))

        # LBP
        if self.config.lbp_use:
            feats.update(self._lbp_features(data, mask))

        # Gradient-based
        feats.update(self._gradient_features(data, mask))

        return feats

    def _glcm_features(self, data: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
        feats: Dict[str, float] = {}
        roi = data[mask]
        if roi.size < 10:
            return feats

        # Quantize
        gl_levels = min(self.config.gray_levels, self.config.glcm_levels_cap)
        q = _quantize_intensities(data, gl_levels)

        if _HAS_SKIMAGE:
            # Use 2D slices aggregated for simplicity
            slice_indices = [i for i in range(data.shape[2]) if np.any(mask[:, :, i])]
            # Limit to avoid huge memory usage
            slice_indices = slice_indices[:min(len(slice_indices), 15)]
            if not slice_indices:
                return feats
            # Standard angles for 2D
            distances = [1]
            angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
            glcm_accum = None
            total = 0
            for z in slice_indices:
                sl_mask = mask[:, :, z]
                if not np.any(sl_mask):
                    continue
                sl_quant = q[:, :, z]
                # Build mask bounding box to reduce matrix size
                coords = np.where(sl_mask)
                x0, x1 = coords[0].min(), coords[0].max()
                y0, y1 = coords[1].min(), coords[1].max()
                sub = sl_quant[x0:x1 + 1, y0:y1 + 1]
                glcm = greycomatrix(
                    sub,
                    distances=distances,
                    angles=angles,
                    levels=gl_levels,
                    symmetric=self.config.glcm_symmetric,
                    normed=True
                )
                if glcm_accum is None:
                    glcm_accum = glcm
                else:
                    glcm_accum += glcm
                total += 1
            if glcm_accum is None:
                return feats
            glcm_accum /= max(total, 1)

            # greycoprops
            for prop in ["contrast", "dissimilarity", "homogeneity", "ASM", "energy", "correlation"]:
                try:
                    val = greycoprops(glcm_accum, prop).mean()
                    feats[f"radiomics_glcm_{prop.lower()}"] = float(val)
                except Exception:
                    pass

            # Manual entropy
            p = glcm_accum.astype(np.float64)
            p_flat = p[p > 0]
            feats["radiomics_glcm_entropy"] = float(-np.sum(p_flat * _safe_log2(p_flat)))
        else:
            # Fallback simplified neighbor pairs (random sampling)
            coords = np.column_stack(np.where(mask))
            if coords.shape[0] > 50_000:
                idx = np.random.choice(coords.shape[0], size=50_000, replace=False)
                coords = coords[idx]

            values = q[mask]
            n_levels = len(np.unique(values))
            if n_levels < 2:
                return feats
            # Simple pair counting along random perm
            perm = np.random.permutation(len(values))
            pairs = np.column_stack([values[perm[:-1]], values[perm[1:]]])
            glcm = np.zeros((gl_levels, gl_levels), dtype=np.float64)
            for a, b in pairs:
                glcm[a, b] += 1
                if self.config.glcm_symmetric:
                    glcm[b, a] += 1
            if glcm.sum() > 0:
                glcm /= glcm.sum()
                feats["radiomics_glcm_energy"] = float(np.sum(glcm ** 2))
                feats["radiomics_glcm_contrast"] = float(
                    sum(((i - j) ** 2) * glcm[i, j] for i in range(gl_levels) for j in range(gl_levels))
                )
                feats["radiomics_glcm_homogeneity"] = float(
                    sum(glcm[i, j] / (1 + abs(i - j)) for i in range(gl_levels) for j in range(gl_levels))
                )
                feats["radiomics_glcm_entropy"] = float(-np.sum(glcm[glcm > 0] * _safe_log2(glcm[glcm > 0])))

        return feats

    def _lbp_features(self, data: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
        feats: Dict[str, float] = {}
        if not _HAS_SKLEARN:
            return feats

        # Strategy: Choose slice with largest mask or average across slices
        slice_counts = np.array([np.sum(mask[:, :, z]) for z in range(mask.shape[2])])
        valid_slices = np.where(slice_counts > 0)[0]
        if valid_slices.size == 0:
            return feats

        lbp_values_all: List[float] = []

        def process_slice(z: int):
            sl_mask = mask[:, :, z]
            sl_data = data[:, :, z]
            # Extract limited patches
            try:
                patches = extract_patches_2d(sl_data, (3, 3),
                                             max_patches=min(self.config.lbp_max_patches, sl_mask.sum()))
            except Exception:
                return []
            vals = []
            for patch in patches:
                c = patch[1, 1]
                neighbors = [
                    patch[0, 0], patch[0, 1], patch[0, 2], patch[1, 2],
                    patch[2, 2], patch[2, 1], patch[2, 0], patch[1, 0]
                ]
                pattern = 0
                for i, n in enumerate(neighbors):
                    if n >= c:
                        pattern |= (1 << i)
                vals.append(pattern)
            return vals

        if self.config.lbp_slice_strategy == "largest_mask":
            best_slice = int(np.argmax(slice_counts))
            lbp_values_all.extend(process_slice(best_slice))
        elif self.config.lbp_slice_strategy == "middle":
            mid = int(mask.shape[2] / 2)
            lbp_values_all.extend(process_slice(mid))
        else:  # all_average
            for z in valid_slices:
                lbp_values_all.extend(process_slice(int(z)))

        if not lbp_values_all:
            return feats

        lbp_vals = np.array(lbp_values_all)
        feats["radiomics_lbp_mean"] = float(np.mean(lbp_vals))
        feats["radiomics_lbp_std"] = float(np.std(lbp_vals))
        hist, _ = np.histogram(lbp_vals, bins=16, density=True)
        feats["radiomics_lbp_uniformity"] = float(np.sum(hist ** 2))
        nonzero = hist[hist > 0]
        if nonzero.size > 0:
            feats["radiomics_lbp_entropy"] = float(-np.sum(nonzero * _safe_log2(nonzero)))
        return feats

    def _gradient_features(self, data: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
        feats: Dict[str, float] = {}
        grads = np.gradient(data)
        grad_mag = np.sqrt(sum(g ** 2 for g in grads))
        roi = grad_mag[mask]
        if roi.size == 0:
            return feats
        feats["radiomics_gradient_mean"] = float(np.mean(roi))
        feats["radiomics_gradient_std"] = float(np.std(roi))
        feats["radiomics_gradient_max"] = float(np.max(roi))
        # Directional
        for axis, g in zip(["x", "y", "z"], grads):
            axis_vals = g[mask]
            feats[f"radiomics_gradient_{axis}_mean"] = float(np.mean(axis_vals))
            feats[f"radiomics_gradient_{axis}_std"] = float(np.std(axis_vals))
        return feats

    def _gradient_directionality(self, data: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
        feats: Dict[str, float] = {}
        grads = np.gradient(data)
        gx, gy, gz = grads
        gx_r = gx[mask]
        gy_r = gy[mask]
        gz_r = gz[mask]
        if gx_r.size == 0:
            return feats

        # Direction vector stats
        magnitude = np.sqrt(gx_r ** 2 + gy_r ** 2 + gz_r ** 2) + 1e-12
        ux, uy, uz = gx_r / magnitude, gy_r / magnitude, gz_r / magnitude
        feats["radiomics_gradient_dir_x_mean"] = float(np.mean(ux))
        feats["radiomics_gradient_dir_y_mean"] = float(np.mean(uy))
        feats["radiomics_gradient_dir_z_mean"] = float(np.mean(uz))
        feats["radiomics_gradient_dir_resultant_length"] = float(
            np.sqrt(np.mean(ux) ** 2 + np.mean(uy) ** 2 + np.mean(uz) ** 2)
        )
        return feats

    # ----------------------------------------------------------------------------------
    # Run-Length (Simplified)
    # ----------------------------------------------------------------------------------
    def _run_length_features(self, data: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
        feats: Dict[str, float] = {}
        roi = data[mask]
        if roi.size == 0:
            return feats

        # Simplified along flattened order
        q = _quantize_intensities(roi, min(self.config.gray_levels, self.config.glcm_levels_cap))
        # Identify runs
        diffs = np.diff(q)
        run_boundaries = np.where(diffs != 0)[0] + 1
        runs = np.split(q, run_boundaries)
        run_lengths = np.array([len(r) for r in runs], dtype=np.float32)
        feats["radiomics_run_length_count"] = float(len(run_lengths))
        feats["radiomics_run_length_mean"] = float(np.mean(run_lengths))
        feats["radiomics_run_length_std"] = float(np.std(run_lengths))
        feats["radiomics_run_length_max"] = float(np.max(run_lengths))
        if run_lengths.sum() > 0:
            feats["radiomics_run_length_uniformity"] = float(
                np.sum((run_lengths / run_lengths.sum()) ** 2)
            )
        return feats


# --------------------------------------------------------------------------------------
# Module Test / Example (Optional Guard)
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # This block is illustrative; replace paths accordingly.
    import sys
    if len(sys.argv) < 2:
        print("Usage: python radiomics.py <image.nii[.gz]> [mask.nii[.gz]]")
        sys.exit(0)

    image_fp = sys.argv[1]
    mask_fp = sys.argv[2] if len(sys.argv) > 2 else None
    extractor = RadiomicsExtractor(RadiomicsConfig())
    feats = extractor.extract_features(image_fp, mask_fp)
    print(f"Extracted {len(feats)} features.")
    # Print a small sample
    for k in list(feats.keys())[:25]:
        print(k, feats[k])
