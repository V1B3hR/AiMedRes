#!/usr/bin/env python
"""
Advanced skull (or foreground) stripping / segmentation module.

Supports brain MRI skull stripping and (optionally) single-cell volumetric
foreground extraction with adaptive preprocessing, multiple algorithms,
and quality metrics.

Author: Advanced Refactor
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import subprocess
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import nibabel as nib

from scipy import ndimage
from scipy.ndimage import (
    binary_closing,
    binary_dilation,
    binary_erosion,
    binary_fill_holes,
    binary_opening,
    distance_transform_edt,
    gaussian_filter,
)

# Optional imports (graceful degradation)
try:
    import SimpleITK as sitk  # For bias field correction
    _HAS_SITK = True
except Exception:
    _HAS_SITK = False

try:
    import cupy as cp  # GPU operations (optional)
    _HAS_CUPY = True
except Exception:
    _HAS_CUPY = False

try:
    from skimage import measure, morphology as skmorph, exposure
    _HAS_SKIMAGE = True
except Exception:
    _HAS_SKIMAGE = False

try:
    import cc3d  # Connected components accelerated
    _HAS_CC3D = True
except Exception:
    _HAS_CC3D = False

try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

logger = logging.getLogger(__name__)


class ExtractionMethod(str, Enum):
    BET = "bet"
    THRESHOLD = "threshold"
    OTSU = "otsu"
    MORPHOLOGICAL = "morphological"
    N4_OTSU = "n4_otsu"
    HYBRID = "hybrid"              # Combined adaptive heuristics
    DEEP = "deep"                  # Placeholder for a DL model
    CONVEX_HULL = "convex_hull"    # Expand largest component via convex hull


class Modality(str, Enum):
    BRAIN_MRI = "brain_mri"
    SINGLE_CELL = "single_cell"


@dataclass
class SkullStripConfig:
    method: ExtractionMethod = ExtractionMethod.BET
    modality: Modality = Modality.BRAIN_MRI

    # BET parameters
    bet_fractional_intensity: float = 0.3
    bet_gradient: float = 0.4

    # Threshold parameters
    threshold_percentile: float = 15.0
    upper_clip_percentile: float = 99.8
    lower_clip_percentile: float = 0.2

    # Otsu / multi-threshold
    use_multi_otsu: bool = False
    multi_otsu_classes: int = 3

    # N4 (bias correction)
    apply_bias_correction: bool = False
    n4_shrink_factor: int = 3
    n4_convergence: Tuple[int, float] = (50, 1e-6)

    # Preprocessing
    denoise_sigma: Optional[float] = 0.5
    intensity_normalization: bool = True
    zscore_outlier_clip: Optional[float] = 6.0
    rescale_to_unit: bool = True

    # Morphology
    clean_small_components: bool = True
    min_component_volume_ratio: float = 0.0002
    aggressive_clean: bool = False
    structure_size_primary: int = 3
    structure_size_aggressive: int = 5
    convex_hull_dilate_iters: int = 2

    # GPU toggles
    try_gpu: bool = False

    # Deep learning placeholder
    deep_model_path: Optional[str] = None

    # Saving
    save_mask: bool = True

    # Debug
    debug: bool = False

    # Hybrid mixing
    hybrid_use_otsu_first: bool = True
    hybrid_post_morph: bool = True

    # Edge enhancement (single cell)
    enhance_edges: bool = False

    # Distance based refinement
    distance_refine: bool = True
    distance_refine_fraction: float = 0.08

    # Logging verbosity
    verbose: bool = False

    extra: Dict[str, Any] = field(default_factory=dict)

    def adjust_for_modality(self):
        if self.modality == Modality.SINGLE_CELL:
            # Single-cell images might have different intensity distributions.
            self.threshold_percentile = min(self.threshold_percentile, 5.0)
            self.min_component_volume_ratio = 0.00001
            self.structure_size_primary = 2
            self.structure_size_aggressive = 3
            self.enhance_edges = True
            self.distance_refine_fraction = 0.03


class SkullStrippingError(Exception):
    pass


class SkullStripper:
    """
    Advanced brain / foreground extraction with multiple algorithms and
    adaptive strategies.
    """

    def __init__(self, config: Optional[SkullStripConfig] = None, **kwargs):
        if config is None:
            config = SkullStripConfig(**kwargs)
        self.config = config
        self.config.adjust_for_modality()
        self.logger = logger

        if self.config.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        else:
            logging.getLogger().setLevel(logging.INFO)

        self.logger.debug(f"Initialized SkullStripper with config: {self.config}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def extract_brain(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        mask_path: Optional[Union[str, Path]] = None,
    ) -> Union[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Extract foreground/brain region from a volumetric image.

        Returns str if output_path specified, else (brain, mask).
        """
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input image not found: {input_path}")

        self.logger.info(
            f"Applying method={self.config.method.value} to {input_path} (modality={self.config.modality})"
        )

        img = nib.load(str(input_path))
        data = img.get_fdata(dtype=np.float32)

        original_data = data.copy()

        data = self._preprocess(data)

        method = self.config.method

        if method == ExtractionMethod.BET:
            brain_data, brain_mask = self._bet_route(input_path, data, img)
        elif method == ExtractionMethod.THRESHOLD:
            brain_mask = self._create_brain_mask_threshold(data)
            brain_data = data * brain_mask
        elif method == ExtractionMethod.OTSU:
            brain_mask = self._create_brain_mask_otsu(data)
            brain_data = data * brain_mask
        elif method == ExtractionMethod.MORPHOLOGICAL:
            brain_mask = self._create_brain_mask_morphological(data)
            brain_data = data * brain_mask
        elif method == ExtractionMethod.N4_OTSU:
            if not self.config.apply_bias_correction:
                self.logger.debug("Forcing bias correction on N4_OTSU method.")
                self.config.apply_bias_correction = True
                data = self._preprocess(original_data, force_bias=True)
            brain_mask = self._create_brain_mask_otsu(data)
            brain_mask = self._refine_mask_post(data, brain_mask)
            brain_data = data * brain_mask
        elif method == ExtractionMethod.HYBRID:
            brain_mask = self._hybrid_mask(data)
            brain_data = data * brain_mask
        elif method == ExtractionMethod.DEEP:
            brain_mask = self._deep_model_inference(data)
            brain_mask = self._refine_mask_post(data, brain_mask)
            brain_data = data * brain_mask
        elif method == ExtractionMethod.CONVEX_HULL:
            base_mask = self._create_brain_mask_otsu(data)
            brain_mask = self._expand_convex_hull(base_mask)
            brain_data = data * brain_mask
        else:
            raise ValueError(f"Unknown method: {method}")

        metrics = self.get_extraction_quality_metrics(
            original_data=original_data,
            brain_data=brain_data,
            brain_mask=brain_mask,
        )
        self.logger.info(f"Extraction metrics: {metrics}")

        if output_path:
            output_path = Path(output_path)
            brain_img = nib.Nifti1Image(brain_data.astype(np.float32), img.affine, img.header)
            nib.save(brain_img, str(output_path))

            if mask_path and self.config.save_mask:
                mask_path = Path(mask_path)
                mask_img = nib.Nifti1Image(brain_mask.astype(np.float32), img.affine, img.header)
                nib.save(mask_img, str(mask_path))
            return str(output_path)
        else:
            return brain_data.astype(np.float32), brain_mask.astype(np.float32)

    # ------------------------------------------------------------------
    # BET route with fallback
    # ------------------------------------------------------------------
    def _bet_route(
        self,
        input_path: Path,
        data: np.ndarray,
        img: nib.Nifti1Image,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Attempt FSL BET; fallback to hybrid if missing."""
        try:
            return self._bet_extraction(input_path)
        except SkullStrippingError:
            self.logger.warning("FSL BET failed or unavailable; falling back to HYBRID.")
            mask = self._hybrid_mask(data)
            return data * mask, mask

    def _bet_extraction(self, input_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Run external FSL BET if available."""
        cmd = [
            "bet",
            str(input_path),
            str(input_path.with_suffix(".bet.nii.gz")),
            "-f",
            str(self.config.bet_fractional_intensity),
            "-g",
            str(self.config.bet_gradient),
            "-m",
        ]
        self.logger.debug(f"Running BET command: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        except FileNotFoundError as e:
            raise SkullStrippingError("FSL BET not found") from e
        except subprocess.TimeoutExpired as e:
            raise SkullStrippingError("FSL BET timed out") from e

        if result.returncode != 0:
            self.logger.error(f"BET failed: {result.stderr}")
            raise SkullStrippingError("BET execution failure")

        brain_nii = input_path.with_suffix(".bet.nii.gz")
        mask_nii = input_path.with_suffix(".bet_mask.nii.gz")

        if not brain_nii.exists():
            raise SkullStrippingError("BET output missing")
        brain_img = nib.load(str(brain_nii))
        brain_data = brain_img.get_fdata(dtype=np.float32)

        if mask_nii.exists():
            mask_img = nib.load(str(mask_nii))
            mask_data = mask_img.get_fdata(dtype=np.float32) > 0
        else:
            mask_data = brain_data > 0

        # Clean up intermediate? Keep for debugging if requested
        if not self.config.debug:
            try:
                brain_nii.unlink(missing_ok=True)
                mask_nii.unlink(missing_ok=True)
            except Exception:
                pass

        return brain_data, mask_data.astype(np.float32)

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------
    def _preprocess(self, data: np.ndarray, force_bias: bool = False) -> np.ndarray:
        cfg = self.config
        if cfg.debug:
            self.logger.debug(f"Preprocess start: shape={data.shape}, dtype={data.dtype}")

        # Clip extreme percentile values to stabilize thresholding
        if cfg.lower_clip_percentile is not None and cfg.upper_clip_percentile is not None:
            lo = np.percentile(data, cfg.lower_clip_percentile)
            hi = np.percentile(data, cfg.upper_clip_percentile)
            data = np.clip(data, lo, hi)

        if cfg.apply_bias_correction or force_bias:
            if _HAS_SITK:
                data = self._bias_correct(data)
            else:
                self.logger.warning("Bias correction requested but SimpleITK not installed.")

        if cfg.denoise_sigma and cfg.denoise_sigma > 0:
            data = gaussian_filter(data, sigma=cfg.denoise_sigma)

        if cfg.zscore_outlier_clip:
            mean = data.mean()
            std = data.std() + 1e-8
            z = (data - mean) / std
            data = np.where(np.abs(z) > cfg.zscore_outlier_clip, mean, data)

        if cfg.intensity_normalization:
            # Normalize by robust statistics
            p2, p98 = np.percentile(data, [2, 98])
            if p98 > p2:
                data = (data - p2) / (p98 - p2 + 1e-8)
            data = np.clip(data, 0, 1)

        if cfg.rescale_to_unit:
            minv, maxv = data.min(), data.max()
            if maxv > minv:
                data = (data - minv) / (maxv - minv)

        if cfg.enhance_edges and _HAS_SKIMAGE:
            # Histogram equalization as simple enhancement
            data = exposure.equalize_adapthist(data.astype(np.float32), clip_limit=0.02)

        if cfg.debug:
            self.logger.debug(
                f"Preprocess end: min={data.min():.4f} max={data.max():.4f} mean={data.mean():.4f}"
            )
        return data.astype(np.float32)

    def _bias_correct(self, data: np.ndarray) -> np.ndarray:
        if not _HAS_SITK:
            return data
        try:
            itk_img = sitk.GetImageFromArray(data)
            mask = sitk.OtsuThreshold(itk_img, 0, 1)
            corrector = sitk.N4BiasFieldCorrectionImageFilter()
            corrector.SetShrinkFactor(self.config.n4_shrink_factor)
            it_out = corrector.Execute(itk_img, mask)
            out = sitk.GetArrayFromImage(it_out)
            return out.astype(np.float32)
        except Exception as e:
            self.logger.warning(f"N4 bias correction failed: {e}")
            return data

    # ------------------------------------------------------------------
    # Mask creation methods
    # ------------------------------------------------------------------
    def _create_brain_mask_threshold(self, data: np.ndarray) -> np.ndarray:
        cfg = self.config
        threshold = np.percentile(data[data > 0], cfg.threshold_percentile)
        self.logger.debug(f"Threshold method: percentile={cfg.threshold_percentile} -> {threshold:.5f}")
        mask = data > threshold
        mask = self._clean_mask(mask, data)
        mask = self._refine_mask_post(data, mask)
        return mask.astype(np.float32)

    def _create_brain_mask_otsu(self, data: np.ndarray) -> np.ndarray:
        flat = data[data > 0]
        if flat.size == 0:
            return np.zeros_like(data, dtype=np.float32)
        hist, bins = np.histogram(flat, bins=256)
        threshold = self._otsu_threshold(hist, bins)
        self.logger.debug(f"Otsu threshold: {threshold:.6f}")

        mask = data > threshold
        if self.config.use_multi_otsu and _HAS_SKIMAGE:
            from skimage.filters import threshold_multiotsu
            try:
                thresholds = threshold_multiotsu(flat, classes=self.config.multi_otsu_classes)
                # Use the highest boundary
                mask = data > thresholds[-1]
                self.logger.debug(f"Multi-Otsu thresholds: {thresholds}")
            except Exception as e:
                self.logger.warning(f"Multi Otsu failed: {e}")

        mask = self._clean_mask(mask, data)
        mask = self._refine_mask_post(data, mask)
        return mask.astype(np.float32)

    def _create_brain_mask_morphological(self, data: np.ndarray) -> np.ndarray:
        # Start from Otsu or threshold
        initial = self._create_brain_mask_otsu(data)
        if self.config.aggressive_clean:
            initial = self._aggressive_morph(initial)
        else:
            initial = binary_closing(initial, iterations=1)
        initial = self._clean_mask(initial, data)
        initial = self._refine_mask_post(data, initial)
        return initial.astype(np.float32)

    def _hybrid_mask(self, data: np.ndarray) -> np.ndarray:
        cfg = self.config
        if cfg.hybrid_use_otsu_first:
            mask_otsu = self._create_brain_mask_otsu(data)
            mask_thresh = self._create_brain_mask_threshold(data)
        else:
            mask_thresh = self._create_brain_mask_threshold(data)
            mask_otsu = self._create_brain_mask_otsu(data)

        combined = (mask_otsu + mask_thresh) > 0.5
        self.logger.debug(
            f"Hybrid combination: otsu_sum={mask_otsu.sum()} thresh_sum={mask_thresh.sum()} merged={combined.sum()}"
        )
        combined = self._clean_mask(combined, data)
        if cfg.hybrid_post_morph:
            combined = binary_closing(combined, structure=np.ones((3, 3, 3)), iterations=1)
            combined = binary_fill_holes(combined)
        combined = self._refine_mask_post(data, combined)
        return combined.astype(np.float32)

    def _deep_model_inference(self, data: np.ndarray) -> np.ndarray:
        if not _HAS_TORCH:
            self.logger.warning("Torch not available; falling back to Otsu.")
            return self._create_brain_mask_otsu(data)
        if not self.config.deep_model_path:
            self.logger.warning("No deep model path provided; fallback to Otsu.")
            return self._create_brain_mask_otsu(data)

        # Placeholder: user to implement actual model logic
        self.logger.info("Deep model inference placeholder – implement model loading here.")
        return self._create_brain_mask_otsu(data)

    def _expand_convex_hull(self, mask: np.ndarray) -> np.ndarray:
        if not _HAS_SKIMAGE:
            self.logger.warning("skimage missing: convex hull fallback to cleaned mask.")
            return mask
        try:
            hull = skmorph.convex_hull_image(mask.astype(bool))
            for _ in range(self.config.convex_hull_dilate_iters):
                hull = binary_dilation(hull)
            return hull.astype(np.float32)
        except Exception as e:
            self.logger.warning(f"Convex hull expansion failed: {e}")
            return mask

    # ------------------------------------------------------------------
    # Mask refinement & morphology
    # ------------------------------------------------------------------
    def _clean_mask(self, mask: np.ndarray, data: Optional[np.ndarray] = None) -> np.ndarray:
        cfg = self.config
        if cfg.clean_small_components:
            mask = self._keep_largest_components(mask, min_ratio=cfg.min_component_volume_ratio)
        mask = binary_fill_holes(mask)
        # Light opening to remove spikes
        structure = np.ones((cfg.structure_size_primary,) * 3)
        mask = binary_opening(mask, structure=structure, iterations=1)
        return mask

    def _aggressive_morph(self, mask: np.ndarray) -> np.ndarray:
        cfg = self.config
        structure = np.ones((cfg.structure_size_aggressive,) * 3)
        mask = binary_erosion(mask, structure=structure, iterations=1)
        mask = binary_dilation(mask, structure=structure, iterations=2)
        mask = binary_fill_holes(mask)
        return mask

    def _refine_mask_post(self, data: np.ndarray, mask: np.ndarray) -> np.ndarray:
        cfg = self.config
        if cfg.distance_refine:
            dist = distance_transform_edt(mask)
            cutoff = dist.max() * cfg.distance_refine_fraction
            refined = dist > cutoff
            # Keep connectivity
            refined = self._keep_largest_components(refined, min_ratio=cfg.min_component_volume_ratio)
            mask = refined

        # Final fill
        mask = binary_fill_holes(mask)

        return mask

    def _keep_largest_components(self, mask: np.ndarray, min_ratio: float) -> np.ndarray:
        total_voxels = mask.size
        if mask.sum() == 0:
            return mask
        if _HAS_CC3D:
            labels = cc3d.connected_components(mask.astype(np.uint8), connectivity=6)
            counts = np.bincount(labels.ravel())
            counts[0] = 0
            keep_labels = [i for i, c in enumerate(counts) if c / total_voxels >= min_ratio]
            if not keep_labels:
                keep_labels = [np.argmax(counts)]
            filtered = np.isin(labels, keep_labels)
            return filtered
        else:
            labeled, num = ndimage.label(mask)
            if num <= 1:
                return mask
            sizes = ndimage.sum(mask, labeled, range(1, num + 1))
            sizes = np.asarray(sizes)
            keep = []
            for idx, s in enumerate(sizes, start=1):
                if s / total_voxels >= min_ratio:
                    keep.append(idx)
            if not keep:
                keep = [int(np.argmax(sizes) + 1)]
            new_mask = np.isin(labeled, keep)
            return new_mask

    # ------------------------------------------------------------------
    # Thresholding utility (Otsu)
    # ------------------------------------------------------------------
    def _otsu_threshold(self, hist: np.ndarray, bins: np.ndarray) -> float:
        total = hist.sum()
        if total == 0:
            return 0.0

        sum_total = (hist * bins[:-1]).sum()
        sum_b = 0.0
        w_b = 0.0
        max_var = 0.0
        threshold = bins[0]

        for i in range(len(hist)):
            w_b += hist[i]
            if w_b == 0:
                continue
            w_f = total - w_b
            if w_f == 0:
                break
            sum_b += hist[i] * bins[i]
            m_b = sum_b / w_b
            m_f = (sum_total - sum_b) / w_f
            var_between = w_b * w_f * (m_b - m_f) ** 2
            if var_between > max_var:
                max_var = var_between
                threshold = bins[i]
        return float(threshold)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def validate_inputs(self, input_path: Union[str, Path]) -> bool:
        try:
            input_path = Path(input_path)
            if not input_path.exists():
                return False
            nib.load(str(input_path))
            return True
        except Exception as e:
            self.logger.error(f"Input validation failed: {e}")
            return False

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    def get_extraction_quality_metrics(
        self,
        original_data: np.ndarray,
        brain_data: np.ndarray,
        brain_mask: np.ndarray,
    ) -> Dict[str, float]:
        total_voxels = int(np.prod(original_data.shape))
        brain_voxels = int(np.sum(brain_mask > 0))
        volume_ratio = brain_voxels / total_voxels if total_voxels > 0 else 0.0

        # Intensity correlation within mask
        mask_idx = brain_mask > 0
        if np.any(mask_idx):
            corr = np.corrcoef(
                original_data[mask_idx].flatten(),
                brain_data[mask_idx].flatten(),
            )[0, 1]
        else:
            corr = 0.0

        # Edge sharpness: gradient magnitude inside vs boundary
        try:
            grad = np.sqrt(
                np.sum([(np.gradient(brain_data, axis=i)) ** 2 for i in range(brain_data.ndim)], axis=0)
            )
            edge_vals = grad[mask_idx]
            edge_sharpness = float(np.mean(edge_vals)) if edge_vals.size > 0 else 0.0
        except Exception:
            edge_sharpness = 0.0

        # Compactness / surface area
        compactness = 0.0
        surface_area = 0.0
        if brain_voxels > 0:
            try:
                if _HAS_SKIMAGE:
                    verts, faces, _, _ = measure.marching_cubes(brain_mask.astype(np.uint8), 0)
                    # Approximate surface area
                    tri_areas = []
                    for f in faces:
                        a, b, c = verts[f]
                        tri_areas.append(
                            0.5 * np.linalg.norm(np.cross(b - a, c - a))
                        )
                    surface_area = float(np.sum(tri_areas))
                    volume = float(brain_voxels)
                    if surface_area > 0:
                        compactness = (math.pi ** (1 / 3)) * ((6 * volume) ** (2 / 3)) / surface_area
                else:
                    # Fallback heuristic
                    surface_area = float(np.sum(np.abs(np.gradient(brain_mask.astype(float)))))
                    volume = float(brain_voxels)
                    compactness = (
                        (math.pi ** (1 / 3)) * ((6 * volume) ** (2 / 3)) / (surface_area + 1e-8)
                    )
            except Exception:
                compactness = 0.0

        # Centroid shift from original intensity center
        mask_coords = np.argwhere(mask_idx)
        if mask_coords.size > 0:
            mask_centroid = mask_coords.mean(axis=0)
            intensity_center = np.array(
                ndimage.center_of_mass(original_data > np.percentile(original_data, 60))
            )
            centroid_shift = float(np.linalg.norm(mask_centroid - intensity_center))
        else:
            centroid_shift = 0.0

        # Bounding box volume ratio
        if mask_coords.size > 0:
            mins = mask_coords.min(axis=0)
            maxs = mask_coords.max(axis=0) + 1
            bbox_volume = float(np.prod(maxs - mins))
            bbox_fill_ratio = brain_voxels / bbox_volume if bbox_volume > 0 else 0.0
        else:
            bbox_fill_ratio = 0.0

        return {
            "brain_volume_ratio": float(volume_ratio),
            "intensity_correlation": float(corr) if not np.isnan(corr) else 0.0,
            "mask_compactness": float(compactness) if not np.isnan(compactness) else 0.0,
            "brain_voxel_count": float(brain_voxels),
            "total_voxel_count": float(total_voxels),
            "edge_sharpness": float(edge_sharpness),
            "centroid_shift": float(centroid_shift),
            "bbox_fill_ratio": float(bbox_fill_ratio),
            "surface_area_est": float(surface_area),
        }


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Advanced skull / foreground stripping tool supporting brain MRI and single-cell volumes."
    )
    p.add_argument("input", type=str, help="Input NIfTI (.nii / .nii.gz)")
    p.add_argument("--output", type=str, help="Output NIfTI for stripped brain/foreground")
    p.add_argument("--mask", type=str, help="Optional output mask path")
    p.add_argument(
        "--method",
        type=str,
        default="bet",
        choices=[m.value for m in ExtractionMethod],
        help="Extraction method",
    )
    p.add_argument(
        "--modality",
        type=str,
        default="brain_mri",
        choices=[m.value for m in Modality],
        help="Imaging modality (adjusts heuristics)",
    )
    p.add_argument("--bias", action="store_true", help="Apply N4 bias field correction (if SimpleITK available)")
    p.add_argument("--denoise-sigma", type=float, default=0.5, help="Gaussian denoise sigma")
    p.add_argument("--no-norm", action="store_true", help="Disable intensity normalization")
    p.add_argument("--debug", action="store_true", help="Enable debug logs")
    p.add_argument("--verbose", action="store_true", help="Verbose logging")
    p.add_argument("--aggressive", action="store_true", help="Aggressive morphological cleanup")
    p.add_argument("--multi-otsu", action="store_true", help="Use multi-level Otsu with skimage if available")
    p.add_argument("--deep-model", type=str, help="Path to deep learning model (placeholder)")
    return p


def main_cli(argv: Optional[List[str]] = None):
    args = build_arg_parser().parse_args(argv)

    config = SkullStripConfig(
        method=ExtractionMethod(args.method),
        modality=Modality(args.modality),
        apply_bias_correction=args.bias,
        denoise_sigma=args.denoise_sigma,
        intensity_normalization=not args.no_norm,
        debug=args.debug,
        verbose=args.verbose,
        aggressive_clean=args.aggressive,
        use_multi_otsu=args.multi_otsu,
        deep_model_path=args.deep_model,
    )

    stripper = SkullStripper(config=config)
    result = stripper.extract_brain(args.input, args.output, args.mask)
    if isinstance(result, str):
        print(f"Saved stripped volume to: {result}")
    else:
        print("Extraction completed (returned arrays).")


if __name__ == "__main__":
    main_cli()
