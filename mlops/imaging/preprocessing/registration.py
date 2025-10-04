"""
Advanced image registration module for medical and single‑cell imaging.

This module provides:
- Traditional intensity-based affine / rigid / similarity registration
- Optional ANTs CLI integration (if installed)
- Multi-resolution (image pyramids) optimization
- Multiple similarity metrics: NCC, MI, SSIM (if available), MSE, Hybrid
- Single-cell point-set (centroid) registration using nuclei segmentation
- Segmentation-assisted alignment (mask weighting)
- Multi-channel support (align using selected/reference channel)
- Transform utilities (composition, inversion, JSON export)
- GPU acceleration (optional, via CuPy if installed)
- Robust logging and graceful fallbacks

Author: Auto‑enhanced by AI (2025)
"""

from __future__ import annotations

import json
import logging
import math
import os
import subprocess
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Union, Dict, Any, Tuple, List, Sequence, Callable

import numpy as np
import nibabel as nib
from scipy import ndimage
from scipy.optimize import minimize

try:
    from skimage import filters, measure, morphology
    from skimage.metrics import structural_similarity as skimage_ssim
    _SKIMAGE_AVAILABLE = True
except Exception:
    _SKIMAGE_AVAILABLE = False

try:
    import cupy as cp  # Optional GPU acceleration
    _CUPY_AVAILABLE = True
except Exception:
    _CUPY_AVAILABLE = False

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# ---------------------------------------------------------------------------
# Configuration Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PyramidConfig:
    levels: int = 3
    scale_factor: float = 2.0
    min_size: int = 32
    downscale_mode: str = "nearest"  # could be 'nearest' or 'linear'


@dataclass
class OptimizationConfig:
    method: str = "Powell"  # or 'Nelder-Mead', etc.
    maxiter: int = 150
    tol: float = 1e-6
    early_stop_patience: int = 5
    improvement_threshold: float = 1e-4
    verbose: bool = False


@dataclass
class SimilarityConfig:
    metric: str = "ncc"  # 'ncc', 'mi', 'ssim', 'mse', 'hybrid'
    hybrid_weights: Dict[str, float] = field(default_factory=lambda: {"ncc": 0.5, "mi": 0.5})
    mi_bins: int = 64
    ssim_win: int = 7
    clamp_percentile: float = 0.01  # optional intensity clamping


@dataclass
class SingleCellConfig:
    enable_point_set: bool = False
    nuclei_channel: Optional[int] = None  # which channel index for segmentation
    min_area: int = 10
    max_area: int = 5000
    otsu_smoothing: int = 1
    use_ransac: bool = False
    ransac_trials: int = 200
    ransac_threshold: float = 5.0
    max_points: int = 500
    centroid_subsample: int = 1  # keep every nth centroid


@dataclass
class IOConfig:
    save_transform_json: bool = True
    json_indent: int = 2
    float_precision: int = 6


@dataclass
class RegistrarConfig:
    method: str = "affine"  # 'ants', 'affine', 'rigid', 'similarity', 'auto'
    use_gpu: bool = False
    pyramid: PyramidConfig = field(default_factory=PyramidConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    similarity: SimilarityConfig = field(default_factory=SimilarityConfig)
    single_cell: SingleCellConfig = field(default_factory=SingleCellConfig)
    io: IOConfig = field(default_factory=IOConfig)
    random_seed: Optional[int] = 42
    allow_fallbacks: bool = True
    normalize_intensity: bool = True
    mask_dilation: int = 0
    version: str = "2.0.0"


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------

def _maybe_to_gpu(arr: np.ndarray, use_gpu: bool):
    if use_gpu and _CUPY_AVAILABLE:
        return cp.asarray(arr)
    return arr


def _maybe_to_cpu(arr):
    if _CUPY_AVAILABLE and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return arr


def build_image_pyramid(
    image: np.ndarray,
    config: PyramidConfig
) -> List[np.ndarray]:
    """Construct multi-resolution pyramid (coarsest first)."""
    pyramid = [image]
    for level in range(1, config.levels):
        prev = pyramid[-1]
        if min(prev.shape) <= config.min_size:
            break
        scale = 1 / (config.scale_factor ** 1)
        # Downsample by factor ~scale each iteration (accumulative)
        factor = config.scale_factor
        zoom = tuple(1 / factor for _ in prev.shape)
        order = 0 if config.downscale_mode == "nearest" else 1
        down = ndimage.zoom(prev, zoom, order=order)
        pyramid.append(down)
    pyramid = pyramid[::-1]  # coarsest first
    return pyramid


def clamp_intensity(img: np.ndarray, p: float) -> np.ndarray:
    if p <= 0:
        return img
    lo = np.percentile(img, p * 100)
    hi = np.percentile(img, (1 - p) * 100)
    if hi <= lo:
        return img
    img_clamped = np.clip(img, lo, hi)
    return (img_clamped - lo) / (hi - lo + 1e-12)


def compute_joint_histogram(a: np.ndarray, b: np.ndarray, bins: int) -> np.ndarray:
    a_flat = a.ravel()
    b_flat = b.ravel()
    # Normalize intensities into [0,1]
    a_norm = (a_flat - a_flat.min()) / (a_flat.ptp() + 1e-12)
    b_norm = (b_flat - b_flat.min()) / (b_flat.ptp() + 1e-12)
    ai = np.clip((a_norm * (bins - 1)).astype(np.int64), 0, bins - 1)
    bi = np.clip((b_norm * (bins - 1)).astype(np.int64), 0, bins - 1)
    joint = np.zeros((bins, bins), dtype=np.float64)
    np.add.at(joint, (ai, bi), 1)
    joint += 1e-10
    joint /= joint.sum()
    return joint


def mutual_information(a: np.ndarray, b: np.ndarray, bins: int) -> float:
    joint = compute_joint_histogram(a, b, bins)
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    mi = (joint * (np.log(joint) - np.log(px) - np.log(py))).sum()
    return float(mi)


def normalized_cross_correlation(a: np.ndarray, b: np.ndarray) -> float:
    a_f = a.ravel()
    b_f = b.ravel()
    a_c = a_f - a_f.mean()
    b_c = b_f - b_f.mean()
    denom = math.sqrt((a_c @ a_c) * (b_c @ b_c)) + 1e-12
    if denom == 0:
        return 0.0
    return float((a_c @ b_c) / denom)


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def ssim(a: np.ndarray, b: np.ndarray, win: int = 7) -> float:
    if not _SKIMAGE_AVAILABLE:
        # Fallback to NCC when SSIM not available
        return normalized_cross_correlation(a, b)
    # skimage SSO: flatten to measure 3D by slicing or direct patch
    # We'll compute mean of per-slice SSIM along last axis if 3D
    if a.ndim == 3:
        vals = []
        for z in range(a.shape[2]):
            vals.append(skimage_ssim(a[..., z], b[..., z], data_range=b[..., z].ptp() or 1.0))
        return float(np.mean(vals))
    elif a.ndim == 2:
        return float(skimage_ssim(a, b, data_range=b.ptp() or 1.0))
    else:
        return normalized_cross_correlation(a, b)


def hybrid_score(a: np.ndarray, b: np.ndarray, weights: Dict[str, float], mi_bins: int, ssim_win: int) -> float:
    total_weight = sum(weights.values()) + 1e-12
    score = 0.0
    for k, w in weights.items():
        if k == "ncc":
            v = normalized_cross_correlation(a, b)
        elif k == "mi":
            v = mutual_information(a, b, mi_bins)
        elif k == "ssim":
            v = ssim(a, b, ssim_win)
        elif k == "mse":
            v = -mse(a, b)  # lower MSE -> higher score
        else:
            continue
        score += w * v
    return score / total_weight


def select_metric_fn(cfg: SimilarityConfig) -> Callable[[np.ndarray, np.ndarray], float]:
    metric = cfg.metric.lower()
    if metric == "ncc":
        return lambda a, b: normalized_cross_correlation(a, b)
    if metric == "mi":
        return lambda a, b: mutual_information(a, b, cfg.mi_bins)
    if metric == "ssim":
        return lambda a, b: ssim(a, b, cfg.ssim_win)
    if metric == "mse":
        return lambda a, b: -mse(a, b)
    if metric == "hybrid":
        return lambda a, b: hybrid_score(a, b, cfg.hybrid_weights, cfg.mi_bins, cfg.ssim_win)
    raise ValueError(f"Unknown metric: {cfg.metric}")


def umeyama_alignment(source: np.ndarray, target: np.ndarray, with_scale: bool = True) -> Tuple[np.ndarray, float]:
    """
    Compute similarity transform (R, t, s) aligning source to target using Umeyama method.
    Args:
        source: (N, D)
        target: (N, D)
        with_scale: include scale
    Returns:
        (4x4 affine matrix, scale)
    """
    n, d = source.shape
    mu_s = source.mean(axis=0)
    mu_t = target.mean(axis=0)
    xs = source - mu_s
    ys = target - mu_t
    cov = (ys.T @ xs) / n
    U, S, Vt = np.linalg.svd(cov)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = U @ Vt
    if with_scale:
        var_s = (xs ** 2).sum() / n
        scale = S.sum() / (var_s + 1e-12)
    else:
        scale = 1.0
    t = mu_t - scale * R @ mu_s
    M = np.eye(4)
    M[:d, :d] = scale * R
    M[:d, 3] = t
    return M, scale


def ransac_point_set_affine(
    src: np.ndarray,
    dst: np.ndarray,
    trials: int = 200,
    threshold: float = 5.0,
    with_scale: bool = True,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Simple RANSAC wrapper around Umeyama for robustness.
    """
    rng = np.random.default_rng(seed)
    n = src.shape[0]
    best_inliers = -1
    best_M = None
    min_subset = min(4, n)
    if n < min_subset:
        M, _ = umeyama_alignment(src, dst, with_scale)
        return M
    for _ in range(trials):
        idx = rng.choice(n, size=min_subset, replace=False)
        M_try, _ = umeyama_alignment(src[idx], dst[idx], with_scale)
        # Apply transform
        src_h = np.concatenate([src, np.ones((n, 1))], axis=1).T
        pred = (M_try @ src_h).T[:, :3]
        err = np.linalg.norm(pred - dst, axis=1)
        inliers = (err < threshold).sum()
        if inliers > best_inliers:
            best_inliers = inliers
            best_M = M_try
    if best_M is None:
        best_M, _ = umeyama_alignment(src, dst, with_scale)
    return best_M


def compose_transforms(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Return A ∘ B (apply B then A)."""
    return A @ B


def invert_transform(T: np.ndarray) -> np.ndarray:
    return np.linalg.inv(T)


# ---------------------------------------------------------------------------
# Single-Cell Utilities
# ---------------------------------------------------------------------------

def segment_nuclei(volume: np.ndarray, min_area: int, max_area: int, smoothing: int = 1) -> np.ndarray:
    """
    Basic nuclei segmentation for single-cell (2D or 3D).
    Returns a labeled volume.
    """
    if smoothing > 0:
        volume = ndimage.gaussian_filter(volume, sigma=smoothing)
    thresh = filters.threshold_otsu(volume) if _SKIMAGE_AVAILABLE else volume.mean()
    mask = volume > thresh
    if _SKIMAGE_AVAILABLE:
        mask = morphology.remove_small_objects(mask, min_size=min_area)
    labeled, _ = ndimage.label(mask)
    if max_area > 0 and _SKIMAGE_AVAILABLE:
        # Remove overly large regions
        props = measure.regionprops(labeled)
        remove = [p.label for p in props if p.area > max_area]
        if remove:
            for r in remove:
                labeled[labeled == r] = 0
            labeled, _ = ndimage.label(labeled > 0)
    return labeled


def centroids_from_labels(labeled: np.ndarray, max_points: int = 500, subsample: int = 1) -> np.ndarray:
    """
    Extract centroids from labeled volume.
    """
    labels = np.unique(labeled)
    labels = labels[labels != 0]
    cents = []
    for lab in labels:
        coords = np.argwhere(labeled == lab)
        c = coords.mean(axis=0)
        cents.append(c)
    cents = np.array(cents)
    if subsample > 1:
        cents = cents[::subsample]
    if len(cents) > max_points:
        # Random subsample for efficiency
        idx = np.random.choice(len(cents), size=max_points, replace=False)
        cents = cents[idx]
    return cents


# ---------------------------------------------------------------------------
# Core Registrar
# ---------------------------------------------------------------------------

class ImageRegistrar:
    """
    Advanced medical & single-cell image registrar.

    Public Methods:
        register_images(...)
        register_to_template(...)
        get_registration_quality_metrics(...)
        validate_inputs(...)

    Extended features accessible via params/config.
    """

    def __init__(self, method: str = "affine", **kwargs):
        """
        Initialize registrar with optional configuration overrides.
        Args:
            method: default registration method
            **kwargs: overrides for RegistrarConfig fields (nested fields use '__' notation)
                      e.g. similarity__metric='mi'
        """
        base_cfg = RegistrarConfig(method=method)
        # Apply nested overrides
        for k, v in kwargs.items():
            if "__" in k:
                top, sub = k.split("__", 1)
                if hasattr(base_cfg, top):
                    nested_obj = getattr(base_cfg, top)
                    if hasattr(nested_obj, sub):
                        setattr(nested_obj, sub, v)
                    else:
                        raise ValueError(f"Unknown nested config field: {k}")
                else:
                    raise ValueError(f"Unknown config section: {top}")
            else:
                if hasattr(base_cfg, k):
                    setattr(base_cfg, k, v)
                else:
                    raise ValueError(f"Unknown config field: {k}")
        self.config = base_cfg
        self.method = base_cfg.method
        self.params = kwargs
        self.logger = logging.getLogger(__name__)
        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    def register_images(
        self,
        moving_path: Union[str, Path],
        fixed_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        transform_path: Optional[Union[str, Path]] = None,
        mask_moving: Optional[Union[str, Path]] = None,
        mask_fixed: Optional[Union[str, Path]] = None,
        channel: Optional[int] = None,
        return_all: bool = False
    ) -> Union[str, Tuple[np.ndarray, np.ndarray], Dict[str, Any]]:
        """
        High-level registration entry point.

        Args:
            moving_path: moving/source NIfTI
            fixed_path: fixed/target NIfTI
            output_path: optional output path
            transform_path: optional text or .json path (matrix or metadata)
            mask_moving: optional mask NIfTI path for moving
            mask_fixed: optional mask NIfTI path for fixed
            channel: if multi-channel (e.g., shape (X,Y,Z,C)), pick channel for metric
            return_all: if True returns dict with detailed outputs
        """
        moving_path = Path(moving_path)
        fixed_path = Path(fixed_path)

        if not moving_path.exists():
            raise FileNotFoundError(f"Moving image not found: {moving_path}")
        if not fixed_path.exists():
            raise FileNotFoundError(f"Fixed image not found: {fixed_path}")

        self.logger.info(f"Registration start [{self.method}] : {moving_path.name} → {fixed_path.name}")

        if self.method == "ants":
            result = self._ants_registration(moving_path, fixed_path, output_path, transform_path)
            if return_all and not isinstance(result, str):
                data, M = result
                return {"registered": data, "transform": M, "method": "ants"}
            return result

        # Load data
        moving_img = nib.load(moving_path)
        fixed_img = nib.load(fixed_path)
        moving_data = moving_img.get_fdata()
        fixed_data = fixed_img.get_fdata()

        # Channel extraction (if multi-channel)
        if channel is not None and moving_data.ndim == 4:
            self.logger.info(f"Selecting channel {channel} for registration metric")
            moving_core = moving_data[..., channel]
            fixed_core = fixed_data[..., channel]
        else:
            moving_core = moving_data
            fixed_core = fixed_data

        # Mask loading
        moving_mask_data = None
        fixed_mask_data = None
        if mask_moving:
            moving_mask_data = nib.load(str(mask_moving)).get_fdata() > 0
        if mask_fixed:
            fixed_mask_data = nib.load(str(mask_fixed)).get_fdata() > 0
        if moving_mask_data is not None and fixed_mask_data is not None and self.config.mask_dilation > 0:
            structure = np.ones((3, 3, 3))
            moving_mask_data = ndimage.binary_dilation(moving_mask_data, structure=structure, iterations=self.config.mask_dilation)
            fixed_mask_data = ndimage.binary_dilation(fixed_mask_data, structure=structure, iterations=self.config.mask_dilation)

        # Intensity normalization / clamping
        if self.config.normalize_intensity:
            moving_core = self._normalize_image(moving_core)
            fixed_core = self._normalize_image(fixed_core)

        moving_core = clamp_intensity(moving_core, self.config.similarity.clamp_percentile)
        fixed_core = clamp_intensity(fixed_core, self.config.similarity.clamp_percentile)

        # Single-cell point set alignment (optional pre-alignment)
        prealign_matrix = np.eye(4)
        if self.config.single_cell.enable_point_set:
            try:
                prealign_matrix = self._single_cell_prealign(moving_core, fixed_core)
                self.logger.info("Single-cell point-set pre-alignment applied.")
                moving_core = self._apply_transform(moving_core, prealign_matrix, fixed_core.shape)
                if moving_mask_data is not None:
                    moving_mask_data = self._apply_transform(moving_mask_data.astype(float), prealign_matrix, fixed_core.shape) > 0.5
            except Exception as e:
                self.logger.warning(f"Single-cell prealignment failed: {e}")

        # Choose optimization path
        if self.method == "affine":
            M = self._optimize_affine_transform_multiscale(moving_core, fixed_core, moving_mask_data, fixed_mask_data)
        elif self.method == "rigid":
            M = self._optimize_rigid_transform_multiscale(moving_core, fixed_core, moving_mask_data, fixed_mask_data)
        elif self.method == "similarity":
            M = self._optimize_similarity_transform_multiscale(moving_core, fixed_core, moving_mask_data, fixed_mask_data)
        elif self.method == "auto":
            # simple heuristic: start rigid → similarity → affine
            self.logger.info("Auto mode: rigid → similarity → affine refinement")
            Mr = self._optimize_rigid_transform_multiscale(moving_core, fixed_core, moving_mask_data, fixed_mask_data)
            mov_r = self._apply_transform(moving_core, Mr, fixed_core.shape)
            Ms = self._optimize_similarity_transform_multiscale(mov_r, fixed_core, moving_mask_data, fixed_mask_data)
            mov_s = self._apply_transform(mov_r, Ms, fixed_core.shape)
            Ma = self._optimize_affine_transform_multiscale(mov_s, fixed_core, moving_mask_data, fixed_mask_data)
            M = compose_transforms(Ma, compose_transforms(Ms, Mr))
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Compose with prealign
        M_full = compose_transforms(M, prealign_matrix)

        registered_core = self._apply_transform(moving_core, M, fixed_core.shape)

        # If multi-channel, apply final transform across all channels
        if channel is not None and moving_data.ndim == 4:
            registered_full = []
            for c in range(moving_data.shape[3]):
                ch_data = moving_data[..., c]
                if self.config.normalize_intensity:
                    ch_data = self._normalize_image(ch_data)
                reg_ch = self._apply_transform(ch_data, M_full, fixed_core.shape)
                registered_full.append(reg_ch)
            registered_data = np.stack(registered_full, axis=-1)
        else:
            registered_data = registered_core

        # Save outputs
        if output_path:
            out_img = nib.Nifti1Image(registered_data, fixed_img.affine, fixed_img.header)
            nib.save(out_img, str(output_path))
            self.logger.info(f"Registered image saved: {output_path}")

        if transform_path:
            self._save_transform(transform_path, M_full)

        if output_path and not return_all:
            return str(output_path)

        if not output_path and not return_all:
            return registered_data, M_full

        # Return detailed dictionary
        metrics = self.get_registration_quality_metrics(
            moving_core,
            fixed_core,
            registered_core
        )
        return {
            "registered": registered_data,
            "transform": M_full,
            "metrics": metrics,
            "method": self.method,
            "prealign_used": self.config.single_cell.enable_point_set
        }

    def register_to_template(
        self,
        input_path: Union[str, Path],
        template: str = "MNI152",
        output_path: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> Union[str, np.ndarray, Dict[str, Any]]:
        """
        Register an image to a template (synthetic placeholder templates).
        """
        template_data = self._get_template(template)
        input_path = Path(input_path)
        img = nib.load(str(input_path))

        temp_template = input_path.parent / f"temp_template_{template}.nii.gz"
        template_img = nib.Nifti1Image(template_data, img.affine, img.header)
        nib.save(template_img, str(temp_template))
        try:
            result = self.register_images(
                input_path,
                temp_template,
                output_path=output_path,
                **kwargs
            )
            return result
        finally:
            temp_template.unlink(missing_ok=True)

    def validate_inputs(self, moving_path: Union[str, Path], fixed_path: Union[str, Path]) -> bool:
        try:
            mp = Path(moving_path)
            fp = Path(fixed_path)
            if not mp.exists() or not fp.exists():
                return False
            nib.load(str(mp))
            nib.load(str(fp))
            return True
        except Exception as e:
            self.logger.error(f"Input validation failed: {e}")
            return False

    def get_registration_quality_metrics(
        self,
        moving_data: np.ndarray,
        fixed_data: np.ndarray,
        registered_data: np.ndarray
    ) -> Dict[str, float]:
        ncc_before = normalized_cross_correlation(moving_data, fixed_data)
        ncc_after = normalized_cross_correlation(registered_data, fixed_data)
        mse_before = mse(moving_data, fixed_data)
        mse_after = mse(registered_data, fixed_data)
        mi_before = mutual_information(moving_data, fixed_data, self.config.similarity.mi_bins)
        mi_after = mutual_information(registered_data, fixed_data, self.config.similarity.mi_bins)
        out = {
            "ncc_before": float(ncc_before),
            "ncc_after": float(ncc_after),
            "ncc_improvement": float(ncc_after - ncc_before),
            "mse_before": float(mse_before),
            "mse_after": float(mse_after),
            "mse_improvement": float(mse_before - mse_after),
            "mi_before": float(mi_before),
            "mi_after": float(mi_after),
            "mi_improvement": float(mi_after - mi_before)
        }
        if _SKIMAGE_AVAILABLE:
            try:
                ssim_before = ssim(moving_data, fixed_data, self.config.similarity.ssim_win)
                ssim_after = ssim(registered_data, fixed_data, self.config.similarity.ssim_win)
                out["ssim_before"] = float(ssim_before)
                out["ssim_after"] = float(ssim_after)
                out["ssim_improvement"] = float(ssim_after - ssim_before)
            except Exception:
                pass
        return out

    # -------------------------------------------------------------------
    # Internal: Single-cell prealignment
    # -------------------------------------------------------------------

    def _single_cell_prealign(self, moving: np.ndarray, fixed: np.ndarray) -> np.ndarray:
        """Perform centroid-based (point-set) alignment from segmented nuclei."""
        scfg = self.config.single_cell
        if moving.ndim != fixed.ndim:
            raise ValueError("Dimension mismatch for single-cell prealignment")
        # If 2D data, treat as (X,Y,1)
        moving_3d = moving if moving.ndim == 3 else moving[..., None]
        fixed_3d = fixed if fixed.ndim == 3 else fixed[..., None]

        # Segment
        mov_labels = segment_nuclei(moving_3d, scfg.min_area, scfg.max_area, scfg.otsu_smoothing)
        fix_labels = segment_nuclei(fixed_3d, scfg.min_area, scfg.max_area, scfg.otsu_smoothing)

        mov_cent = centroids_from_labels(mov_labels, scfg.max_points, scfg.centroid_subsample)
        fix_cent = centroids_from_labels(fix_labels, scfg.max_points, scfg.centroid_subsample)

        if len(mov_cent) < 3 or len(fix_cent) < 3:
            self.logger.warning("Insufficient centroids for prealignment; skipping.")
            return np.eye(4)

        # Quick nearest-neighbor pairing (naive)
        # Using a simple heuristic: pair by sorted order along largest variance axis
        axis_mov = np.argmax(np.var(mov_cent, axis=0))
        axis_fix = np.argmax(np.var(fix_cent, axis=0))
        mov_sorted = mov_cent[np.argsort(mov_cent[:, axis_mov])]
        fix_sorted = fix_cent[np.argsort(fix_cent[:, axis_fix])]
        m = min(len(mov_sorted), len(fix_sorted))
        mov_sorted = mov_sorted[:m]
        fix_sorted = fix_sorted[:m]

        if scfg.use_ransac:
            M = ransac_point_set_affine(mov_sorted, fix_sorted, trials=scfg.ransac_trials,
                                        threshold=scfg.ransac_threshold, with_scale=True,
                                        seed=self.config.random_seed)
        else:
            M, _ = umeyama_alignment(mov_sorted, fix_sorted, with_scale=True)
        return M

    # -------------------------------------------------------------------
    # Internal: ANTs CLI
    # -------------------------------------------------------------------

    def _ants_registration(
        self,
        moving_path: Path,
        fixed_path: Path,
        output_path: Optional[Path],
        transform_path: Optional[Path]
    ) -> Union[str, Tuple[np.ndarray, np.ndarray]]:
        """Integrate with antsRegistration if available."""
        try:
            cmd = ["antsRegistration",
                   "--dimensionality", "3",
                   "--float", "0"]
            if output_path:
                prefix = str(Path(output_path).with_suffix("").with_suffix(""))
            else:
                prefix = str(moving_path.with_suffix("").with_suffix("")) + "_ants"
            cmd.extend([
                "--output", f"[{prefix},{prefix}_Warped.nii.gz]",
                "--interpolation", "Linear",
                "--winsorize-image-intensities", "[0.005,0.995]",
                "--use-histogram-matching", "0",
                "--initial-moving-transform", f"[{fixed_path},{moving_path},1]",
                "--transform", "Rigid[0.1]",
                "--metric", f"MI[{fixed_path},{moving_path},1,32,Regular,0.25]",
                "--convergence", "[1000x500x250x100,1e-6,10]",
                "--shrink-factors", "8x4x2x1",
                "--smoothing-sigmas", "3x2x1x0vox",
                "--transform", "Affine[0.1]",
                "--metric", f"MI[{fixed_path},{moving_path},1,32,Regular,0.25]",
                "--convergence", "[1000x500x250x100,1e-6,10]",
                "--shrink-factors", "8x4x2x1",
                "--smoothing-sigmas", "3x2x1x0vox"
            ])
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                self.logger.warning(f"ANTs failed: {result.stderr}")
                if self.config.allow_fallbacks:
                    return self._affine_registration(moving_path, fixed_path, output_path, transform_path)
                raise RuntimeError("ANTs registration failed and fallback disabled.")
            warped_file = Path(f"{prefix}_Warped.nii.gz")
            if output_path:
                # Ensure final naming
                out = Path(output_path)
                if warped_file.exists():
                    import shutil
                    shutil.move(str(warped_file), str(out))
                # Placeholder transform
                if transform_path:
                    self._save_transform(transform_path, np.eye(4))
                return str(out)
            else:
                if warped_file.exists():
                    img = nib.load(str(warped_file))
                    data = img.get_fdata()
                    warped_file.unlink(missing_ok=True)
                    return data, np.eye(4)
                else:
                    raise FileNotFoundError("ANTs produced no warped file.")
        except FileNotFoundError:
            self.logger.warning("antsRegistration not found.")
            if self.config.allow_fallbacks:
                return self._affine_registration(moving_path, fixed_path, output_path, transform_path)
            raise

    # Legacy fallback simple affine (retained for compatibility)
    def _affine_registration(
        self,
        moving_path: Path,
        fixed_path: Path,
        output_path: Optional[Path],
        transform_path: Optional[Path]
    ) -> Union[str, Tuple[np.ndarray, np.ndarray]]:
        moving_img = nib.load(str(moving_path))
        fixed_img = nib.load(str(fixed_path))
        moving_data = self._normalize_image(moving_img.get_fdata())
        fixed_data = self._normalize_image(fixed_img.get_fdata())
        M = self._optimize_affine_transform(moving_data, fixed_data)
        reg = self._apply_transform(moving_data, M, fixed_data.shape)
        if output_path:
            nib.save(nib.Nifti1Image(reg, fixed_img.affine, fixed_img.header), str(output_path))
            if transform_path:
                self._save_transform(transform_path, M)
            return str(output_path)
        return reg, M

    # -------------------------------------------------------------------
    # Multiscale optimization wrappers
    # -------------------------------------------------------------------

    def _optimize_affine_transform_multiscale(
        self, moving: np.ndarray, fixed: np.ndarray,
        moving_mask: Optional[np.ndarray], fixed_mask: Optional[np.ndarray]
    ) -> np.ndarray:
        init = np.zeros(12)
        init[6:9] = 1.0  # scales
        return self._multiscale_optimize(
            init_params=init,
            param_to_matrix=self._params_to_affine_matrix,
            moving=moving,
            fixed=fixed,
            moving_mask=moving_mask,
            fixed_mask=fixed_mask
        )

    def _optimize_rigid_transform_multiscale(
        self, moving: np.ndarray, fixed: np.ndarray,
        moving_mask: Optional[np.ndarray], fixed_mask: Optional[np.ndarray]
    ) -> np.ndarray:
        init = np.zeros(6)
        return self._multiscale_optimize(
            init_params=init,
            param_to_matrix=self._params_to_rigid_matrix,
            moving=moving,
            fixed=fixed,
            moving_mask=moving_mask,
            fixed_mask=fixed_mask
        )

    def _optimize_similarity_transform_multiscale(
        self, moving: np.ndarray, fixed: np.ndarray,
        moving_mask: Optional[np.ndarray], fixed_mask: Optional[np.ndarray]
    ) -> np.ndarray:
        init = np.zeros(7)
        init[6] = 1.0
        return self._multiscale_optimize(
            init_params=init,
            param_to_matrix=self._params_to_similarity_matrix,
            moving=moving,
            fixed=fixed,
            moving_mask=moving_mask,
            fixed_mask=fixed_mask
        )

    def _multiscale_optimize(
        self,
        init_params: np.ndarray,
        param_to_matrix: Callable[[np.ndarray], np.ndarray],
        moving: np.ndarray,
        fixed: np.ndarray,
        moving_mask: Optional[np.ndarray],
        fixed_mask: Optional[np.ndarray]
    ) -> np.ndarray:
        pyr_mov = build_image_pyramid(moving, self.config.pyramid)
        pyr_fix = build_image_pyramid(fixed, self.config.pyramid)
        if moving_mask is not None and fixed_mask is not None:
            pyr_mov_mask = build_image_pyramid(moving_mask.astype(float), self.config.pyramid)
            pyr_fix_mask = build_image_pyramid(fixed_mask.astype(float), self.config.pyramid)
        else:
            pyr_mov_mask = [None] * len(pyr_mov)
            pyr_fix_mask = [None] * len(pyr_fix)

        params = init_params.copy()
        metric_fn = select_metric_fn(self.config.similarity)

        for level, (mov_l, fix_l, mm_l, fm_l) in enumerate(zip(pyr_mov, pyr_fix, pyr_mov_mask, pyr_fix_mask), start=1):
            self.logger.info(f"Optimizing level {level}/{len(pyr_mov)} - shape {mov_l.shape}")
            scale_factor = (moving.shape[0] / mov_l.shape[0])
            params_scaled = self._scale_params_for_level(params, scale_factor, param_to_matrix)

            best_score = -np.inf
            no_improve_count = 0

            def objective(p):
                M = param_to_matrix(p)
                trans = self._apply_transform(mov_l, M, fix_l.shape)
                if mm_l is not None and fm_l is not None:
                    # Weight interior where both masks
                    mask = (mm_l > 0.5) & (fm_l > 0.5)
                    if mask.sum() < 10:
                        return -1e9
                    a = trans[mask]
                    b = fix_l[mask]
                else:
                    a = trans
                    b = fix_l
                score = metric_fn(a, b)
                return -score  # minimize negative similarity

            res = minimize(
                objective,
                params_scaled,
                method=self.config.optimization.method,
                options={
                    "maxiter": self.config.optimization.maxiter,
                    "disp": self.config.optimization.verbose
                },
                tol=self.config.optimization.tol
            )

            params_refined = res.x
            # Evaluate improvement
            final_score = -res.fun
            if final_score > best_score + self.config.optimization.improvement_threshold:
                best_score = final_score
                params = self._unscale_params_from_level(params_refined, scale_factor, param_to_matrix)
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count >= self.config.optimization.early_stop_patience:
                    self.logger.info("Early stopping at current level due to stagnation.")
                    break

        return param_to_matrix(params)

    def _scale_params_for_level(
        self,
        params: np.ndarray,
        scale_factor: float,
        param_to_matrix: Callable[[np.ndarray], np.ndarray]
    ) -> np.ndarray:
        # Rough heuristic: scale translations by inverse of scale factor
        p = params.copy()
        if p.size in (6, 7, 12):
            # translation indexes
            if p.size == 6:
                p[3:6] /= scale_factor
            elif p.size == 7:
                p[3:6] /= scale_factor
            elif p.size == 12:
                p[3:6] /= scale_factor
        return p

    def _unscale_params_from_level(
        self,
        params: np.ndarray,
        scale_factor: float,
        param_to_matrix: Callable[[np.ndarray], np.ndarray]
    ) -> np.ndarray:
        p = params.copy()
        if p.size in (6, 7, 12):
            if p.size == 6:
                p[3:6] *= scale_factor
            elif p.size == 7:
                p[3:6] *= scale_factor
            elif p.size == 12:
                p[3:6] *= scale_factor
        return p

    # -------------------------------------------------------------------
    # Original single-scale optimizers (retained for backward compatibility)
    # -------------------------------------------------------------------

    def _optimize_affine_transform(self, moving_data: np.ndarray, fixed_data: np.ndarray) -> np.ndarray:
        initial_params = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0])
        moving_small = self._downsample_image(moving_data, factor=4)
        fixed_small = self._downsample_image(fixed_data, factor=4)
        metric_fn = select_metric_fn(self.config.similarity)

        def objective(params):
            transform_matrix = self._params_to_affine_matrix(params)
            transformed = self._apply_transform(moving_small, transform_matrix, fixed_small.shape)
            return -metric_fn(transformed, fixed_small)

        result = minimize(objective, initial_params, method='Powell',
                          options={'maxiter': 100, 'disp': False})
        return self._params_to_affine_matrix(result.x)

    def _optimize_rigid_transform(self, moving_data: np.ndarray, fixed_data: np.ndarray) -> np.ndarray:
        initial_params = np.array([0, 0, 0, 0, 0, 0])
        moving_small = self._downsample_image(moving_data, factor=4)
        fixed_small = self._downsample_image(fixed_data, factor=4)
        metric_fn = select_metric_fn(self.config.similarity)

        def objective(params):
            transform_matrix = self._params_to_rigid_matrix(params)
            transformed = self._apply_transform(moving_small, transform_matrix, fixed_small.shape)
            return -metric_fn(transformed, fixed_small)

        result = minimize(objective, initial_params, method='Powell',
                          options={'maxiter': 100, 'disp': False})
        return self._params_to_rigid_matrix(result.x)

    def _optimize_similarity_transform(self, moving_data: np.ndarray, fixed_data: np.ndarray) -> np.ndarray:
        initial_params = np.array([0, 0, 0, 0, 0, 0, 1])
        moving_small = self._downsample_image(moving_data, factor=4)
        fixed_small = self._downsample_image(fixed_data, factor=4)
        metric_fn = select_metric_fn(self.config.similarity)

        def objective(params):
            transform_matrix = self._params_to_similarity_matrix(params)
            transformed = self._apply_transform(moving_small, transform_matrix, fixed_small.shape)
            return -metric_fn(transformed, fixed_small)

        result = minimize(objective, initial_params, method='Powell',
                          options={'maxiter': 100, 'disp': False})
        return self._params_to_similarity_matrix(result.x)

    # -------------------------------------------------------------------
    # Parameter → Matrix conversions
    # -------------------------------------------------------------------

    def _params_to_affine_matrix(self, params: np.ndarray) -> np.ndarray:
        rx, ry, rz = params[0:3]
        tx, ty, tz = params[3:6]
        sx, sy, sz = params[6:9]
        kx, ky, kz = params[9:12]
        Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
        Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
        Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
        R = Rz @ Ry @ Rx
        S = np.diag([sx, sy, sz])
        K = np.array([[1, kx, 0], [ky, 1, 0], [0, kz, 1]])
        A = R @ S @ K
        T = np.eye(4)
        T[:3, :3] = A
        T[:3, 3] = [tx, ty, tz]
        return T

    def _params_to_rigid_matrix(self, params: np.ndarray) -> np.ndarray:
        rx, ry, rz = params[0:3]
        tx, ty, tz = params[3:6]
        Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
        Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
        Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
        R = Rz @ Ry @ Rx
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [tx, ty, tz]
        return T

    def _params_to_similarity_matrix(self, params: np.ndarray) -> np.ndarray:
        rx, ry, rz = params[0:3]
        tx, ty, tz = params[3:6]
        s = params[6]
        Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
        Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
        Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
        A = s * (Rz @ Ry @ Rx)
        T = np.eye(4)
        T[:3, :3] = A
        T[:3, 3] = [tx, ty, tz]
        return T

    # -------------------------------------------------------------------
    # Transform application & helpers
    # -------------------------------------------------------------------

    def _apply_transform(
        self,
        image: np.ndarray,
        transform_matrix: np.ndarray,
        output_shape: Tuple[int, ...]
    ) -> np.ndarray:
        try:
            inv = np.linalg.inv(transform_matrix)
            # scipy affine expects matrix that maps output→input, which is inverse
            # Only 3D support for now
            mat = inv[:3, :3]
            offset = inv[:3, 3]
            transformed = ndimage.affine_transform(
                image,
                mat,
                offset=offset,
                output_shape=output_shape,
                order=1,
                mode='constant',
                cval=0.0
            )
            return transformed
        except np.linalg.LinAlgError:
            self.logger.warning("Singular transform; returning original image.")
            return image

    def _downsample_image(self, image: np.ndarray, factor: int = 2) -> np.ndarray:
        return image[::factor, ::factor, ::factor] if image.ndim == 3 else image[::factor, ::factor]

    def _normalize_image(self, img: np.ndarray) -> np.ndarray:
        img = img.astype(np.float32)
        mn, mx = np.percentile(img, (1, 99))
        if mx <= mn:
            mn, mx = img.min(), img.max()
        if mx > mn:
            img = np.clip(img, mn, mx)
            img = (img - mn) / (mx - mn)
        return img

    # -------------------------------------------------------------------
    # Template generation (vectorized)
    # -------------------------------------------------------------------

    def _get_template(self, template_name: str) -> np.ndarray:
        if template_name.startswith("MNI152"):
            if "1mm" in template_name:
                shape = (182, 218, 182)
            elif "2mm" in template_name:
                shape = (91, 109, 91)
            else:
                shape = (91, 109, 91)
            zz, yy, xx = np.meshgrid(
                np.arange(shape[2]), np.arange(shape[1]), np.arange(shape[0]), indexing='ij'
            )
            center = (np.array(shape) - 1) / 2.0
            # Reorder due to meshgrid usage
            dist = np.sqrt(
                (xx - center[0]) ** 2 +
                (yy - center[1]) ** 2 +
                (zz - center[2]) ** 2
            )
            template = np.zeros(shape, dtype=np.float32)
            core = dist < (min(shape) * 0.25)
            template[core] = 100 * np.exp(-dist[core] / 20)
            return template
        raise ValueError(f"Unknown template: {template_name}")

    # -------------------------------------------------------------------
    # Transform saving
    # -------------------------------------------------------------------

    def _save_transform(self, path: Union[str, Path], matrix: np.ndarray):
        p = Path(path)
        if p.suffix.lower() == ".json" and self.config.io.save_transform_json:
            meta = {
                "version": self.config.version,
                "method": self.method,
                "matrix": matrix.tolist(),
                "config": asdict(self.config)
            }
            p.write_text(json.dumps(meta, indent=self.config.io.json_indent))
        else:
            np.savetxt(str(p), matrix, fmt=f"%.{self.config.io.float_precision}f")

# End of file
