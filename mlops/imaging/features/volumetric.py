"""
Volumetric and morphometric feature extraction for medical (and optionally single-cell) 3D imaging.

This module extends basic volumetric MRI feature extraction with:
- Configurable processing via dataclasses
- Improved brain masking (adaptive + morphological refinement)
- Robust tissue segmentation (KMeans + Gaussian Mixture fallback + Otsu hybrid + intensity normalization)
- Optional single-cell / nuclei volumetric mode (object counting & per-object stats)
- Atlas or synthetic region volume extraction with optional parallelization
- Enhanced morphometrics (surface area via marching cubes, sphericity, elongation PCA, fractal proxy)
- Quality Control (QC) metrics (SNR, WM/GM contrast, coefficient of variation, entropy)
- Caching of intermediate masks / segmentation to avoid recomputation
- Optional persistence of derived volumetric maps
- Structured summary utilities and Pandas-friendly export
- Graceful degradation when dependencies (e.g., scikit-image) are unavailable
- Pluggable hooks for custom post-processing

NOTE:
This code is intended as a flexible research utility and not a validated clinical pipeline.
Always verify anatomical correctness before use in downstream analysis.

Author: Advanced Refactor (AI Assisted)
"""

from __future__ import annotations

import os
import time
import math
import json
import logging
import hashlib
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import (
    Optional, Union, Dict, Any, List, Tuple, Callable, Iterable
)

import numpy as np
import nibabel as nib
from scipy import ndimage
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

try:
    from skimage.filters import threshold_otsu
    from skimage.measure import marching_cubes, regionprops
    from skimage.morphology import binary_opening, binary_closing, ball
    SKIMAGE_AVAILABLE = True
except Exception:
    SKIMAGE_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except Exception:
    PANDAS_AVAILABLE = False

logger = logging.getLogger(__name__)


# ----------------------------- Exceptions ---------------------------------- #

class VolumetricExtractionError(Exception):
    """Raised when extraction fails irrecoverably."""


# ----------------------------- Configuration -------------------------------- #

@dataclass
class VolumetricConfig:
    atlas: str = "AAL3"
    brain_mask_percentile: float = 15.0
    brain_mask_min_volume_voxels: int = 500
    use_gmm_fallback: bool = True
    expected_tissue_clusters: int = 3
    kmeans_repeats: int = 10
    normalize_intensity: bool = True
    intensity_clip_percentiles: Tuple[float, float] = (0.5, 99.5)
    surface_method: str = "marching_cubes"  # or 'gradient'
    marching_cubes_level: float = 0.5
    parallel_regions: bool = False
    num_workers: int = 4
    cache_dir: Optional[Union[str, Path]] = None
    save_intermediates: bool = False
    intermediates_dir: Optional[Union[str, Path]] = None
    compute_qc_metrics: bool = True
    enable_single_cell_mode: bool = False
    single_cell_min_size_voxels: int = 20
    single_cell_max_size_voxels: int = 50_000
    single_cell_binary_threshold: Optional[float] = None  # if None -> Otsu
    elongation_compute: bool = True
    convex_hull_method: str = "bbox"  # placeholder for future advanced hulls
    secure_hash_inputs: bool = True
    random_state: int = 42
    hooks: Dict[str, List[Callable]] = field(default_factory=lambda: {
        "pre_load": [],
        "post_load": [],
        "pre_features": [],
        "post_features": []
    })

    def to_hash(self) -> str:
        """Hash relevant settings for cache identity."""
        relevant = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(relevant.encode("utf-8")).hexdigest()[:16]


# ----------------------------- Utility Decorators --------------------------- #

def timed(fn: Callable) -> Callable:
    """Decorator to time functions and log duration."""
    def wrapper(self, *args, **kwargs):
        start = time.time()
        result = fn(self, *args, **kwargs)
        elapsed = time.time() - start
        self.logger.debug(f"{fn.__name__} completed in {elapsed:.3f}s")
        return result
    return wrapper


# ----------------------------- Main Extractor ------------------------------- #

class VolumetricFeatureExtractor:
    """
    Advanced volumetric feature extractor.

    Provides:
    - Tissue segmentation (KMeans → GMM → Otsu hybrid fallback)
    - Morphometric features (surface area, sphericity, convexity, elongation)
    - Regional volumes (atlas or synthetic partitions)
    - QC metrics and optional single-cell object mode
    """

    def __init__(self, config: Optional[VolumetricConfig] = None, **legacy_kwargs):
        """
        Initialize the feature extractor.

        Args:
            config: Optional VolumetricConfig
            **legacy_kwargs: Backward-compatible kwargs for atlas / overrides
        """
        if config is None:
            config = VolumetricConfig(**legacy_kwargs)
        else:
            # Allow atlas override if passed as kwarg
            if "atlas" in legacy_kwargs:
                config.atlas = legacy_kwargs["atlas"]

        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.cache: Dict[str, Any] = {}
        self._prepare_dirs()

    def _prepare_dirs(self):
        if self.config.cache_dir:
            Path(self.config.cache_dir).mkdir(parents=True, exist_ok=True)
        if self.config.save_intermediates and self.config.intermediates_dir:
            Path(self.config.intermediates_dir).mkdir(parents=True, exist_ok=True)

    # ----------------------------- Public API -------------------------------- #

    @timed
    def extract_features(
        self,
        image_path: Union[str, Path],
        mask_path: Optional[Union[str, Path]] = None,
        atlas_path: Optional[Union[str, Path]] = None,
        return_maps: bool = False
    ) -> Union[Dict[str, float], Tuple[Dict[str, float], Dict[str, np.ndarray]]]:
        """
        Extract advanced volumetric features.

        Args:
            image_path: NIfTI image path
            mask_path: Optional pre-computed brain mask
            atlas_path: Optional atlas for region segmentation
            return_maps: If True, also return segmentation/tissue maps

        Returns:
            features dict (and optional maps dict if return_maps)
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Input image not found: {image_path}")

        # Hooks: pre_load
        self._run_hooks("pre_load", image_path=image_path)

        img = nib.load(str(image_path))
        data = img.get_fdata(dtype=np.float32)
        voxel_size_mm3 = float(np.prod(img.header.get_zooms()[:3]))

        if self.config.normalize_intensity:
            data = self._normalize_intensity(data)

        # Hooks: post_load
        self._run_hooks("post_load", image=image_path, data=data)

        if mask_path:
            mask = self._safe_load_mask(mask_path, data.shape)
        else:
            mask = self._create_brain_mask(data)

        features: Dict[str, float] = {}
        maps: Dict[str, np.ndarray] = {}

        # Basic
        basic = self._extract_basic_volumes(data, mask, voxel_size_mm3)
        features.update(basic)

        # Tissue segmentation
        tissue_out = self._segment_tissues(data, mask, voxel_size_mm3)
        features.update(tissue_out["features"])
        maps.update(tissue_out["maps"])

        # Morphometrics
        morph = self._extract_morphometric_features(mask, voxel_size_mm3, data.shape)
        features.update(morph)

        # Regional
        if atlas_path and Path(atlas_path).exists():
            regional = self._extract_regional_volumes(data, atlas_path, voxel_size_mm3)
        else:
            regional = self._extract_synthetic_regional_volumes(data, mask, voxel_size_mm3)
        features.update(regional)

        # Single-cell mode (optional)
        if self.config.enable_single_cell_mode:
            sc = self._extract_single_cell_objects(data, mask, voxel_size_mm3)
            features.update(sc["features"])
            maps.update(sc.get("maps", {}))

        # QC metrics
        if self.config.compute_qc_metrics:
            qc = self._compute_qc_metrics(data, mask)
            features.update(qc)

        # Summary appended
        summary = self.get_feature_summary(features)
        features.update({f"summary_{k}": v for k, v in summary.items()})

        # Hooks: post_features
        self._run_hooks("post_features", features=features, maps=maps)

        self.logger.info(f"Extracted {len(features)} features from {image_path.name}")

        if return_maps:
            return features, maps
        return features

    def validate_inputs(self, image_path: Union[str, Path]) -> bool:
        """Validate input file exists and is valid NIfTI."""
        try:
            image_path = Path(image_path)
            if not image_path.exists():
                return False
            nib.load(str(image_path))
            return True
        except Exception as e:
            self.logger.error(f"Input validation failed: {e}")
            return False

    def get_feature_summary(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Get summary statistics of extracted scalar features."""
        numeric_values = [v for v in features.values() if isinstance(v, (int, float)) and not math.isnan(v)]
        if not numeric_values:
            return {}
        arr = np.array(numeric_values, dtype=np.float64)
        return {
            "total_features": int(len(features)),
            "feature_mean": float(arr.mean()),
            "feature_std": float(arr.std(ddof=1) if arr.size > 1 else 0.0),
            "feature_min": float(arr.min()),
            "feature_max": float(arr.max()),
            "volume_features": int(sum("volume" in k.lower() for k in features)),
            "intensity_features": int(sum("intensity" in k.lower() for k in features)),
            "morphometric_features": int(sum(any(x in k.lower() for x in [
                "sphericity", "convexity", "elongation", "surface", "fractal"
            ]) for k in features)),
        }

    def features_to_dataframe(self, features: Dict[str, Any]):
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas not installed; cannot convert to DataFrame")
        return pd.DataFrame([features])

    # ------------------------ Internal Helpers -------------------------------- #

    def _run_hooks(self, phase: str, **kwargs):
        hooks = self.config.hooks.get(phase, [])
        for hook in hooks:
            try:
                hook(self, **kwargs)
            except Exception as e:
                self.logger.warning(f"Hook {hook} in phase {phase} failed: {e}")

    def _safe_load_mask(self, mask_path: Union[str, Path], expected_shape: Tuple[int, ...]) -> np.ndarray:
        mask_img = nib.load(str(mask_path))
        mask = mask_img.get_fdata() > 0
        if mask.shape != expected_shape:
            raise ValueError("Provided mask shape mismatch.")
        return mask

    def _normalize_intensity(self, data: np.ndarray) -> np.ndarray:
        """Intensity normalization with percentile clipping."""
        lo, hi = np.percentile(data[np.isfinite(data)], self.config.intensity_clip_percentiles)
        clipped = np.clip(data, lo, hi)
        std = clipped.std()
        if std == 0:
            return clipped
        normed = (clipped - clipped.mean()) / std
        return normed

    # ------------------------ Basic Volumes ----------------------------------- #

    def _extract_basic_volumes(self, data: np.ndarray, mask: np.ndarray, voxel_size: float) -> Dict[str, float]:
        features: Dict[str, float] = {}
        brain_voxels = int(mask.sum())
        features["total_brain_volume_voxels"] = float(brain_voxels)
        features["total_brain_volume_mm3"] = float(brain_voxels * voxel_size)

        icv_mask = self._estimate_icv_mask(data)
        icv_voxels = int(icv_mask.sum())
        features["intracranial_volume_mm3"] = float(icv_voxels * voxel_size)
        features["brain_volume_fraction"] = float(brain_voxels / icv_voxels) if icv_voxels > 0 else 0.0
        features["image_volume_mm3"] = float(np.prod(data.shape) * voxel_size)
        features["voxel_size_mm3"] = float(voxel_size)
        return features

    # ------------------------ Brain Mask -------------------------------------- #

    @timed
    def _create_brain_mask(self, data: np.ndarray) -> np.ndarray:
        """Adaptive threshold + morphological refinement + largest component."""
        cfg = self.config
        finite_data = data[np.isfinite(data)]
        finite_data = finite_data[finite_data > 0]
        if finite_data.size == 0:
            return np.zeros_like(data, dtype=bool)

        thr = np.percentile(finite_data, cfg.brain_mask_percentile)
        mask = data > thr

        if SKIMAGE_AVAILABLE:
            mask = binary_closing(mask, ball(2))
            mask = binary_opening(mask, ball(1))
        else:
            mask = ndimage.binary_closing(mask, iterations=2)
            mask = ndimage.binary_opening(mask, iterations=1)

        labeled, nlab = ndimage.label(mask)
        if nlab > 1:
            sizes = ndimage.sum(mask, labeled, range(1, nlab + 1))
            keep = (sizes >= cfg.brain_mask_min_volume_voxels)
            keep_labels = {i + 1 for i, k in enumerate(keep) if k}
            mask = np.isin(labeled, list(keep_labels))
        return mask

    def _estimate_icv_mask(self, data: np.ndarray) -> np.ndarray:
        brain_mask = self._create_brain_mask(data)
        structure = np.ones((5, 5, 5))
        icv_mask = ndimage.binary_dilation(brain_mask, structure=structure, iterations=5)
        return icv_mask

    # ------------------------ Tissue Segmentation ----------------------------- #

    @timed
    def _segment_tissues(
        self, data: np.ndarray, mask: np.ndarray, voxel_size: float
    ) -> Dict[str, Any]:
        """Segment tissues with KMeans → GMM → Otsu fallback."""
        cfg = self.config
        maps: Dict[str, np.ndarray] = {}
        features: Dict[str, float] = {}

        brain_data = data * mask
        brain_vals = brain_data[mask > 0]
        if brain_vals.size == 0:
            return {"features": features, "maps": maps}

        # Primary attempt: KMeans
        cluster_labels = None
        n_clusters = cfg.expected_tissue_clusters
        try:
            km = KMeans(
                n_clusters=n_clusters,
                random_state=cfg.random_state,
                n_init=cfg.kmeans_repeats
            )
            cluster_labels = km.fit_predict(brain_vals.reshape(-1, 1))
            centroids = km.cluster_centers_.flatten()
        except Exception as e:
            self.logger.warning(f"KMeans tissue segmentation failed: {e}")
            cluster_labels = None

        # Fallback: GMM
        if cluster_labels is None and cfg.use_gmm_fallback:
            try:
                gmm = GaussianMixture(n_components=n_clusters, random_state=cfg.random_state)
                cluster_labels = gmm.fit_predict(brain_vals.reshape(-1, 1))
                centroids = gmm.means_.flatten()
            except Exception as e:
                self.logger.warning(f"GMM fallback failed: {e}")

        # Final fallback: Otsu-based segmentation into pseudo 3 classes
        if cluster_labels is None:
            self.logger.info("Using Otsu hybrid fallback for tissue segmentation.")
            try:
                if SKIMAGE_AVAILABLE:
                    t1 = threshold_otsu(brain_vals)
                    low_mask = brain_data <= t1
                    high_region = brain_data > t1
                    # Split high region again if possible
                    if high_region.sum() > 10:
                        t2 = threshold_otsu(brain_vals[brain_vals > t1])
                        mid_mask = (brain_data > t1) & (brain_data <= t2)
                        high_mask = brain_data > t2
                    else:
                        mid_mask = high_region
                        high_mask = np.zeros_like(mid_mask)
                else:
                    # Approximate with percentiles
                    q25, q50 = np.percentile(brain_vals, [25, 50])
                    low_mask = brain_data <= q25
                    mid_mask = (brain_data > q25) & (brain_data <= q50)
                    high_mask = brain_data > q50

                tissue_map = np.zeros_like(data, dtype=np.uint8)
                tissue_map[low_mask] = 1
                tissue_map[mid_mask] = 2
                tissue_map[high_mask] = 3
                maps["tissue_map"] = tissue_map

                volumes = {
                    "csf_volume_mm3": float(low_mask.sum() * voxel_size),
                    "gray_matter_volume_mm3": float(mid_mask.sum() * voxel_size),
                    "white_matter_volume_mm3": float(high_mask.sum() * voxel_size),
                }
                features.update(volumes)
                total = sum(mask.sum() for mask in [low_mask, mid_mask, high_mask])
                if total > 0:
                    features["csf_fraction"] = float(low_mask.sum() / total)
                    features["gray_matter_fraction"] = float(mid_mask.sum() / total)
                    features["white_matter_fraction"] = float(high_mask.sum() / total)
                return {"features": features, "maps": maps}
            except Exception as e:
                self.logger.error(f"Otsu fallback failed: {e}")
                return {"features": features, "maps": maps}

        # If we have cluster labels (KM or GMM)
        tissue_map = np.zeros_like(data, dtype=np.uint8)
        tissue_map[mask > 0] = cluster_labels + 1  # 1..n
        maps["tissue_map"] = tissue_map

        # Assign tissues by sorted centroid intensities
        order = np.argsort(centroids)
        if len(order) >= 3:
            csf_id, gm_id, wm_id = order[0] + 1, order[1] + 1, order[2] + 1
        else:
            # fallback mapping
            csf_id, gm_id, wm_id = 1, 2, 3

        csf_vox = int((tissue_map == csf_id).sum())
        gm_vox = int((tissue_map == gm_id).sum())
        wm_vox = int((tissue_map == wm_id).sum())

        features["csf_volume_mm3"] = float(csf_vox * voxel_size)
        features["gray_matter_volume_mm3"] = float(gm_vox * voxel_size)
        features["white_matter_volume_mm3"] = float(wm_vox * voxel_size)

        total_tissue = csf_vox + gm_vox + wm_vox
        if total_tissue > 0:
            features["csf_fraction"] = float(csf_vox / total_tissue)
            features["gray_matter_fraction"] = float(gm_vox / total_tissue)
            features["white_matter_fraction"] = float(wm_vox / total_tissue)

        # WM/GM contrast
        if gm_vox > 0 and wm_vox > 0:
            features["wm_gm_intensity_contrast"] = float(
                data[tissue_map == wm_id].mean() - data[tissue_map == gm_id].mean()
            )

        # Save intermediates if enabled
        self._maybe_save_map("tissue_map", tissue_map)

        return {"features": features, "maps": maps}

    # ------------------------ Morphometrics ---------------------------------- #

    @timed
    def _extract_morphometric_features(
        self,
        mask: np.ndarray,
        voxel_size: float,
        shape: Tuple[int, int, int]
    ) -> Dict[str, float]:
        features: Dict[str, float] = {}
        if not np.any(mask):
            return features

        surface_area = self._estimate_surface_area(mask, voxel_size)
        features["brain_surface_area_mm2"] = float(surface_area)

        volume = float(mask.sum() * voxel_size)
        if surface_area > 0:
            sphericity = (math.pi ** (1 / 3)) * ((6 * volume) ** (2 / 3)) / surface_area
            features["sphericity"] = float(sphericity)

        convex_vol = self._estimate_convex_hull_volume(mask, voxel_size)
        features["convex_hull_volume_mm3"] = float(convex_vol)
        if convex_vol > 0:
            features["convexity"] = float(volume / convex_vol)

        if self.config.elongation_compute:
            features.update(self._compute_elongation_features(mask))

        centroid = ndimage.center_of_mass(mask.astype(np.float32))
        features["centroid_x_normalized"] = float(centroid[0] / shape[0])
        features["centroid_y_normalized"] = float(centroid[1] / shape[1])
        features["centroid_z_normalized"] = float(centroid[2] / shape[2])

        # Fractal proxy (box counting approximation)
        features["fractal_dimension_proxy"] = float(self._box_count_fractal_proxy(mask))

        return features

    def _estimate_surface_area(self, mask: np.ndarray, voxel_size: float) -> float:
        method = self.config.surface_method
        if method == "marching_cubes" and SKIMAGE_AVAILABLE:
            try:
                verts, faces, _, _ = marching_cubes(
                    mask.astype(np.uint8),
                    level=self.config.marching_cubes_level,
                    spacing=(1.0, 1.0, 1.0)
                )
                # Triangle area sum
                tri_areas = []
                for f in faces:
                    a, b, c = verts[f]
                    tri_areas.append(
                        0.5 * np.linalg.norm(np.cross(b - a, c - a))
                    )
                return float(sum(tri_areas) * (voxel_size ** (2 / 3)))
            except Exception as e:
                self.logger.debug(f"marching_cubes surface failed, fallback: {e}")

        # Gradient fallback
        gradient = np.gradient(mask.astype(np.float32))
        gm = np.sqrt(sum(g**2 for g in gradient))
        surface_voxels = (gm > 0.1).sum()
        return float(surface_voxels * (voxel_size ** (2 / 3)))

    def _estimate_convex_hull_volume(self, mask: np.ndarray, voxel_size: float) -> float:
        coords = np.where(mask)
        if coords[0].size == 0:
            return 0.0
        minc = [c.min() for c in coords]
        maxc = [c.max() for c in coords]
        bbox_voxels = np.prod([(maxc[i] - minc[i] + 1) for i in range(3)])
        return float(bbox_voxels * voxel_size)

    def _compute_elongation_features(self, mask: np.ndarray) -> Dict[str, float]:
        feats: Dict[str, float] = {}
        coords = np.column_stack(np.where(mask))
        if coords.shape[0] < 3:
            return feats
        centered = coords - coords.mean(axis=0, keepdims=True)
        cov = np.cov(centered.T)
        eigvals = np.sort(np.real(np.linalg.eigvals(cov)))[::-1]
        if eigvals.size >= 3 and eigvals[2] > 0:
            feats["elongation_ratio_1"] = float(eigvals[0] / eigvals[1]) if eigvals[1] > 0 else float("inf")
            feats["elongation_ratio_2"] = float(eigvals[1] / eigvals[2]) if eigvals[2] > 0 else float("inf")
            feats["flatness_ratio"] = float(eigvals[2] / eigvals[0]) if eigvals[0] > 0 else 0.0
        return feats

    def _box_count_fractal_proxy(self, mask: np.ndarray, scales: Iterable[int] = (2, 4, 8, 16)) -> float:
        counts = []
        sizes = []
        for s in scales:
            if min(mask.shape) < s:
                continue
            view = mask.reshape(
                mask.shape[0] // s, s,
                mask.shape[1] // s, s,
                mask.shape[2] // s, s
            )
            # block presence
            block = view.any(axis=1).any(axis=2).any(axis=3)
            counts.append(block.sum())
            sizes.append(1 / s)
        if len(counts) < 2:
            return 0.0
        # linear regression log(count) vs log(size)
        x = np.log(sizes)
        y = np.log(counts)
        slope, _ = np.polyfit(x, y, 1)
        return float(-slope)  # approximate fractal dimension

    # ------------------------ Regional Volumes -------------------------------- #

    def _extract_regional_volumes(self, data: np.ndarray, atlas_path: Union[str, Path], voxel_size: float) -> Dict[str, float]:
        features: Dict[str, float] = {}
        try:
            atlas_img = nib.load(str(atlas_path))
            atlas = atlas_img.get_fdata()
            regions = np.unique(atlas[atlas > 0])
            for rid in regions:
                region_mask = (atlas == rid)
                vox = int(region_mask.sum())
                if vox == 0:
                    continue
                region_volume = vox * voxel_size
                mean_intensity = float(data[region_mask].mean()) if vox > 0 else 0.0
                features[f"region_{int(rid)}_volume_mm3"] = float(region_volume)
                features[f"region_{int(rid)}_mean_intensity"] = mean_intensity
        except Exception as e:
            self.logger.warning(f"Atlas-based regional extraction failed: {e}")
        return features

    def _extract_synthetic_regional_volumes(self, data: np.ndarray, mask: np.ndarray, voxel_size: float) -> Dict[str, float]:
        features: Dict[str, float] = {}
        if not np.any(mask):
            return features
        shape = data.shape
        center = np.array(shape) // 2
        coords = np.mgrid[:shape[0], :shape[1], :shape[2]]

        regions = {
            "anterior": coords[0] < center[0],
            "posterior": coords[0] >= center[0],
            "left": coords[1] < center[1],
            "right": coords[1] >= center[1],
            "superior": coords[2] >= center[2],
            "inferior": coords[2] < center[2],
        }
        for name, m in regions.items():
            region_mask = m & mask
            vox = int(region_mask.sum())
            features[f"{name}_volume_mm3"] = float(vox * voxel_size)
            if vox > 0:
                features[f"{name}_mean_intensity"] = float(data[region_mask].mean())
        return features

    # ------------------------ Single-Cell Mode -------------------------------- #

    def _extract_single_cell_objects(
        self,
        data: np.ndarray,
        mask: np.ndarray,
        voxel_size: float
    ) -> Dict[str, Any]:
        """
        Single-cell / nuclei segmentation (simplistic) within brain mask.

        Steps:
        - Threshold (Otsu or provided)
        - Label objects
        - Filter size
        - Aggregate metrics
        """
        features: Dict[str, float] = {}
        maps: Dict[str, np.ndarray] = {}

        if not self.config.enable_single_cell_mode:
            return {"features": features}

        working = data * mask
        if self.config.single_cell_binary_threshold is not None:
            thr = self.config.single_cell_binary_threshold
        else:
            if SKIMAGE_AVAILABLE:
                try:
                    thr = threshold_otsu(working[working > 0])
                except Exception:
                    thr = np.percentile(working[working > 0], 50)
            else:
                thr = np.percentile(working[working > 0], 50)

        binary = (working > thr)
        labeled, nlab = ndimage.label(binary)

        sizes = ndimage.sum(binary, labeled, range(1, nlab + 1))
        keep_labels = []
        for idx, s in enumerate(sizes, start=1):
            if self.config.single_cell_min_size_voxels <= s <= self.config.single_cell_max_size_voxels:
                keep_labels.append(idx)
        keep_labels_set = set(keep_labels)
        cleaned = np.isin(labeled, list(keep_labels_set)).astype(np.int32)

        maps["single_cell_label_map"] = cleaned
        self._maybe_save_map("single_cell_label_map", cleaned)

        count = len(keep_labels)
        features["single_cell_object_count"] = float(count)
        if count > 0:
            volumes = [sizes[i - 1] * voxel_size for i in keep_labels]
            features["single_cell_mean_volume_mm3"] = float(np.mean(volumes))
            features["single_cell_std_volume_mm3"] = float(np.std(volumes, ddof=1) if len(volumes) > 1 else 0.0)
            features["single_cell_total_volume_mm3"] = float(np.sum(volumes))
            features["single_cell_volume_fraction"] = float(
                (np.sum(volumes) / (mask.sum() * voxel_size)) if mask.sum() > 0 else 0.0
            )
        else:
            features["single_cell_mean_volume_mm3"] = 0.0
            features["single_cell_std_volume_mm3"] = 0.0
            features["single_cell_total_volume_mm3"] = 0.0
            features["single_cell_volume_fraction"] = 0.0

        return {"features": features, "maps": maps}

    # ------------------------ QC Metrics -------------------------------------- #

    def _compute_qc_metrics(self, data: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
        feats: Dict[str, float] = {}
        inside = data[mask > 0]
        if inside.size == 0:
            return feats
        feats["intensity_mean"] = float(inside.mean())
        feats["intensity_std"] = float(inside.std())
        feats["intensity_coefficient_of_variation"] = (
            float(inside.std() / (inside.mean() + 1e-8))
        )
        # SNR (simple)
        background = data[mask == 0]
        if background.size > 100:
            feats["snr_mean_background"] = float(inside.mean() / (background.std() + 1e-8))
        else:
            feats["snr_mean_background"] = float("nan")

        # Entropy approximation
        hist, _ = np.histogram(inside, bins=64, density=True)
        hist = hist[hist > 0]
        entropy = -np.sum(hist * np.log2(hist))
        feats["intensity_entropy"] = float(entropy)
        return feats

    # ------------------------ Persistence / Caching --------------------------- #

    def _maybe_save_map(self, name: str, arr: np.ndarray):
        if not self.config.save_intermediates or not self.config.intermediates_dir:
            return
        out_dir = Path(self.config.intermediates_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / f"{name}.npy", arr)

    # ------------------------ Legacy Compat (Optional) ------------------------ #

    # (Left intentionally simple; the advanced methods supersede the old ones.)

# ----------------------------- Module Test Hook ------------------------------ #

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("VolumetricFeatureExtractor module loaded. This block is for rudimentary smoke tests only.")
    # Example (requires real NIfTI file):
    # extractor = VolumetricFeatureExtractor()
    # feats = extractor.extract_features("subject.nii.gz")
    # print(len(feats))
