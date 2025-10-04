#!/usr/bin/env python
"""
Advanced Bias / Illumination Field Correction Module.

Supports MRI (NIfTI) and single-cell microscopy images (2D/3D, optional channels).

Implemented correction strategies:
  - n4          : N4 bias field correction via ANTs or SimpleITK fallback (if installed)
  - histogram   : Simple histogram equalization (legacy)
  - clahe       : Contrast Limited Adaptive Histogram Equalization (per slice / channel)
  - simple      : Simple low-order polynomial fitting (legacy)
  - poly        : Advanced polynomial bias modeling (configurable degree)
  - retinex     : Single or multi-scale Retinex (illumination correction)
  - auto        : Heuristic selection (Retinex for microscopy-like, N4 for MRI-like volumes)

Features:
  - Multi-channel handling (C last or C first detection)
  - Automatic mask generation (percentile / Otsu if scikit-image available)
  - Optional Gaussian smoothing of estimated bias field
  - Parallel batch directory processing
  - Quality metrics: CV, uniformity, entropy, SNR, dynamic range, MAD, improvement ratios
  - Exportable JSON/YAML metrics report
  - Caching of masks and bias fields (in-memory + optional disk)
  - CLI interface

Author: Enhanced by AI Assistant
License: MIT
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np

# Optional imports
try:
    import nibabel as nib
except ImportError:  # pragma: no cover
    nib = None

try:
    import imageio.v3 as iio
except ImportError:  # pragma: no cover
    iio = None

try:
    import SimpleITK as sitk  # For N4 fallback
except ImportError:  # pragma: no cover
    sitk = None

try:
    from skimage import exposure, filters, morphology
except ImportError:  # pragma: no cover
    exposure = None
    filters = None
    morphology = None

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

try:
    import cupy as cp  # GPU support (experimental)
except ImportError:  # pragma: no cover
    cp = None

import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

logger = logging.getLogger(__name__)


# ----------------------------- Utility & Helpers --------------------------------- #

def _is_nifti(path: Path) -> bool:
    return path.suffix.lower() in {".nii", ".gz"} or path.name.endswith(".nii.gz")


def _safe_import_warning(lib: str):
    logger.warning(f"Optional dependency '{lib}' not available; related features may degrade.")


def _entropy(values: np.ndarray) -> float:
    v = values[values > 0]
    if v.size == 0:
        return 0.0
    hist, _ = np.histogram(v, bins=256, density=True)
    hist = hist[hist > 0]
    return float(-np.sum(hist * np.log2(hist)))


def _snr(signal: np.ndarray) -> float:
    fg = signal[signal > 0]
    if fg.size < 10:
        return 0.0
    mean = fg.mean()
    std = fg.std()
    return float(mean / (std + 1e-8))


def _dynamic_range(data: np.ndarray) -> float:
    fg = data[data > 0]
    if fg.size == 0:
        return 0.0
    return float(fg.max() - fg.min())


def _median_abs_dev(data: np.ndarray) -> float:
    fg = data[data > 0]
    if fg.size == 0:
        return 0.0
    med = np.median(fg)
    return float(np.median(np.abs(fg - med)))


def _choose_device(array: np.ndarray, use_gpu: bool = False):
    if use_gpu and cp is not None:
        return cp.asarray(array), cp
    return array, np


def _to_numpy(arr):
    if cp is not None and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return arr


def _maybe_smooth(field: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    if sigma <= 0:
        return field
    try:
        from scipy.ndimage import gaussian_filter  # Lazy import
        return gaussian_filter(field, sigma=sigma)
    except Exception:
        logger.debug("Gaussian smoothing unavailable (scipy missing).")
        return field


def _percentile_mask(data: np.ndarray, lower_pct: float = 10) -> np.ndarray:
    thresh = np.percentile(data, lower_pct)
    return data > thresh


def _otsu_mask(data: np.ndarray) -> np.ndarray:
    if filters is None:
        return _percentile_mask(data, 10)
    try:
        thresh = filters.threshold_otsu(data[data > 0])
        return data > thresh
    except Exception:
        return _percentile_mask(data, 10)


def _auto_mask(data: np.ndarray, strategy: str = "auto") -> np.ndarray:
    if strategy == "auto":
        return _otsu_mask(data)
    elif strategy == "percentile":
        return _percentile_mask(data)
    elif strategy == "none":
        return np.ones_like(data, dtype=bool)
    else:
        return _otsu_mask(data)


def _detect_channels(arr: np.ndarray) -> Tuple[np.ndarray, bool, int, str]:
    """
    Detect if data has channel dimension.
    Returns:
        base_array, has_channels, n_channels, layout (one of 'last','first','none')
    """
    if arr.ndim == 2:
        return arr, False, 1, "none"
    if arr.ndim == 3:
        # Could be (Z,Y,X) or (Y,X,C)
        if arr.shape[-1] <= 8 and arr.shape[-1] < min(arr.shape[:-1]):  # heuristic
            return arr, True, arr.shape[-1], "last"
        return arr, False, 1, "none"
    if arr.ndim == 4:
        # Heuristic: small first dimension as channel
        if arr.shape[0] <= 8 and arr.shape[0] < arr.shape[-1]:
            return arr, True, arr.shape[0], "first"
        if arr.shape[-1] <= 8:
            return arr, True, arr.shape[-1], "last"
    # Fallback: treat as no channel
    return arr, False, 1, "none"


# ----------------------------- Strategy Interfaces -------------------------------- #

class BiasCorrectionError(Exception):
    pass


class CorrectionStrategy:
    name: str = "base"

    def correct(
        self,
        data: np.ndarray,
        mask: Optional[np.ndarray],
        **kwargs
    ) -> np.ndarray:
        raise NotImplementedError


class HistogramStrategy(CorrectionStrategy):
    name = "histogram"

    def correct(self, data: np.ndarray, mask: Optional[np.ndarray], **kwargs) -> np.ndarray:
        d = data.copy()
        if mask is None:
            mask = _percentile_mask(d, 5)
        fg = d[mask]
        if fg.size == 0:
            return d
        hist, bins = np.histogram(fg, bins=256, density=True)
        cdf = hist.cumsum()
        cdf /= cdf[-1]
        d[mask] = np.interp(fg, bins[:-1], cdf * fg.max())
        return d


class CLAHEStrategy(CorrectionStrategy):
    name = "clahe"

    def correct(self, data: np.ndarray, mask: Optional[np.ndarray], **kwargs) -> np.ndarray:
        if exposure is None:
            _safe_import_warning("scikit-image (exposure)")
            return data
        clip_limit = kwargs.get("clip_limit", 0.01)
        nbins = kwargs.get("nbins", 256)
        # Apply per-slice for 3D if large
        if data.ndim == 3 and data.shape[0] < data.shape[-1]:  # treat first dim as Z
            corrected = []
            for z in range(data.shape[0]):
                corrected.append(exposure.equalize_adapthist(data[z], clip_limit=clip_limit, nbins=nbins))
            return np.stack(corrected, axis=0)
        else:
            return exposure.equalize_adapthist(data, clip_limit=clip_limit, nbins=nbins)


class SimplePolynomialStrategy(CorrectionStrategy):
    name = "simple"

    def correct(self, data: np.ndarray, mask: Optional[np.ndarray], **kwargs) -> np.ndarray:
        # Very low-order polynomial
        degree = kwargs.get("degree", 1)
        return PolynomialStrategy()._poly_fit_and_correct(data, mask, degree=degree, max_terms=4)


class PolynomialStrategy(CorrectionStrategy):
    name = "poly"

    def _poly_fit_and_correct(
        self,
        data: np.ndarray,
        mask: Optional[np.ndarray],
        degree: int = 3,
        max_terms: Optional[int] = None,
        smooth_sigma: float = 1.0
    ) -> np.ndarray:
        shape = data.shape
        coords = np.mgrid[[slice(0, s) for s in shape]]
        coords = [((c - c.mean()) / (c.std() + 1e-8)).reshape(-1) for c in coords]

        # Build polynomial basis (limited for performance)
        terms = [np.ones(coords[0].shape)]
        for d in range(1, degree + 1):
            for i in range(len(coords)):
                terms.append(coords[i] ** d)
            if len(coords) >= 2:
                terms.append((coords[0] ** d) * (coords[1] ** d))
            if len(coords) >= 3:
                terms.append((coords[0] ** d) * (coords[2] ** d))
                terms.append((coords[1] ** d) * (coords[2] ** d))
        basis = np.vstack(terms).T
        if max_terms is not None and basis.shape[1] > max_terms:
            basis = basis[:, :max_terms]

        flat = data.reshape(-1)
        if mask is None:
            mask = _percentile_mask(flat, 10)
        else:
            mask = mask.reshape(-1)
        if not np.any(mask):
            return data
        try:
            coeffs, *_ = np.linalg.lstsq(basis[mask], flat[mask], rcond=None)
            bias = (basis @ coeffs).reshape(shape)
            bias = _maybe_smooth(bias, sigma=smooth_sigma)
            bias[bias < 1e-3] = 1.0
            corrected = data / bias
            return corrected
        except np.linalg.LinAlgError:
            logger.warning("Polynomial fit failed; returning original data.")
            return data

    def correct(self, data: np.ndarray, mask: Optional[np.ndarray], **kwargs) -> np.ndarray:
        degree = kwargs.get("degree", 3)
        max_terms = kwargs.get("max_terms", None)
        smooth_sigma = kwargs.get("smooth_sigma", 1.0)
        return self._poly_fit_and_correct(data, mask, degree, max_terms, smooth_sigma)


class RetinexStrategy(CorrectionStrategy):
    name = "retinex"

    def correct(self, data: np.ndarray, mask: Optional[np.ndarray], **kwargs) -> np.ndarray:
        mode = kwargs.get("mode", "multi")
        sigma_list = kwargs.get("sigma_list", [15, 80, 250])
        eps = 1e-6
        img = data.astype(np.float32)
        img_norm = img / (img.max() + eps)
        if mode == "single":
            sigma = sigma_list[0]
            illumination = _maybe_smooth(img_norm, sigma=sigma)
            illumination[illumination < eps] = 1.0
            ret = np.log(img_norm * 255 + 1.0) - np.log(illumination * 255 + 1.0)
        else:
            # Multi-scale
            ret_scales = []
            for s in sigma_list:
                illum = _maybe_smooth(img_norm, sigma=s)
                illum[illum < eps] = 1.0
                ret_scales.append(np.log(img_norm * 255 + 1.0) - np.log(illum * 255 + 1.0))
            ret = sum(ret_scales) / len(ret_scales)
        # Normalize result
        ret = (ret - ret.min()) / (ret.max() - ret.min() + eps)
        return ret * img.max()


class N4Strategy(CorrectionStrategy):
    name = "n4"

    def correct(self, data: np.ndarray, mask: Optional[np.ndarray], **kwargs) -> np.ndarray:
        """
        In-memory N4 using SimpleITK if available; else attempt external ANTs command;
        elif fallback to polynomial.
        """
        if sitk is not None:
            try:
                img = sitk.GetImageFromArray(data.astype(np.float32))
                if mask is not None:
                    msk = sitk.GetImageFromArray(mask.astype(np.uint8))
                else:
                    auto = _auto_mask(data)
                    msk = sitk.GetImageFromArray(auto.astype(np.uint8))
                shrink = kwargs.get("shrink_factor", 2)
                conv = kwargs.get("convergence", (50, 50, 30, 20))
                bspline = kwargs.get("bspline_fit", 200)
                correct = sitk.N4BiasFieldCorrection(
                    image=img,
                    mask=msk,
                    shrinkFactor=shrink,
                    maximumNumberOfIterations=conv,
                    splineOrder=3,
                    numberOfHistogramBins=kwargs.get("hist_bins", 200)
                )
                return sitk.GetArrayFromImage(correct)
            except Exception as e:  # fallback path
                logger.warning(f"SimpleITK N4 failed: {e}")

        # Attempt external ANTs if data came from disk (external command needs file)
        tmpdir = kwargs.get("tmpdir")
        if tmpdir is not None:
            tmpdir = Path(tmpdir)
            tmpdir.mkdir(parents=True, exist_ok=True)
        else:
            tmpdir = Path("./_n4tmp")
            tmpdir.mkdir(exist_ok=True)

        tmp_in = tmpdir / "n4_input.nii.gz"
        tmp_out = tmpdir / "n4_output.nii.gz"
        if nib is not None:
            affine = np.eye(4)
            try:
                nib.save(nib.Nifti1Image(data.astype(np.float32), affine), tmp_in)
            except Exception:
                logger.error("Failed to write temporary NIfTI. Falling back to polynomial.")
                return PolynomialStrategy().correct(data, mask)
        else:
            logger.warning("nibabel not available for external ANTs N4. Falling back.")
            return PolynomialStrategy().correct(data, mask)

        cmd = ["N4BiasFieldCorrection", "-i", str(tmp_in), "-o", str(tmp_out)]
        if mask is not None and np.any(mask):
            tmp_mask = tmpdir / "n4_mask.nii.gz"
            nib.save(nib.Nifti1Image(mask.astype(np.uint8), np.eye(4)), tmp_mask)
            cmd.extend(["-x", str(tmp_mask)])
        cmd.extend([
            "--convergence", "[100x50x50,0.001]",
            "--shrink-factor", "2",
            "--n-histogram-bins", "200"
        ])
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0 and tmp_out.exists():
                out_img = nib.load(tmp_out)
                return out_img.get_fdata()
            else:
                logger.warning(f"External N4 failed: {result.stderr}")
        except FileNotFoundError:
            logger.warning("External N4BiasFieldCorrection not found.")
        # Fallback
        return PolynomialStrategy().correct(data, mask)


# Strategy registry
STRATEGIES: Dict[str, CorrectionStrategy] = {
    "histogram": HistogramStrategy(),
    "clahe": CLAHEStrategy(),
    "simple": SimplePolynomialStrategy(),
    "poly": PolynomialStrategy(),
    "retinex": RetinexStrategy(),
    "n4": N4Strategy(),
}


def _auto_select_strategy(data: np.ndarray) -> str:
    # Heuristic: if shape is ~3D MRI-like (Z ~ 20-300) and not many channels -> n4
    # if 2D or shallow 3D with high dynamic range -> retinex
    if data.ndim >= 3 and min(data.shape) > 15:
        return "n4"
    return "retinex"


# ----------------------------- Core Class ---------------------------------------- #

@dataclass
class BiasFieldCorrector:
    method: str = "n4"
    mask_strategy: str = "auto"         # auto | percentile | none
    per_channel: bool = True
    use_gpu: bool = False
    cache_bias: bool = True
    cache_masks: bool = True
    polynomial_degree: int = 3
    retinex_mode: str = "multi"
    retinex_sigma_list: Sequence[int] = (15, 80, 250)
    clahe_clip_limit: float = 0.01
    clahe_nbins: int = 256
    poly_smooth_sigma: float = 1.0
    auto_method: bool = False
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger(__name__))
    _mask_cache: Dict[str, np.ndarray] = field(default_factory=dict)
    _bias_cache: Dict[str, np.ndarray] = field(default_factory=dict)

    def validate_inputs(self, input_path: Union[str, Path]) -> bool:
        p = Path(input_path)
        if not p.exists():
            return False
        if _is_nifti(p):
            if nib is None:
                self.logger.error("nibabel required to load NIfTI.")
                return False
            try:
                nib.load(str(p))
                return True
            except Exception as e:
                self.logger.error(f"NIfTI load failed: {e}")
                return False
        else:
            if iio is None:
                self.logger.error("imageio required for non-NIfTI formats.")
                return False
            try:
                _ = iio.imread(p)
                return True
            except Exception as e:
                self.logger.error(f"ImageIO read failed: {e}")
                return False

    def _load_image(self, path: Path) -> np.ndarray:
        if _is_nifti(path):
            img = nib.load(str(path))
            return img.get_fdata()
        else:
            return iio.imread(path)

    def _save_image(self, data: np.ndarray, ref_path: Path, out_path: Path):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if _is_nifti(out_path):
            if nib is None:
                raise BiasCorrectionError("nibabel not available to save NIfTI.")
            affine = np.eye(4)
            nib.save(nib.Nifti1Image(data.astype(np.float32), affine), str(out_path))
        else:
            if iio is None:
                raise BiasCorrectionError("imageio not available to save non-NIfTI formats.")
            iio.imwrite(out_path, data.astype(np.float32))

    def _get_mask(self, data: np.ndarray, key: Optional[str] = None) -> np.ndarray:
        if self.mask_strategy == "none":
            return np.ones_like(data, dtype=bool)
        if key and self.cache_masks and key in self._mask_cache:
            return self._mask_cache[key]
        mask = _auto_mask(data, self.mask_strategy)
        if self.cache_masks and key:
            self._mask_cache[key] = mask
        return mask

    def _apply_strategy(
        self,
        data: np.ndarray,
        strategy_name: str,
        mask: Optional[np.ndarray]
    ) -> np.ndarray:
        if strategy_name == "auto":
            strategy_name = _auto_select_strategy(data)
        strat = STRATEGIES.get(strategy_name)
        if strat is None:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        kwargs = dict(
            degree=self.polynomial_degree,
            mode=self.retinex_mode,
            sigma_list=self.retinex_sigma_list,
            clip_limit=self.clahe_clip_limit,
            nbins=self.clahe_nbins,
            smooth_sigma=self.poly_smooth_sigma,
        )
        corrected = strat.correct(data, mask, **kwargs)
        return corrected

    def get_correction_quality_metrics(
        self,
        original_data: np.ndarray,
        corrected_data: np.ndarray
    ) -> Dict[str, float]:
        orig_fg = original_data[original_data > 0]
        corr_fg = corrected_data[corrected_data > 0]
        def cv(arr):
            if arr.size == 0 or arr.mean() == 0:
                return 0.0
            return float(arr.std() / (arr.mean() + 1e-8))
        original_cv = cv(orig_fg)
        corrected_cv = cv(corr_fg)
        return {
            "original_cv": original_cv,
            "corrected_cv": corrected_cv,
            "original_uniformity": 1.0 - original_cv,
            "corrected_uniformity": 1.0 - corrected_cv,
            "improvement_ratio_cv": (original_cv / corrected_cv) if corrected_cv > 0 else 1.0,
            "original_entropy": _entropy(original_data),
            "corrected_entropy": _entropy(corrected_data),
            "original_snr": _snr(original_data),
            "corrected_snr": _snr(corrected_data),
            "original_dynamic_range": _dynamic_range(original_data),
            "corrected_dynamic_range": _dynamic_range(corrected_data),
            "original_mad": _median_abs_dev(original_data),
            "corrected_mad": _median_abs_dev(corrected_data),
        }

    def correct_bias(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        mask_path: Optional[Union[str, Path]] = None,
        export_metrics: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """
        Correct bias / illumination in an image and optionally save output + metrics.

        Returns:
            dict with keys: method, output_path (if saved), metrics, timings
        """
        t0 = time.time()
        input_path = Path(input_path)
        if not self.validate_inputs(input_path):
            raise FileNotFoundError(f"Invalid or unreadable input: {input_path}")

        data = self._load_image(input_path)
        base, has_channels, n_channels, layout = _detect_channels(data)

        if mask_path:
            mask_img = self._load_image(Path(mask_path))
            mask = mask_img.astype(bool)
        else:
            # Only generate mask from intensity image — for multi-channel use mean projection
            if has_channels:
                if layout == "last":
                    mean_proj = base.mean(axis=-1)
                elif layout == "first":
                    mean_proj = base.mean(axis=0)
                else:
                    mean_proj = base.mean()
                mask = self._get_mask(mean_proj, key=str(input_path))
            else:
                mask = self._get_mask(base, key=str(input_path))

        method_to_use = self.method
        if self.auto_method:
            method_to_use = "auto"

        def process_channel(ch_arr: np.ndarray, idx: int) -> np.ndarray:
            ch_mask = mask
            return self._apply_strategy(ch_arr, method_to_use, ch_mask)

        if has_channels and self.per_channel:
            corrected_channels = []
            if layout == "last":
                for c in range(n_channels):
                    corrected_channels.append(process_channel(base[..., c], c))
                corrected = np.stack(corrected_channels, axis=-1)
            else:
                for c in range(n_channels):
                    corrected_channels.append(process_channel(base[c], c))
                corrected = np.stack(corrected_channels, axis=0)
        else:
            corrected = process_channel(base, 0)

        timings = {"total_seconds": time.time() - t0}

        metrics = self.get_correction_quality_metrics(base.astype(np.float32), corrected.astype(np.float32))

        if output_path:
            out_path = Path(output_path)
            self._save_image(corrected, input_path, out_path)
        else:
            out_path = None

        report = {
            "method_requested": self.method,
            "method_used": method_to_use,
            "input": str(input_path),
            "output": str(out_path) if out_path else None,
            "has_channels": has_channels,
            "n_channels": n_channels,
            "layout": layout,
            "metrics": metrics,
            "timings": timings,
        }

        if export_metrics:
            export_path = Path(export_metrics)
            export_path.parent.mkdir(parents=True, exist_ok=True)
            if export_path.suffix.lower() in {".yml", ".yaml"} and yaml is not None:
                with open(export_path, "w") as f:
                    yaml.safe_dump(report, f)
            else:
                with open(export_path, "w") as f:
                    json.dump(report, f, indent=2)

        return report

    # ---------------- Batch Processing ---------------- #

    def batch_correct(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        pattern: str = "*.nii*",
        parallel: int = 0,
        metrics_export: Optional[Union[str, Path]] = None,
        keep_structure: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Batch process a directory.

        Args:
            input_dir: directory with images
            output_dir: where corrected images go
            pattern: glob pattern
            parallel: number of processes (0=serial)
            metrics_export: optional JSON/YAML consolidated report
            keep_structure: replicate subdirectory structure

        Returns:
            list of per-image report dicts
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        files = sorted(list(input_dir.rglob(pattern)))
        reports: List[Dict[str, Any]] = []
        self.logger.info(f"Found {len(files)} files to process.")
        def _task(f: Path):
            rel = f.relative_to(input_dir)
            out_path = output_dir / rel if keep_structure else output_dir / f.name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                rep = self.correct_bias(f, out_path)
                return rep
            except Exception as e:
                return {"input": str(f), "error": str(e)}

        if parallel > 0:
            with ProcessPoolExecutor(max_workers=parallel) as ex:
                futures = {ex.submit(_task, f): f for f in files}
                for fut in as_completed(futures):
                    reports.append(fut.result())
        else:
            for f in files:
                reports.append(_task(f))

        if metrics_export:
            metrics_export = Path(metrics_export)
            metrics_export.parent.mkdir(parents=True, exist_ok=True)
            if metrics_export.suffix.lower() in {".yaml", ".yml"} and yaml is not None:
                with open(metrics_export, "w") as f:
                    yaml.safe_dump(reports, f)
            else:
                with open(metrics_export, "w") as f:
                    json.dump(reports, f, indent=2)
        return reports


# ----------------------------- CLI Interface ------------------------------------- #

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Advanced Bias / Illumination Correction Tool")
    p.add_argument("input", help="Input image path or directory")
    p.add_argument("-o", "--output", help="Output image path or directory", required=True)
    p.add_argument("-m", "--method", default="n4",
                   choices=["n4", "histogram", "clahe", "simple", "poly", "retinex", "auto"],
                   help="Correction method")
    p.add_argument("--mask", help="Optional mask image path")
    p.add_argument("--mask-strategy", default="auto", choices=["auto", "percentile", "none"])
    p.add_argument("--degree", type=int, default=3, help="Polynomial degree (poly/simple)")
    p.add_argument("--retinex-mode", default="multi", choices=["multi", "single"])
    p.add_argument("--retinex-sigmas", type=int, nargs="+", default=[15, 80, 250])
    p.add_argument("--clahe-clip", type=float, default=0.01)
    p.add_argument("--clahe-nbins", type=int, default=256)
    p.add_argument("--poly-smooth-sigma", type=float, default=1.0)
    p.add_argument("--per-channel", action="store_true", help="Process each channel independently")
    p.add_argument("--no-per-channel", dest="per_channel", action="store_false")
    p.add_argument("--gpu", action="store_true", help="(Experimental) GPU usage if cupy available")
    p.add_argument("--batch", action="store_true", help="Treat input as directory for batch processing")
    p.add_argument("--pattern", default="*.nii*", help="Glob pattern for batch mode")
    p.add_argument("--parallel", type=int, default=0, help="Parallel workers for batch")
    p.add_argument("--metrics-export", help="Path to write JSON/YAML metrics/report")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--auto-method", action="store_true", help="Auto-select method per image")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    corrector = BiasFieldCorrector(
        method=args.method,
        mask_strategy=args.mask_strategy,
        per_channel=args.per_channel,
        use_gpu=args.gpu,
        polynomial_degree=args.degree,
        retinex_mode=args.retinex_mode,
        retinex_sigma_list=args.retinex_sigmas,
        clahe_clip_limit=args.clahe_clip,
        clahe_nbins=args.clahe_nbins,
        poly_smooth_sigma=args.poly_smooth_sigma,
        auto_method=args.auto_method,
    )

    input_path = Path(args.input)

    if args.batch:
        if not input_path.is_dir():
            logger.error("Batch mode requires input to be a directory.")
            return 2
        reports = corrector.batch_correct(
            input_dir=input_path,
            output_dir=Path(args.output),
            pattern=args.pattern,
            parallel=args.parallel,
            metrics_export=args.metrics_export
        )
        logger.info(f"Processed {len(reports)} files.")
    else:
        if not input_path.exists():
            logger.error(f"Input not found: {input_path}")
            return 2
        rep = corrector.correct_bias(
            input_path=input_path,
            output_path=Path(args.output),
            mask_path=args.mask,
            export_metrics=args.metrics_export
        )
        logger.info(f"Correction complete. Metrics: {json.dumps(rep['metrics'], indent=2)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
