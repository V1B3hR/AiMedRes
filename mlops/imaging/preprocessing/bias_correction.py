"""
Bias field correction module for medical images.

This module provides bias field correction capabilities, which is essential
for preprocessing MRI images to correct for intensity non-uniformity.
"""

import logging
import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Optional, Union, Dict, Any
import subprocess

logger = logging.getLogger(__name__)


class BiasFieldCorrector:
    """
    Bias field correction for medical images.
    
    Supports multiple methods including N4 bias field correction
    and simple histogram-based correction.
    """
    
    def __init__(self, method: str = "n4", **kwargs):
        """
        Initialize bias field corrector.
        
        Args:
            method: Correction method ('n4', 'histogram', 'simple')
            **kwargs: Additional parameters for the chosen method
        """
        self.method = method
        self.params = kwargs
        self.logger = logging.getLogger(__name__)
        
    def correct_bias(
        self, 
        input_path: Union[str, Path], 
        output_path: Optional[Union[str, Path]] = None,
        mask_path: Optional[Union[str, Path]] = None
    ) -> Union[str, np.ndarray]:
        """
        Apply bias field correction to image.
        
        Args:
            input_path: Path to input NIfTI image
            output_path: Path for output corrected image (optional)
            mask_path: Path to brain mask (optional)
            
        Returns:
            Output path if output_path provided, else corrected image array
        """
        input_path = Path(input_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input image not found: {input_path}")
            
        self.logger.info(f"Applying {self.method} bias correction to {input_path}")
        
        if self.method == "n4":
            return self._n4_correction(input_path, output_path, mask_path)
        elif self.method == "histogram":
            return self._histogram_correction(input_path, output_path)
        elif self.method == "simple":
            return self._simple_correction(input_path, output_path)
        else:
            raise ValueError(f"Unknown correction method: {self.method}")
    
    def _n4_correction(
        self, 
        input_path: Path, 
        output_path: Optional[Path],
        mask_path: Optional[Path]
    ) -> Union[str, np.ndarray]:
        """Apply N4 bias field correction using ANTs or fallback method."""
        try:
            # Try to use ANTs N4BiasFieldCorrection if available
            cmd = ["N4BiasFieldCorrection"]
            cmd.extend(["-i", str(input_path)])
            
            if output_path:
                cmd.extend(["-o", str(output_path)])
                
            if mask_path and Path(mask_path).exists():
                cmd.extend(["-x", str(mask_path)])
                
            # Add default parameters
            cmd.extend([
                "--convergence", "[100x50x50,0.001]",
                "--shrink-factor", "2",
                "--n-histogram-bins", "200"
            ])
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.logger.info("N4 bias correction completed successfully")
                return str(output_path) if output_path else self._load_image_data(input_path)
            else:
                self.logger.warning(f"N4 correction failed: {result.stderr}")
                # Fallback to simple method
                return self._simple_correction(input_path, output_path)
                
        except FileNotFoundError:
            self.logger.warning("N4BiasFieldCorrection not found, using fallback method")
            return self._simple_correction(input_path, output_path)
    
    def _histogram_correction(
        self, 
        input_path: Path, 
        output_path: Optional[Path]
    ) -> Union[str, np.ndarray]:
        """Apply histogram-based bias correction."""
        img = nib.load(input_path)
        data = img.get_fdata()
        
        # Apply histogram equalization-based correction
        corrected_data = self._histogram_equalize(data)
        
        if output_path:
            corrected_img = nib.Nifti1Image(corrected_data, img.affine, img.header)
            nib.save(corrected_img, output_path)
            self.logger.info(f"Histogram bias correction saved to {output_path}")
            return str(output_path)
        else:
            return corrected_data
    
    def _simple_correction(
        self, 
        input_path: Path, 
        output_path: Optional[Path]
    ) -> Union[str, np.ndarray]:
        """Apply simple polynomial bias correction."""
        img = nib.load(input_path)
        data = img.get_fdata()
        
        # Simple polynomial fitting for bias field estimation
        corrected_data = self._polynomial_correction(data)
        
        if output_path:
            corrected_img = nib.Nifti1Image(corrected_data, img.affine, img.header)
            nib.save(corrected_img, output_path)
            self.logger.info(f"Simple bias correction saved to {output_path}")
            return str(output_path)
        else:
            return corrected_data
    
    def _histogram_equalize(self, data: np.ndarray) -> np.ndarray:
        """Apply histogram equalization for bias correction."""
        # Remove background (assume values near 0 are background)
        mask = data > np.percentile(data, 5)
        
        if not np.any(mask):
            return data
            
        # Calculate cumulative distribution function
        hist, bins = np.histogram(data[mask], bins=256, density=True)
        cdf = hist.cumsum()
        cdf = cdf / cdf[-1]  # Normalize
        
        # Interpolate to get equalized values
        corrected_data = data.copy()
        corrected_data[mask] = np.interp(data[mask], bins[:-1], cdf * np.max(data[mask]))
        
        return corrected_data
    
    def _polynomial_correction(self, data: np.ndarray, degree: int = 3) -> np.ndarray:
        """Apply polynomial-based bias field correction."""
        shape = data.shape
        
        # Create coordinate grids
        coords = np.mgrid[:shape[0], :shape[1], :shape[2]]
        coords = coords.reshape(3, -1)
        
        # Normalize coordinates
        for i in range(3):
            coords[i] = (coords[i] - coords[i].mean()) / coords[i].std()
        
        # Create polynomial basis (simplified for performance)
        basis = np.ones((coords.shape[1], 4))
        basis[:, 1] = coords[0]  # x
        basis[:, 2] = coords[1]  # y  
        basis[:, 3] = coords[2]  # z
        
        # Flatten data and find foreground
        data_flat = data.flatten()
        mask = data_flat > np.percentile(data_flat, 10)
        
        if not np.any(mask):
            return data
            
        try:
            # Fit polynomial to foreground intensities
            coeffs = np.linalg.lstsq(basis[mask], data_flat[mask], rcond=None)[0]
            
            # Estimate bias field
            bias_field = basis @ coeffs
            bias_field = bias_field.reshape(shape)
            
            # Correct by dividing by bias field (avoid division by zero)
            bias_field[bias_field < 0.1] = 1.0
            corrected_data = data / bias_field
            
            return corrected_data
            
        except np.linalg.LinAlgError:
            self.logger.warning("Polynomial fitting failed, returning original data")
            return data
    
    def _load_image_data(self, path: Path) -> np.ndarray:
        """Load image data as numpy array."""
        img = nib.load(path)
        return img.get_fdata()
    
    def validate_inputs(self, input_path: Union[str, Path]) -> bool:
        """Validate input file exists and is a valid NIfTI image."""
        try:
            input_path = Path(input_path)
            if not input_path.exists():
                return False
                
            # Try to load as NIfTI
            nib.load(input_path)
            return True
            
        except Exception as e:
            self.logger.error(f"Input validation failed: {e}")
            return False
    
    def get_correction_quality_metrics(
        self, 
        original_data: np.ndarray, 
        corrected_data: np.ndarray
    ) -> Dict[str, float]:
        """Calculate quality metrics for bias correction."""
        # Calculate coefficient of variation (CV) - lower is better for bias correction
        original_cv = np.std(original_data[original_data > 0]) / np.mean(original_data[original_data > 0])
        corrected_cv = np.std(corrected_data[corrected_data > 0]) / np.mean(corrected_data[corrected_data > 0])
        
        # Calculate intensity uniformity
        original_uniformity = 1.0 - original_cv
        corrected_uniformity = 1.0 - corrected_cv
        
        return {
            "original_cv": float(original_cv),
            "corrected_cv": float(corrected_cv),
            "original_uniformity": float(original_uniformity),
            "corrected_uniformity": float(corrected_uniformity),
            "improvement_ratio": float(original_cv / corrected_cv) if corrected_cv > 0 else 1.0
        }