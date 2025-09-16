"""
Skull stripping module for brain MRI images.

This module provides skull stripping capabilities to extract brain tissue
from head MRI scans, removing skull, scalp, and other non-brain tissue.
"""

import logging
import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Optional, Union, Dict, Any, Tuple
import subprocess
from scipy import ndimage
from scipy.ndimage import binary_erosion, binary_dilation

logger = logging.getLogger(__name__)


class SkullStripper:
    """
    Brain extraction/skull stripping for MRI images.
    
    Supports multiple methods including FSL BET, simple thresholding,
    and morphological operations.
    """
    
    def __init__(self, method: str = "bet", **kwargs):
        """
        Initialize skull stripper.
        
        Args:
            method: Skull stripping method ('bet', 'threshold', 'morphological')
            **kwargs: Additional parameters for the chosen method
        """
        self.method = method
        self.params = kwargs
        self.logger = logging.getLogger(__name__)
        
    def extract_brain(
        self, 
        input_path: Union[str, Path], 
        output_path: Optional[Union[str, Path]] = None,
        mask_path: Optional[Union[str, Path]] = None
    ) -> Union[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Extract brain from skull.
        
        Args:
            input_path: Path to input NIfTI image
            output_path: Path for output brain-extracted image (optional)
            mask_path: Path for output brain mask (optional)
            
        Returns:
            Output path if output_path provided, else (brain_data, mask_data) tuple
        """
        input_path = Path(input_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input image not found: {input_path}")
            
        self.logger.info(f"Applying {self.method} skull stripping to {input_path}")
        
        if self.method == "bet":
            return self._bet_extraction(input_path, output_path, mask_path)
        elif self.method == "threshold":
            return self._threshold_extraction(input_path, output_path, mask_path)
        elif self.method == "morphological":
            return self._morphological_extraction(input_path, output_path, mask_path)
        else:
            raise ValueError(f"Unknown extraction method: {self.method}")
    
    def _bet_extraction(
        self, 
        input_path: Path, 
        output_path: Optional[Path],
        mask_path: Optional[Path]
    ) -> Union[str, Tuple[np.ndarray, np.ndarray]]:
        """Apply FSL BET brain extraction."""
        try:
            # Try to use FSL BET if available
            cmd = ["bet", str(input_path)]
            
            if output_path:
                cmd.append(str(output_path))
            else:
                # Create temporary output
                temp_output = input_path.with_suffix(".bet.nii.gz")
                cmd.append(str(temp_output))
                
            # Add BET parameters
            fractional_intensity = self.params.get("fractional_intensity", 0.3)
            gradient_threshold = self.params.get("gradient_threshold", 0.4)
            
            cmd.extend(["-f", str(fractional_intensity)])
            cmd.extend(["-g", str(gradient_threshold)])
            cmd.append("-m")  # Generate brain mask
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.logger.info("BET skull stripping completed successfully")
                
                if output_path:
                    # Also save mask if requested
                    if mask_path:
                        mask_output = output_path.with_suffix("_mask.nii.gz")
                        if mask_output.exists():
                            import shutil
                            shutil.move(str(mask_output), str(mask_path))
                    return str(output_path)
                else:
                    # Load results
                    brain_img = nib.load(temp_output)
                    brain_data = brain_img.get_fdata()
                    
                    mask_output = temp_output.with_suffix("_mask.nii.gz")
                    if mask_output.exists():
                        mask_img = nib.load(mask_output)
                        mask_data = mask_img.get_fdata()
                    else:
                        mask_data = (brain_data > 0).astype(np.float32)
                    
                    # Clean up temp files
                    temp_output.unlink(missing_ok=True)
                    mask_output.unlink(missing_ok=True)
                    
                    return brain_data, mask_data
                    
            else:
                self.logger.warning(f"BET extraction failed: {result.stderr}")
                # Fallback to threshold method
                return self._threshold_extraction(input_path, output_path, mask_path)
                
        except FileNotFoundError:
            self.logger.warning("FSL BET not found, using fallback method")
            return self._threshold_extraction(input_path, output_path, mask_path)
    
    def _threshold_extraction(
        self, 
        input_path: Path, 
        output_path: Optional[Path],
        mask_path: Optional[Path]
    ) -> Union[str, Tuple[np.ndarray, np.ndarray]]:
        """Apply threshold-based skull stripping."""
        img = nib.load(input_path)
        data = img.get_fdata()
        
        # Create brain mask using adaptive thresholding
        brain_mask = self._create_brain_mask_threshold(data)
        brain_data = data * brain_mask
        
        if output_path:
            # Save brain-extracted image
            brain_img = nib.Nifti1Image(brain_data, img.affine, img.header)
            nib.save(brain_img, output_path)
            
            # Save mask if requested
            if mask_path:
                mask_img = nib.Nifti1Image(brain_mask, img.affine, img.header)
                nib.save(mask_img, mask_path)
            
            self.logger.info(f"Threshold skull stripping saved to {output_path}")
            return str(output_path)
        else:
            return brain_data, brain_mask
    
    def _morphological_extraction(
        self, 
        input_path: Path, 
        output_path: Optional[Path],
        mask_path: Optional[Path]
    ) -> Union[str, Tuple[np.ndarray, np.ndarray]]:
        """Apply morphological operations for skull stripping."""
        img = nib.load(input_path)
        data = img.get_fdata()
        
        # Create brain mask using morphological operations
        brain_mask = self._create_brain_mask_morphological(data)
        brain_data = data * brain_mask
        
        if output_path:
            # Save brain-extracted image
            brain_img = nib.Nifti1Image(brain_data, img.affine, img.header)
            nib.save(brain_img, output_path)
            
            # Save mask if requested
            if mask_path:
                mask_img = nib.Nifti1Image(brain_mask, img.affine, img.header)
                nib.save(mask_img, mask_path)
            
            self.logger.info(f"Morphological skull stripping saved to {output_path}")
            return str(output_path)
        else:
            return brain_data, brain_mask
    
    def _create_brain_mask_threshold(self, data: np.ndarray) -> np.ndarray:
        """Create brain mask using adaptive thresholding."""
        # Calculate threshold based on image statistics
        threshold_percentile = self.params.get("threshold_percentile", 15)
        threshold = np.percentile(data[data > 0], threshold_percentile)
        
        # Create initial mask
        brain_mask = data > threshold
        
        # Apply morphological operations to clean up mask
        brain_mask = self._clean_mask(brain_mask)
        
        return brain_mask.astype(np.float32)
    
    def _create_brain_mask_morphological(self, data: np.ndarray) -> np.ndarray:
        """Create brain mask using morphological operations."""
        # Otsu-like thresholding
        hist, bins = np.histogram(data[data > 0], bins=256)
        threshold = self._otsu_threshold(hist, bins)
        
        # Initial mask
        brain_mask = data > threshold
        
        # Apply extensive morphological cleaning
        brain_mask = self._clean_mask_aggressive(brain_mask)
        
        return brain_mask.astype(np.float32)
    
    def _otsu_threshold(self, hist: np.ndarray, bins: np.ndarray) -> float:
        """Calculate Otsu threshold for binary segmentation."""
        total = hist.sum()
        if total == 0:
            return 0
            
        current_max, threshold = 0, 0
        sum_total = (hist * bins[:-1]).sum()
        sum_foreground = 0
        weight_background = 0
        
        for i in range(len(hist)):
            weight_background += hist[i]
            if weight_background == 0:
                continue
                
            weight_foreground = total - weight_background
            if weight_foreground == 0:
                break
                
            sum_foreground += i * hist[i]
            
            mean_background = sum_foreground / weight_background
            mean_foreground = (sum_total - sum_foreground) / weight_foreground
            
            # Between-class variance
            variance_between = (weight_background * weight_foreground * 
                              (mean_background - mean_foreground) ** 2)
            
            if variance_between > current_max:
                current_max = variance_between
                threshold = bins[i]
                
        return threshold
    
    def _clean_mask(self, mask: np.ndarray) -> np.ndarray:
        """Clean brain mask using morphological operations."""
        # Remove small components
        labeled_mask, num_labels = ndimage.label(mask)
        if num_labels > 1:
            # Keep only the largest component
            sizes = ndimage.sum(mask, labeled_mask, range(1, num_labels + 1))
            max_label = np.argmax(sizes) + 1
            mask = labeled_mask == max_label
        
        # Morphological closing to fill holes
        structure = np.ones((3, 3, 3))
        mask = ndimage.binary_closing(mask, structure=structure, iterations=2)
        
        # Morphological opening to remove small connections
        mask = ndimage.binary_opening(mask, structure=structure, iterations=1)
        
        return mask
    
    def _clean_mask_aggressive(self, mask: np.ndarray) -> np.ndarray:
        """Aggressively clean brain mask."""
        # Start with basic cleaning
        mask = self._clean_mask(mask)
        
        # Additional morphological operations
        structure = np.ones((5, 5, 5))
        
        # Erode then dilate to separate connected components
        mask = binary_erosion(mask, structure=structure, iterations=1)
        mask = binary_dilation(mask, structure=structure, iterations=2)
        
        # Fill holes
        mask = ndimage.binary_fill_holes(mask)
        
        # Remove small components again after processing
        labeled_mask, num_labels = ndimage.label(mask)
        if num_labels > 1:
            sizes = ndimage.sum(mask, labeled_mask, range(1, num_labels + 1))
            max_label = np.argmax(sizes) + 1
            mask = labeled_mask == max_label
        
        return mask
    
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
    
    def get_extraction_quality_metrics(
        self, 
        original_data: np.ndarray, 
        brain_data: np.ndarray,
        brain_mask: np.ndarray
    ) -> Dict[str, float]:
        """Calculate quality metrics for brain extraction."""
        # Calculate brain volume ratio
        total_voxels = np.prod(original_data.shape)
        brain_voxels = np.sum(brain_mask > 0)
        brain_volume_ratio = brain_voxels / total_voxels
        
        # Calculate intensity preservation (correlation)
        mask_indices = brain_mask > 0
        if np.any(mask_indices):
            correlation = np.corrcoef(
                original_data[mask_indices].flatten(),
                brain_data[mask_indices].flatten()
            )[0, 1]
        else:
            correlation = 0.0
        
        # Calculate mask compactness (sphericity approximation)
        if brain_voxels > 0:
            surface_area = np.sum(np.gradient(brain_mask.astype(float)))
            volume = brain_voxels
            compactness = (np.pi ** (1/3)) * ((6 * volume) ** (2/3)) / surface_area
        else:
            compactness = 0.0
        
        return {
            "brain_volume_ratio": float(brain_volume_ratio),
            "intensity_correlation": float(correlation) if not np.isnan(correlation) else 0.0,
            "mask_compactness": float(compactness) if not np.isnan(compactness) else 0.0,
            "brain_voxel_count": int(brain_voxels),
            "total_voxel_count": int(total_voxels)
        }