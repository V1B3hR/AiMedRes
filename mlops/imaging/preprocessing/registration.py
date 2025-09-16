"""
Image registration module for medical images.

This module provides image registration capabilities to align medical images
to standard anatomical templates or between different imaging sessions.
"""

import logging
import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Optional, Union, Dict, Any, Tuple, List
import subprocess
from scipy import ndimage
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


class ImageRegistrar:
    """
    Medical image registration for spatial alignment.
    
    Supports multiple registration methods including ANTs, simple affine
    transformation, and intensity-based registration.
    """
    
    def __init__(self, method: str = "ants", **kwargs):
        """
        Initialize image registrar.
        
        Args:
            method: Registration method ('ants', 'affine', 'rigid', 'similarity')
            **kwargs: Additional parameters for the chosen method
        """
        self.method = method
        self.params = kwargs
        self.logger = logging.getLogger(__name__)
        
    def register_images(
        self, 
        moving_path: Union[str, Path],
        fixed_path: Union[str, Path], 
        output_path: Optional[Union[str, Path]] = None,
        transform_path: Optional[Union[str, Path]] = None
    ) -> Union[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Register moving image to fixed image.
        
        Args:
            moving_path: Path to moving (source) image
            fixed_path: Path to fixed (target/template) image
            output_path: Path for output registered image (optional)
            transform_path: Path for output transformation (optional)
            
        Returns:
            Output path if output_path provided, else (registered_data, transform_matrix) tuple
        """
        moving_path = Path(moving_path)
        fixed_path = Path(fixed_path)
        
        if not moving_path.exists():
            raise FileNotFoundError(f"Moving image not found: {moving_path}")
        if not fixed_path.exists():
            raise FileNotFoundError(f"Fixed image not found: {fixed_path}")
            
        self.logger.info(f"Applying {self.method} registration: {moving_path} → {fixed_path}")
        
        if self.method == "ants":
            return self._ants_registration(moving_path, fixed_path, output_path, transform_path)
        elif self.method == "affine":
            return self._affine_registration(moving_path, fixed_path, output_path, transform_path)
        elif self.method == "rigid":
            return self._rigid_registration(moving_path, fixed_path, output_path, transform_path)
        elif self.method == "similarity":
            return self._similarity_registration(moving_path, fixed_path, output_path, transform_path)
        else:
            raise ValueError(f"Unknown registration method: {self.method}")
    
    def register_to_template(
        self,
        input_path: Union[str, Path],
        template: str = "MNI152",
        output_path: Optional[Union[str, Path]] = None
    ) -> Union[str, np.ndarray]:
        """
        Register image to standard template.
        
        Args:
            input_path: Path to input image
            template: Template name ('MNI152', 'MNI152_1mm', 'MNI152_2mm')
            output_path: Path for output registered image (optional)
            
        Returns:
            Output path if output_path provided, else registered image data
        """
        # Create synthetic template for demonstration
        # In practice, this would load actual template files
        template_data = self._get_template(template)
        
        input_path = Path(input_path)
        img = nib.load(input_path)
        
        # Create temporary template file
        temp_template = input_path.parent / f"temp_template_{template}.nii.gz"
        template_img = nib.Nifti1Image(template_data, img.affine, img.header)
        nib.save(template_img, temp_template)
        
        try:
            result = self.register_images(input_path, temp_template, output_path)
            return result
        finally:
            # Clean up temporary template
            temp_template.unlink(missing_ok=True)
    
    def _ants_registration(
        self, 
        moving_path: Path,
        fixed_path: Path, 
        output_path: Optional[Path],
        transform_path: Optional[Path]
    ) -> Union[str, Tuple[np.ndarray, np.ndarray]]:
        """Apply ANTs registration."""
        try:
            # Try to use ANTs if available
            cmd = ["antsRegistration"]
            
            # Set up output prefix
            if output_path:
                output_prefix = str(output_path.with_suffix(""))
            else:
                output_prefix = str(moving_path.with_suffix("").with_suffix("")) + "_registered"
            
            cmd.extend([
                "--dimensionality", "3",
                "--float", "0",
                "--output", f"[{output_prefix},{output_prefix}_Warped.nii.gz]",
                "--interpolation", "Linear",
                "--winsorize-image-intensities", "[0.005,0.995]",
                "--use-histogram-matching", "0"
            ])
            
            # Add transformation stages
            cmd.extend([
                "--initial-moving-transform", f"[{fixed_path},{moving_path},1]",
                "--transform", "Rigid[0.1]",
                "--metric", f"MI[{fixed_path},{moving_path},1,32,Regular,0.25]",
                "--convergence", "[1000x500x250x100,1e-6,10]",
                "--shrink-factors", "8x4x2x1",
                "--smoothing-sigmas", "3x2x1x0vox"
            ])
            
            # Add affine stage
            cmd.extend([
                "--transform", "Affine[0.1]",
                "--metric", f"MI[{fixed_path},{moving_path},1,32,Regular,0.25]",
                "--convergence", "[1000x500x250x100,1e-6,10]",
                "--shrink-factors", "8x4x2x1",
                "--smoothing-sigmas", "3x2x1x0vox"
            ])
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.logger.info("ANTs registration completed successfully")
                
                warped_file = Path(f"{output_prefix}_Warped.nii.gz")
                if output_path and warped_file.exists():
                    import shutil
                    shutil.move(str(warped_file), str(output_path))
                    return str(output_path)
                elif warped_file.exists():
                    # Load result
                    registered_img = nib.load(warped_file)
                    registered_data = registered_img.get_fdata()
                    
                    # Load transform (simplified)
                    transform_data = np.eye(4)  # Placeholder
                    
                    # Clean up
                    warped_file.unlink(missing_ok=True)
                    
                    return registered_data, transform_data
                    
            else:
                self.logger.warning(f"ANTs registration failed: {result.stderr}")
                # Fallback to affine method
                return self._affine_registration(moving_path, fixed_path, output_path, transform_path)
                
        except FileNotFoundError:
            self.logger.warning("ANTs not found, using fallback method")
            return self._affine_registration(moving_path, fixed_path, output_path, transform_path)
    
    def _affine_registration(
        self, 
        moving_path: Path,
        fixed_path: Path, 
        output_path: Optional[Path],
        transform_path: Optional[Path]
    ) -> Union[str, Tuple[np.ndarray, np.ndarray]]:
        """Apply affine registration using intensity-based optimization."""
        moving_img = nib.load(moving_path)
        fixed_img = nib.load(fixed_path)
        
        moving_data = moving_img.get_fdata()
        fixed_data = fixed_img.get_fdata()
        
        # Perform affine registration
        transform_matrix = self._optimize_affine_transform(moving_data, fixed_data)
        registered_data = self._apply_transform(moving_data, transform_matrix, fixed_data.shape)
        
        if output_path:
            # Save registered image
            registered_img = nib.Nifti1Image(registered_data, fixed_img.affine, fixed_img.header)
            nib.save(registered_img, output_path)
            
            # Save transform if requested
            if transform_path:
                np.savetxt(transform_path, transform_matrix)
            
            self.logger.info(f"Affine registration saved to {output_path}")
            return str(output_path)
        else:
            return registered_data, transform_matrix
    
    def _rigid_registration(
        self, 
        moving_path: Path,
        fixed_path: Path, 
        output_path: Optional[Path],
        transform_path: Optional[Path]
    ) -> Union[str, Tuple[np.ndarray, np.ndarray]]:
        """Apply rigid registration (rotation + translation only)."""
        moving_img = nib.load(moving_path)
        fixed_img = nib.load(fixed_path)
        
        moving_data = moving_img.get_fdata()
        fixed_data = fixed_img.get_fdata()
        
        # Perform rigid registration
        transform_matrix = self._optimize_rigid_transform(moving_data, fixed_data)
        registered_data = self._apply_transform(moving_data, transform_matrix, fixed_data.shape)
        
        if output_path:
            # Save registered image
            registered_img = nib.Nifti1Image(registered_data, fixed_img.affine, fixed_img.header)
            nib.save(registered_img, output_path)
            
            # Save transform if requested
            if transform_path:
                np.savetxt(transform_path, transform_matrix)
            
            self.logger.info(f"Rigid registration saved to {output_path}")
            return str(output_path)
        else:
            return registered_data, transform_matrix
    
    def _similarity_registration(
        self, 
        moving_path: Path,
        fixed_path: Path, 
        output_path: Optional[Path],
        transform_path: Optional[Path]
    ) -> Union[str, Tuple[np.ndarray, np.ndarray]]:
        """Apply similarity transformation (rigid + uniform scaling)."""
        moving_img = nib.load(moving_path)
        fixed_img = nib.load(fixed_path)
        
        moving_data = moving_img.get_fdata()
        fixed_data = fixed_img.get_fdata()
        
        # Perform similarity registration
        transform_matrix = self._optimize_similarity_transform(moving_data, fixed_data)
        registered_data = self._apply_transform(moving_data, transform_matrix, fixed_data.shape)
        
        if output_path:
            # Save registered image
            registered_img = nib.Nifti1Image(registered_data, fixed_img.affine, fixed_img.header)
            nib.save(registered_img, output_path)
            
            # Save transform if requested
            if transform_path:
                np.savetxt(transform_path, transform_matrix)
            
            self.logger.info(f"Similarity registration saved to {output_path}")
            return str(output_path)
        else:
            return registered_data, transform_matrix
    
    def _optimize_affine_transform(self, moving_data: np.ndarray, fixed_data: np.ndarray) -> np.ndarray:
        """Optimize affine transformation parameters."""
        # Initialize with identity transform
        initial_params = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0])  # 12 parameters
        
        # Downsample for efficiency
        moving_small = self._downsample_image(moving_data, factor=4)
        fixed_small = self._downsample_image(fixed_data, factor=4)
        
        def objective(params):
            transform_matrix = self._params_to_affine_matrix(params)
            transformed = self._apply_transform(moving_small, transform_matrix, fixed_small.shape)
            return -self._normalized_cross_correlation(transformed, fixed_small)
        
        # Optimize
        result = minimize(objective, initial_params, method='Powell',
                         options={'maxiter': 100, 'disp': False})
        
        return self._params_to_affine_matrix(result.x)
    
    def _optimize_rigid_transform(self, moving_data: np.ndarray, fixed_data: np.ndarray) -> np.ndarray:
        """Optimize rigid transformation parameters."""
        # Initialize with identity transform (6 parameters: 3 rotations + 3 translations)
        initial_params = np.array([0, 0, 0, 0, 0, 0])
        
        # Downsample for efficiency
        moving_small = self._downsample_image(moving_data, factor=4)
        fixed_small = self._downsample_image(fixed_data, factor=4)
        
        def objective(params):
            transform_matrix = self._params_to_rigid_matrix(params)
            transformed = self._apply_transform(moving_small, transform_matrix, fixed_small.shape)
            return -self._normalized_cross_correlation(transformed, fixed_small)
        
        # Optimize
        result = minimize(objective, initial_params, method='Powell',
                         options={'maxiter': 100, 'disp': False})
        
        return self._params_to_rigid_matrix(result.x)
    
    def _optimize_similarity_transform(self, moving_data: np.ndarray, fixed_data: np.ndarray) -> np.ndarray:
        """Optimize similarity transformation parameters."""
        # Initialize with identity transform (7 parameters: 3 rotations + 3 translations + 1 scale)
        initial_params = np.array([0, 0, 0, 0, 0, 0, 1])
        
        # Downsample for efficiency
        moving_small = self._downsample_image(moving_data, factor=4)
        fixed_small = self._downsample_image(fixed_data, factor=4)
        
        def objective(params):
            transform_matrix = self._params_to_similarity_matrix(params)
            transformed = self._apply_transform(moving_small, transform_matrix, fixed_small.shape)
            return -self._normalized_cross_correlation(transformed, fixed_small)
        
        # Optimize
        result = minimize(objective, initial_params, method='Powell',
                         options={'maxiter': 100, 'disp': False})
        
        return self._params_to_similarity_matrix(result.x)
    
    def _params_to_affine_matrix(self, params: np.ndarray) -> np.ndarray:
        """Convert 12-parameter vector to 4x4 affine matrix."""
        rx, ry, rz = params[0:3]  # rotations
        tx, ty, tz = params[3:6]  # translations
        sx, sy, sz = params[6:9]  # scales
        kx, ky, kz = params[9:12]  # shears
        
        # Rotation matrices
        Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
        Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
        Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
        
        # Combined rotation
        R = Rz @ Ry @ Rx
        
        # Scale matrix
        S = np.diag([sx, sy, sz])
        
        # Shear matrix
        K = np.array([[1, kx, 0], [ky, 1, 0], [0, kz, 1]])
        
        # Combine transformations
        A = R @ S @ K
        
        # Create 4x4 matrix
        transform = np.eye(4)
        transform[0:3, 0:3] = A
        transform[0:3, 3] = [tx, ty, tz]
        
        return transform
    
    def _params_to_rigid_matrix(self, params: np.ndarray) -> np.ndarray:
        """Convert 6-parameter vector to 4x4 rigid transformation matrix."""
        rx, ry, rz = params[0:3]  # rotations
        tx, ty, tz = params[3:6]  # translations
        
        # Rotation matrices
        Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
        Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
        Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
        
        # Combined rotation
        R = Rz @ Ry @ Rx
        
        # Create 4x4 matrix
        transform = np.eye(4)
        transform[0:3, 0:3] = R
        transform[0:3, 3] = [tx, ty, tz]
        
        return transform
    
    def _params_to_similarity_matrix(self, params: np.ndarray) -> np.ndarray:
        """Convert 7-parameter vector to 4x4 similarity transformation matrix."""
        rx, ry, rz = params[0:3]  # rotations
        tx, ty, tz = params[3:6]  # translations
        s = params[6]  # uniform scale
        
        # Rotation matrices
        Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
        Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
        Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
        
        # Combined rotation and scale
        A = s * (Rz @ Ry @ Rx)
        
        # Create 4x4 matrix
        transform = np.eye(4)
        transform[0:3, 0:3] = A
        transform[0:3, 3] = [tx, ty, tz]
        
        return transform
    
    def _apply_transform(self, image: np.ndarray, transform_matrix: np.ndarray, output_shape: Tuple[int, ...]) -> np.ndarray:
        """Apply transformation matrix to image."""
        # Use scipy's affine_transform
        # Note: scipy expects inverse transform
        try:
            inverse_transform = np.linalg.inv(transform_matrix)
            transformed = ndimage.affine_transform(
                image,
                inverse_transform[0:3, 0:3],
                offset=inverse_transform[0:3, 3],
                output_shape=output_shape,
                order=1  # Linear interpolation
            )
            return transformed
        except np.linalg.LinAlgError:
            self.logger.warning("Transform matrix is singular, returning original image")
            return image
    
    def _downsample_image(self, image: np.ndarray, factor: int = 2) -> np.ndarray:
        """Downsample image for faster processing."""
        return image[::factor, ::factor, ::factor]
    
    def _normalized_cross_correlation(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate normalized cross correlation between two images."""
        # Flatten and normalize
        img1_flat = img1.flatten()
        img2_flat = img2.flatten()
        
        # Remove mean
        img1_centered = img1_flat - np.mean(img1_flat)
        img2_centered = img2_flat - np.mean(img2_flat)
        
        # Calculate correlation
        numerator = np.sum(img1_centered * img2_centered)
        denominator = np.sqrt(np.sum(img1_centered**2) * np.sum(img2_centered**2))
        
        if denominator == 0:
            return 0.0
            
        return numerator / denominator
    
    def _get_template(self, template_name: str) -> np.ndarray:
        """Get template image data (synthetic for demonstration)."""
        # In practice, this would load actual template files
        if template_name.startswith("MNI152"):
            # Create synthetic MNI152-like template
            shape = (182, 218, 182) if "1mm" in template_name else (91, 109, 91)
            template = np.zeros(shape)
            
            # Add synthetic brain structure
            center = np.array(shape) // 2
            for i in range(shape[0]):
                for j in range(shape[1]):
                    for k in range(shape[2]):
                        dist = np.sqrt((i - center[0])**2 + (j - center[1])**2 + (k - center[2])**2)
                        if dist < min(shape) // 4:
                            template[i, j, k] = 100 * np.exp(-dist / 20)
            
            return template
        else:
            raise ValueError(f"Unknown template: {template_name}")
    
    def validate_inputs(self, moving_path: Union[str, Path], fixed_path: Union[str, Path]) -> bool:
        """Validate input files exist and are valid NIfTI images."""
        try:
            moving_path = Path(moving_path)
            fixed_path = Path(fixed_path)
            
            if not moving_path.exists() or not fixed_path.exists():
                return False
                
            # Try to load as NIfTI
            nib.load(moving_path)
            nib.load(fixed_path)
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
        """Calculate quality metrics for image registration."""
        # Normalized cross correlation
        ncc_before = self._normalized_cross_correlation(moving_data, fixed_data)
        ncc_after = self._normalized_cross_correlation(registered_data, fixed_data)
        
        # Mean squared error
        mse_before = np.mean((moving_data - fixed_data)**2)
        mse_after = np.mean((registered_data - fixed_data)**2)
        
        # Mutual information approximation (simplified)
        mi_before = self._mutual_information_approx(moving_data, fixed_data)
        mi_after = self._mutual_information_approx(registered_data, fixed_data)
        
        return {
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
    
    def _mutual_information_approx(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Approximate mutual information between two images."""
        # Simplified MI calculation using histograms
        hist_2d, _, _ = np.histogram2d(img1.flatten(), img2.flatten(), bins=32)
        
        # Add small value to avoid log(0)
        hist_2d = hist_2d + 1e-10
        
        # Normalize
        hist_2d = hist_2d / np.sum(hist_2d)
        
        # Calculate marginal distributions
        hist_x = np.sum(hist_2d, axis=1)
        hist_y = np.sum(hist_2d, axis=0)
        
        # Calculate MI
        mi = 0
        for i in range(len(hist_x)):
            for j in range(len(hist_y)):
                if hist_2d[i, j] > 0:
                    mi += hist_2d[i, j] * np.log(hist_2d[i, j] / (hist_x[i] * hist_y[j]))
        
        return mi