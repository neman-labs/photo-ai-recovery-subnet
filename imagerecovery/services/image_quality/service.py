import logging
from typing import Optional, Tuple

import bittensor as bt
import lpips
import numpy as np
import torch
from pytorch_msssim import ssim

from imagerecovery.utils import bytes_to_image

# Global LPIPS model instance for reuse
_lpips_model = None
_lpips_device = None


class ImageQualityService:
    """
    Service for calculating image quality metrics between original and restored images.

    Uses a weighted combination of three metrics:
    - PSNR (Peak Signal-to-Noise Ratio): 10% weight
    - SSIM (Structural Similarity Index): 10% weight
    - LPIPS (Learned Perceptual Image Patch Similarity): 80% weight
    """

    # Weights for each metric
    PSNR_WEIGHT = 0.1
    SSIM_WEIGHT = 0.1
    LPIPS_WEIGHT = 0.8

    def __init__(self, device: Optional[str] = None) -> None:
        """
        Initialize the service with metric parameters.

        Args:
            device: Computation device for LPIPS ('cuda' or 'cpu').
                   If None, will use CUDA if available, otherwise CPU.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # PSNR parameters
        self.psnr_max_value = 255.0
        self.psnr_upper_bound = 50.0
        self.psnr_lower_bound = 10.0

        # SSIM parameters
        self.ssim_win_size = 11

        # LPIPS parameters
        self.lpips_threshold = 0.5

        # Initialize LPIPS model
        self._init_lpips_model()

        bt.logging.info(f"ImageQualityService initialized with device: {self.device}")
        bt.logging.info(f"Metric weights - PSNR: {self.PSNR_WEIGHT}, SSIM: {self.SSIM_WEIGHT}, LPIPS: {self.LPIPS_WEIGHT}")

    def calculate_score(self, reference_img_bytes: bytes, restored_img_bytes: bytes, miner_uid: int) -> float:
        """
        Calculate weighted quality score between original and restored images.

        Args:
            reference_img_bytes: Original high-quality image in bytes format
            restored_img_bytes: Restored image in bytes format

        Returns:
            float: Combined quality score (0.0-1.0 range)
        """
        try:
            # Convert bytes to numpy arrays
            reference_img = self._bytes_to_numpy(reference_img_bytes)
            restored_img = self._bytes_to_numpy(restored_img_bytes)

            # Apply metrics
            return self._calculate_weighted_score(reference_img, restored_img, miner_uid)
        except Exception as e:
            bt.logging.error(f"Error calculating image quality score: {e}")
            return 0.0  # Return minimum score on error

    def _calculate_weighted_score(self, reference_img: np.ndarray, restored_img: np.ndarray, miner_uid: int) -> float:
        """
        Calculate weighted combination of quality metrics.

        Args:
            reference_img: Reference image as numpy array
            restored_img: Restored image as numpy array

        Returns:
            float: Combined quality score (0.0-1.0 range)
        """
        # Apply metrics and get normalized scores
        _, psnr_score = self._calculate_psnr(reference_img, restored_img)
        _, ssim_score = self._calculate_ssim(reference_img, restored_img)
        _, lpips_score = self._calculate_lpips(reference_img, restored_img)

        # Apply weights and combine
        weighted_score = self.PSNR_WEIGHT * psnr_score + self.SSIM_WEIGHT * ssim_score + self.LPIPS_WEIGHT * lpips_score

        bt.logging.info(
            f"Image quality metrics for user UID {miner_uid} - PSNR: {psnr_score:.4f}, SSIM: {ssim_score:.4f}, "
            f"LPIPS: {lpips_score:.4f}, reward: {weighted_score:.4f}"
        )

        return weighted_score

    def _bytes_to_numpy(self, image_bytes: bytes) -> np.ndarray:
        """
        Convert image bytes to numpy array in RGB format.

        Args:
            image_bytes: Image in bytes format

        Returns:
            np.ndarray: Image as numpy array with shape (H, W, 3) and type uint8
        """
        # Convert bytes to PIL Image
        pil_image = bytes_to_image(image_bytes)

        # Ensure image is in RGB format
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        # Convert to numpy array
        return np.array(pil_image)

    def _init_lpips_model(self):
        """Initialize the global LPIPS model."""
        global _lpips_model, _lpips_device

        if _lpips_model is None or _lpips_device != self.device:
            _lpips_model = lpips.LPIPS(net="alex", verbose=False)
            _lpips_model.to(self.device)
            _lpips_model.eval()
            _lpips_device = self.device

    def _calculate_psnr(self, reference_img: np.ndarray, restored_img: np.ndarray) -> Tuple[float, float]:
        """Calculate PSNR and normalized PSNR between two images."""
        ref = reference_img.astype(np.float32)
        res = restored_img.astype(np.float32)

        mse = np.mean((ref - res) ** 2)

        if mse == 0:
            return float("inf"), 1.0

        psnr = 20 * np.log10(self.psnr_max_value) - 10 * np.log10(mse)

        if np.isinf(psnr):
            return psnr, 1.0

        clamped = np.clip(psnr, self.psnr_lower_bound, self.psnr_upper_bound)
        normalized = (clamped - self.psnr_lower_bound) / (self.psnr_upper_bound - self.psnr_lower_bound)

        return psnr, normalized

    def _calculate_ssim(self, reference_img: np.ndarray, restored_img: np.ndarray) -> Tuple[float, float]:
        """Calculate SSIM between two images."""
        if reference_img.shape != restored_img.shape:
            raise ValueError("Image dimensions must match")

        # Convert to float32 and move channel axis to first position
        ref = reference_img.astype(np.float32)
        res = restored_img.astype(np.float32)

        # Convert to PyTorch format (N, C, H, W)
        ref_tensor = torch.from_numpy(np.moveaxis(ref, 2, 0)).unsqueeze(0)
        res_tensor = torch.from_numpy(np.moveaxis(res, 2, 0)).unsqueeze(0)

        with torch.no_grad():
            ssim_value = ssim(
                ref_tensor,
                res_tensor,
                win_size=self.ssim_win_size,
                data_range=int(255.0 if reference_img.dtype == np.uint8 else 1.0),
            )

        return ssim_value.item(), ssim_value.item()

    def _calculate_lpips(self, reference_img: np.ndarray, restored_img: np.ndarray) -> Tuple[float, float]:
        """Calculate LPIPS and normalized LPIPS between two images."""
        global _lpips_model

        if reference_img.shape != restored_img.shape:
            raise ValueError("Image dimensions must match")

        # Preprocess images
        ref = reference_img.astype(np.float32) / 255.0 * 2.0 - 1.0
        res = restored_img.astype(np.float32) / 255.0 * 2.0 - 1.0

        # Convert to PyTorch format (N, C, H, W)
        ref_tensor = torch.from_numpy(ref.transpose(2, 0, 1)).unsqueeze(0)
        res_tensor = torch.from_numpy(res.transpose(2, 0, 1)).unsqueeze(0)

        ref_tensor = ref_tensor.to(self.device)
        res_tensor = res_tensor.to(self.device)

        with torch.no_grad():
            lpips_tensor = _lpips_model(ref_tensor, res_tensor)

        lpips_value = float(lpips_tensor.cpu().numpy().mean())
        clamped = min(lpips_value, self.lpips_threshold)
        normalized = 1.0 - (clamped / self.lpips_threshold)

        return lpips_value, normalized
