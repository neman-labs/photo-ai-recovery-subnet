import io
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import torch
from PIL import Image

from imagerecovery.services.image_quality.service import ImageQualityService


class TestImageQualityServiceInitialization:
    """Tests for ImageQualityService initialization."""

    def test_init_with_cpu_device(self):
        """Test initialization with CPU device."""
        service = ImageQualityService(device="cpu")

        assert service.device == "cpu"
        assert service.psnr_max_value == 255.0
        assert service.psnr_upper_bound == 50.0
        assert service.psnr_lower_bound == 10.0
        assert service.ssim_win_size == 11
        assert service.lpips_threshold == 0.5

    @patch("torch.cuda.is_available")
    def test_init_with_auto_device_cuda_available(self, mock_cuda):
        """Test initialization with automatic device selection when CUDA is available."""
        mock_cuda.return_value = True

        with patch("imagerecovery.services.image_quality.service.lpips.LPIPS") as mock_lpips:
            mock_model = MagicMock()
            mock_lpips.return_value = mock_model

            service = ImageQualityService()

            assert service.device == "cuda"
            mock_lpips.assert_called_once_with(net="alex", verbose=False)
            mock_model.to.assert_called_once_with("cuda")
            mock_model.eval.assert_called_once()

    @patch("torch.cuda.is_available")
    def test_init_with_auto_device_cuda_not_available(self, mock_cuda):
        """Test initialization with automatic device selection when CUDA is not available."""
        mock_cuda.return_value = False

        with patch("imagerecovery.services.image_quality.service.lpips.LPIPS") as mock_lpips:
            mock_model = MagicMock()
            mock_lpips.return_value = mock_model

            service = ImageQualityService()

            assert service.device == "cpu"

    def test_weights_sum_to_one(self):
        """Test that metric weights sum to 1.0."""
        total_weight = ImageQualityService.PSNR_WEIGHT + ImageQualityService.SSIM_WEIGHT + ImageQualityService.LPIPS_WEIGHT
        assert abs(total_weight - 1.0) < 0.001


class TestImageQualityServicePSNR:
    """Tests for PSNR calculation."""

    @patch("imagerecovery.services.image_quality.service.lpips.LPIPS")
    def test_calculate_psnr_identical_images_returns_infinity(self, mock_lpips):
        """Test PSNR calculation for identical images."""
        mock_lpips.return_value = MagicMock()
        service = ImageQualityService()

        # Create identical images
        img = np.ones((100, 100, 3), dtype=np.uint8) * 128

        psnr_value, normalized_score = service._calculate_psnr(img, img)

        assert psnr_value == float("inf")
        assert normalized_score == 1.0

    @patch("imagerecovery.services.image_quality.service.lpips.LPIPS")
    def test_calculate_psnr_different_images_returns_valid_range(self, mock_lpips):
        """Test PSNR calculation for different images."""
        mock_lpips.return_value = MagicMock()
        service = ImageQualityService()

        # Create different images
        img1 = np.ones((100, 100, 3), dtype=np.uint8) * 100
        img2 = np.ones((100, 100, 3), dtype=np.uint8) * 150

        psnr_value, normalized_score = service._calculate_psnr(img1, img2)

        assert isinstance(psnr_value, float)
        assert 0.0 <= normalized_score <= 1.0
        assert psnr_value > 0

    @patch("imagerecovery.services.image_quality.service.lpips.LPIPS")
    def test_calculate_psnr_clamping(self, mock_lpips):
        """Test PSNR value clamping."""
        mock_lpips.return_value = MagicMock()
        service = ImageQualityService()

        # Create images with large difference (low PSNR)
        img1 = np.ones((100, 100, 3), dtype=np.uint8) * 50
        img2 = np.ones((100, 100, 3), dtype=np.uint8) * 200  # Large difference

        psnr_value, normalized_score = service._calculate_psnr(img1, img2)

        # PSNR value can be below lower bound (will be clamped for normalization)
        # but normalized score should be within bounds
        assert isinstance(psnr_value, float)
        assert 0.0 <= normalized_score <= 1.0

    @pytest.mark.parametrize(
        "noise_level,expected_range",
        [
            (10, (0.3, 0.8)),  # Low noise - medium-high score
            (50, (0.05, 0.4)),  # Medium noise - low-medium score
            (100, (0.0, 0.2)),  # High noise - very low score
        ],
    )
    @patch("imagerecovery.services.image_quality.service.lpips.LPIPS")
    def test_calculate_psnr_with_different_noise_levels(self, mock_lpips, noise_level, expected_range):
        """Test PSNR with different noise levels."""
        mock_lpips.return_value = MagicMock()
        service = ImageQualityService()

        # Create reference image
        img1 = np.ones((100, 100, 3), dtype=np.uint8) * 128

        # Add noise
        noise = np.random.normal(0, noise_level, img1.shape)
        img2 = np.clip(img1.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        _, normalized_score = service._calculate_psnr(img1, img2)

        assert expected_range[0] <= normalized_score <= expected_range[1]


class TestImageQualityServiceSSIM:
    """Tests for SSIM calculation."""

    @patch("imagerecovery.services.image_quality.service.lpips.LPIPS")
    def test_calculate_ssim_identical_images_returns_one(self, mock_lpips):
        """Test SSIM calculation for identical images."""
        mock_lpips.return_value = MagicMock()
        service = ImageQualityService()

        # Create identical images
        img = np.ones((100, 100, 3), dtype=np.uint8) * 128

        with patch("imagerecovery.services.image_quality.service.ssim") as mock_ssim:
            mock_ssim.return_value = torch.tensor(1.0)

            ssim_value, normalized_score = service._calculate_ssim(img, img)

            assert ssim_value == 1.0
            assert normalized_score == 1.0

    @patch("imagerecovery.services.image_quality.service.lpips.LPIPS")
    def test_calculate_ssim_different_images(self, mock_lpips):
        """Test SSIM calculation for different images."""
        mock_lpips.return_value = MagicMock()
        service = ImageQualityService()

        img1 = np.ones((100, 100, 3), dtype=np.uint8) * 100
        img2 = np.ones((100, 100, 3), dtype=np.uint8) * 200

        with patch("imagerecovery.services.image_quality.service.ssim") as mock_ssim:
            mock_ssim.return_value = torch.tensor(0.5)

            ssim_value, normalized_score = service._calculate_ssim(img1, img2)

            assert ssim_value == 0.5
            assert normalized_score == 0.5

    @patch("imagerecovery.services.image_quality.service.lpips.LPIPS")
    def test_calculate_ssim_mismatched_dimensions_raises_error(self, mock_lpips):
        """Test SSIM calculation with mismatched image dimensions."""
        mock_lpips.return_value = MagicMock()
        service = ImageQualityService()

        img1 = np.ones((100, 100, 3), dtype=np.uint8)
        img2 = np.ones((200, 200, 3), dtype=np.uint8)

        with pytest.raises(ValueError, match="Image dimensions must match"):
            service._calculate_ssim(img1, img2)

    @patch("imagerecovery.services.image_quality.service.lpips.LPIPS")
    def test_calculate_ssim_tensor_conversion(self, mock_lpips):
        """Test correct tensor conversion for SSIM."""
        mock_lpips.return_value = MagicMock()
        service = ImageQualityService()

        img = np.ones((64, 64, 3), dtype=np.uint8) * 128

        with patch("imagerecovery.services.image_quality.service.ssim") as mock_ssim:
            mock_ssim.return_value = torch.tensor(0.95)

            service._calculate_ssim(img, img)

            # Check that ssim was called with correct parameters
            mock_ssim.assert_called_once()
            args, kwargs = mock_ssim.call_args

            # Check tensor shape (N, C, H, W)
            assert args[0].shape == (1, 3, 64, 64)
            assert args[1].shape == (1, 3, 64, 64)
            assert kwargs["win_size"] == 11
            assert kwargs["data_range"] == 255


class TestImageQualityServiceLPIPS:
    """Tests for LPIPS calculation."""

    @patch("imagerecovery.services.image_quality.service.lpips.LPIPS")
    def test_calculate_lpips_identical_images_returns_zero(self, mock_lpips_class):
        """Test LPIPS calculation for identical images."""
        # Setup mock LPIPS model instance
        mock_model_instance = MagicMock()
        mock_model_instance.return_value = torch.tensor([[[[0.0]]]])
        mock_lpips_class.return_value = mock_model_instance

        # Mock the global model to return our mock instance
        import imagerecovery.services.image_quality.service as service_module

        service_module._lpips_model = mock_model_instance
        service_module._lpips_device = "cpu"

        service = ImageQualityService(device="cpu")

        # Create identical images
        img = np.ones((100, 100, 3), dtype=np.uint8) * 128

        lpips_value, normalized_score = service._calculate_lpips(img, img)

        assert abs(lpips_value - 0.0) < 0.001
        assert abs(normalized_score - 1.0) < 0.001

    @patch("imagerecovery.services.image_quality.service.lpips.LPIPS")
    def test_calculate_lpips_different_images(self, mock_lpips_class):
        """Test LPIPS calculation for different images."""
        # Setup mock LPIPS model instance
        mock_model_instance = MagicMock()
        mock_model_instance.return_value = torch.tensor([[[[0.3]]]])
        mock_lpips_class.return_value = mock_model_instance

        # Mock the global model to return our mock instance
        import imagerecovery.services.image_quality.service as service_module

        service_module._lpips_model = mock_model_instance
        service_module._lpips_device = "cpu"

        service = ImageQualityService(device="cpu")

        img1 = np.ones((100, 100, 3), dtype=np.uint8) * 100
        img2 = np.ones((100, 100, 3), dtype=np.uint8) * 200

        lpips_value, normalized_score = service._calculate_lpips(img1, img2)

        assert abs(lpips_value - 0.3) < 0.001
        expected_normalized = 1.0 - (0.3 / service.lpips_threshold)
        assert abs(normalized_score - expected_normalized) < 0.001

    @patch("imagerecovery.services.image_quality.service.lpips.LPIPS")
    def test_calculate_lpips_clamping(self, mock_lpips_class):
        """Test LPIPS value clamping."""
        # Setup mock LPIPS model instance
        mock_model_instance = MagicMock()
        mock_model_instance.return_value = torch.tensor([[[[0.8]]]])  # Above threshold
        mock_lpips_class.return_value = mock_model_instance

        # Mock the global model to return our mock instance
        import imagerecovery.services.image_quality.service as service_module

        service_module._lpips_model = mock_model_instance
        service_module._lpips_device = "cpu"

        service = ImageQualityService(device="cpu")

        img1 = np.ones((100, 100, 3), dtype=np.uint8) * 50
        img2 = np.ones((100, 100, 3), dtype=np.uint8) * 250

        lpips_value, normalized_score = service._calculate_lpips(img1, img2)

        assert abs(lpips_value - 0.8) < 0.001
        assert abs(normalized_score - 0.0) < 0.001  # Clamped to threshold

    @patch("imagerecovery.services.image_quality.service.lpips.LPIPS")
    def test_calculate_lpips_mismatched_dimensions_raises_error(self, mock_lpips_class):
        """Test LPIPS calculation with mismatched dimensions."""
        mock_model = MagicMock()
        mock_lpips_class.return_value = mock_model

        service = ImageQualityService()

        img1 = np.ones((100, 100, 3), dtype=np.uint8)
        img2 = np.ones((200, 200, 3), dtype=np.uint8)

        with pytest.raises(ValueError, match="Image dimensions must match"):
            service._calculate_lpips(img1, img2)

    @patch("imagerecovery.services.image_quality.service.lpips.LPIPS")
    def test_calculate_lpips_tensor_preprocessing(self, mock_lpips_class):
        """Test correct tensor preprocessing for LPIPS."""
        mock_model_instance = MagicMock()
        mock_model_instance.return_value = torch.tensor([[[[0.1]]]])
        mock_lpips_class.return_value = mock_model_instance

        # Mock the global model to return our mock instance
        import imagerecovery.services.image_quality.service as service_module

        service_module._lpips_model = mock_model_instance
        service_module._lpips_device = "cpu"

        service = ImageQualityService(device="cpu")

        img = np.ones((64, 64, 3), dtype=np.uint8) * 128

        service._calculate_lpips(img, img)

        # Check that model was called
        mock_model_instance.assert_called_once()
        args = mock_model_instance.call_args[0]

        # Check tensor shape (N, C, H, W)
        assert args[0].shape == (1, 3, 64, 64)
        assert args[1].shape == (1, 3, 64, 64)

        # Check preprocessing (should be in [-1, 1] range)
        assert args[0].min() >= -1.0
        assert args[0].max() <= 1.0


class TestImageQualityServiceBytesConversion:
    """Tests for bytes to numpy conversion."""

    @patch("imagerecovery.services.image_quality.service.lpips.LPIPS")
    def test_bytes_to_numpy_rgb_image(self, mock_lpips):
        """Test conversion of RGB image bytes to numpy."""
        mock_lpips.return_value = MagicMock()
        service = ImageQualityService()

        # Create test RGB image
        img = Image.new("RGB", (64, 64), color=(100, 150, 200))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        img_bytes = buffer.getvalue()

        result = service._bytes_to_numpy(img_bytes)

        assert isinstance(result, np.ndarray)
        assert result.shape == (64, 64, 3)
        assert result.dtype == np.uint8

    @patch("imagerecovery.services.image_quality.service.lpips.LPIPS")
    def test_bytes_to_numpy_grayscale_converted_to_rgb(self, mock_lpips):
        """Test conversion of grayscale image bytes to RGB numpy."""
        mock_lpips.return_value = MagicMock()
        service = ImageQualityService()

        # Create test grayscale image
        img = Image.new("L", (32, 32), color=128)
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        img_bytes = buffer.getvalue()

        result = service._bytes_to_numpy(img_bytes)

        assert isinstance(result, np.ndarray)
        assert result.shape == (32, 32, 3)  # Converted to RGB
        assert result.dtype == np.uint8


class TestImageQualityServiceCalculateScore:
    """Tests for the main calculate_score method."""

    @patch("imagerecovery.services.image_quality.service.lpips.LPIPS")
    def test_calculate_score_success(self, mock_lpips_class):
        """Test successful score calculation."""
        # Setup mock LPIPS model
        mock_model = MagicMock()
        mock_model.return_value = torch.tensor([[[[0.2]]]])
        mock_lpips_class.return_value = mock_model

        service = ImageQualityService()

        # Create test images
        img1 = Image.new("RGB", (64, 64), color=(100, 100, 100))
        img2 = Image.new("RGB", (64, 64), color=(120, 120, 120))

        buffer1 = io.BytesIO()
        buffer2 = io.BytesIO()
        img1.save(buffer1, format="JPEG")
        img2.save(buffer2, format="JPEG")

        with patch.object(service, "_calculate_psnr", return_value=(30.0, 0.5)), patch.object(
            service, "_calculate_ssim", return_value=(0.8, 0.8)
        ), patch.object(service, "_calculate_lpips", return_value=(0.2, 0.6)):
            score = service.calculate_score(buffer1.getvalue(), buffer2.getvalue(), miner_uid=1)

            # Verify weighted combination
            expected_score = service.PSNR_WEIGHT * 0.5 + service.SSIM_WEIGHT * 0.8 + service.LPIPS_WEIGHT * 0.6
            assert abs(score - expected_score) < 0.001

    @patch("imagerecovery.services.image_quality.service.lpips.LPIPS")
    def test_calculate_score_exception_returns_zero(self, mock_lpips):
        """Test that exceptions in score calculation return 0.0."""
        mock_lpips.return_value = MagicMock()
        service = ImageQualityService()

        # Invalid image bytes
        invalid_bytes = b"invalid image data"
        valid_img = Image.new("RGB", (64, 64))
        buffer = io.BytesIO()
        valid_img.save(buffer, format="JPEG")

        score = service.calculate_score(invalid_bytes, buffer.getvalue(), miner_uid=1)

        assert score == 0.0

    @patch("imagerecovery.services.image_quality.service.lpips.LPIPS")
    def test_calculate_weighted_score_all_metrics(self, mock_lpips_class):
        """Test weighted score calculation with all metrics."""
        mock_model = MagicMock()
        mock_lpips_class.return_value = mock_model

        service = ImageQualityService()

        img = np.ones((64, 64, 3), dtype=np.uint8) * 128

        with patch.object(service, "_calculate_psnr", return_value=(35.0, 0.7)), patch.object(
            service, "_calculate_ssim", return_value=(0.9, 0.9)
        ), patch.object(service, "_calculate_lpips", return_value=(0.1, 0.8)):
            score = service._calculate_weighted_score(img, img, miner_uid=1)

            expected = 0.1 * 0.7 + 0.1 * 0.9 + 0.8 * 0.8
            assert abs(score - expected) < 0.001


class TestImageQualityServiceIntegration:
    """Integration tests for ImageQualityService."""

    @pytest.mark.integration
    @patch("imagerecovery.services.image_quality.service.lpips.LPIPS")
    def test_full_pipeline_identical_images(self, mock_lpips_class):
        """Test full pipeline with identical images."""
        # Setup mock LPIPS model instance
        mock_model_instance = MagicMock()
        mock_model_instance.return_value = torch.tensor([[[[0.0]]]])
        mock_lpips_class.return_value = mock_model_instance

        # Mock the global model to return our mock instance
        import imagerecovery.services.image_quality.service as service_module

        service_module._lpips_model = mock_model_instance
        service_module._lpips_device = "cpu"

        service = ImageQualityService(device="cpu")

        # Create identical images
        img = Image.new("RGB", (128, 128), color=(128, 128, 128))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        img_bytes = buffer.getvalue()

        with patch("imagerecovery.services.image_quality.service.ssim") as mock_ssim:
            mock_ssim.return_value = torch.tensor(1.0)

            score = service.calculate_score(img_bytes, img_bytes, miner_uid=1)

            # Calculate expected score: PSNR=1.0 (inf), SSIM=1.0, LPIPS=1.0
            # Score = 0.1*1.0 + 0.1*1.0 + 0.8*1.0 = 1.0
            assert score > 0.95

    @pytest.mark.integration
    @patch("imagerecovery.services.image_quality.service.lpips.LPIPS")
    def test_full_pipeline_different_images(self, mock_lpips_class):
        """Test full pipeline with different images."""
        # Setup mock LPIPS model instance
        mock_model_instance = MagicMock()
        mock_model_instance.return_value = torch.tensor([[[[0.4]]]])
        mock_lpips_class.return_value = mock_model_instance

        # Mock the global model to return our mock instance
        import imagerecovery.services.image_quality.service as service_module

        service_module._lpips_model = mock_model_instance
        service_module._lpips_device = "cpu"

        service = ImageQualityService(device="cpu")

        # Create different images
        img1 = Image.new("RGB", (128, 128), color=(50, 50, 50))
        img2 = Image.new("RGB", (128, 128), color=(200, 200, 200))

        buffer1 = io.BytesIO()
        buffer2 = io.BytesIO()
        img1.save(buffer1, format="JPEG")
        img2.save(buffer2, format="JPEG")

        with patch("imagerecovery.services.image_quality.service.ssim") as mock_ssim:
            mock_ssim.return_value = torch.tensor(0.3)

            score = service.calculate_score(buffer1.getvalue(), buffer2.getvalue(), miner_uid=1)

            # Calculate expected score with weights
            # PSNR for very different solid colors will be low
            # SSIM = 0.3, LPIPS = 0.4 -> LPIPS_normalized = 1.0 - (0.4/0.5) = 0.2
            # Total should be lower than 0.5
            assert score < 0.5


class TestImageQualityServiceDeviceHandling:
    """Tests for device handling and LPIPS model management."""

    @patch("imagerecovery.services.image_quality.service.lpips.LPIPS")
    def test_lpips_model_singleton(self, mock_lpips_class):
        """Test that LPIPS model is reused across instances."""
        mock_model1 = MagicMock()
        mock_model2 = MagicMock()
        mock_lpips_class.side_effect = [mock_model1, mock_model2]

        # Reset global model
        import imagerecovery.services.image_quality.service as service_module

        service_module._lpips_model = None
        service_module._lpips_device = None

        service1 = ImageQualityService(device="cpu")
        service2 = ImageQualityService(device="cpu")

        # Should only create model once
        assert mock_lpips_class.call_count == 1

    @patch("imagerecovery.services.image_quality.service.lpips.LPIPS")
    def test_lpips_model_recreated_for_different_device(self, mock_lpips_class):
        """Test that LPIPS model is recreated when device changes."""
        mock_model1 = MagicMock()
        mock_model2 = MagicMock()
        mock_lpips_class.side_effect = [mock_model1, mock_model2]

        # Reset global model
        import imagerecovery.services.image_quality.service as service_module

        service_module._lpips_model = None
        service_module._lpips_device = None

        service1 = ImageQualityService(device="cpu")
        service2 = ImageQualityService(device="cuda")

        # Should create model twice for different devices
        assert mock_lpips_class.call_count == 2


@pytest.mark.parametrize("image_format", ["JPEG", "PNG", "BMP"])
class TestImageQualityServiceFormats:
    """Tests for different image formats."""

    @patch("imagerecovery.services.image_quality.service.lpips.LPIPS")
    def test_different_image_formats(self, mock_lpips, image_format):
        """Test service works with different image formats."""
        mock_lpips.return_value = MagicMock()
        service = ImageQualityService()

        # Create test image
        img = Image.new("RGB", (64, 64), color=(100, 150, 200))
        buffer = io.BytesIO()
        img.save(buffer, format=image_format)
        img_bytes = buffer.getvalue()

        result = service._bytes_to_numpy(img_bytes)

        assert isinstance(result, np.ndarray)
        assert result.shape == (64, 64, 3)
