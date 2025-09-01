import io
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
from PIL import Image

from imagerecovery.services.augmentation.augmentation_service import AugmentationService


class TestAugmentationServiceInitialization:
    """Tests for AugmentationService initialization."""

    def test_init_with_default_probability(self):
        """Test initialization with default augmentation probability."""
        service = AugmentationService()
        assert service.augmentation_prob == 0.7

    def test_init_with_custom_probability(self):
        """Test initialization with custom augmentation probability."""
        service = AugmentationService(augmentation_prob=0.5)
        assert service.augmentation_prob == 0.5

    def test_init_with_invalid_probability_raises_error(self):
        """Test initialization with invalid probability value."""
        with pytest.raises(ValueError, match="augmentation_prob must be between"):
            AugmentationService(augmentation_prob=1.5)

        with pytest.raises(ValueError, match="augmentation_prob must be between"):
            AugmentationService(augmentation_prob=-0.1)

    @pytest.mark.parametrize("prob", [0.0, 0.3, 0.7, 1.0])
    def test_init_with_valid_probabilities(self, prob):
        """Test initialization with various valid probabilities."""
        service = AugmentationService(augmentation_prob=prob)
        assert service.augmentation_prob == prob


class TestAugmentationServiceFlips:
    """Tests for flip transformations."""

    @patch("random.random")
    def test_apply_flips_both_applied(self, mock_random):
        """Test both horizontal and vertical flips are applied."""
        mock_random.return_value = 0.5  # Less than 0.7, so flips should occur
        service = AugmentationService(augmentation_prob=0.7)

        # Create test image
        img = Image.new("RGB", (64, 64), color=(100, 150, 200))

        result_img, params = service._apply_flips(img)

        assert params["horizontal_flip"] is True
        assert params["vertical_flip"] is True
        assert isinstance(result_img, Image.Image)

    @patch("random.random")
    def test_apply_flips_none_applied(self, mock_random):
        """Test no flips are applied when random > probability."""
        mock_random.return_value = 0.8  # Greater than 0.7, so no flips
        service = AugmentationService(augmentation_prob=0.7)

        img = Image.new("RGB", (64, 64), color=(100, 150, 200))

        result_img, params = service._apply_flips(img)

        assert params["horizontal_flip"] is False
        assert params["vertical_flip"] is False

    @patch("random.random")
    def test_apply_flips_horizontal_only(self, mock_random):
        """Test only horizontal flip is applied."""
        mock_random.side_effect = [0.5, 0.8]  # First passes, second fails
        service = AugmentationService(augmentation_prob=0.7)

        img = Image.new("RGB", (64, 64), color=(100, 150, 200))

        result_img, params = service._apply_flips(img)

        assert params["horizontal_flip"] is True
        assert params["vertical_flip"] is False

    def test_apply_flips_creates_copy(self):
        """Test that flips operate on a copy of the image."""
        service = AugmentationService()
        original_img = Image.new("RGB", (32, 32))

        result_img, _ = service._apply_flips(original_img)

        assert result_img is not original_img


class TestAugmentationServiceRotateCrop:
    """Tests for rotation and crop transformations."""

    @patch("random.uniform")
    def test_apply_rotate_crop_transform_success(self, mock_uniform):
        """Test successful rotation and crop."""
        mock_uniform.return_value = 5.0  # 5 degree rotation
        service = AugmentationService()

        img = Image.new("RGB", (128, 128), color=(100, 100, 100))

        result_img, params = service._apply_rotate_crop_transform(img, max_angle=10.0)

        assert params["rotation_applied"] is True
        assert params["rotation_angle"] == 5.0
        assert "crop_dimensions" in params
        assert isinstance(result_img, Image.Image)
        # After rotation and crop, size should be smaller
        assert result_img.size[0] < img.size[0]
        assert result_img.size[1] < img.size[1]

    @patch("random.uniform")
    def test_apply_rotate_crop_with_different_angles(self, mock_uniform):
        """Test rotation with different angles."""
        service = AugmentationService()
        img = Image.new("RGB", (100, 100))

        angles = [-10.0, -5.0, 0.0, 5.0, 10.0]
        for angle in angles:
            mock_uniform.return_value = angle

            result_img, params = service._apply_rotate_crop_transform(img)

            assert params["rotation_angle"] == angle
            assert params["rotation_applied"] is True

    def test_apply_rotate_crop_error_handling(self):
        """Test error handling in rotation and crop."""
        service = AugmentationService()

        # Create an image that might cause issues
        img = Image.new("RGB", (1, 1))  # Very small image

        with patch.object(service, "_rotate_and_crop_without_borders", side_effect=Exception("Test error")):
            result_img, params = service._apply_rotate_crop_transform(img)

            assert result_img is img  # Should return original on error
            assert params["rotation_applied"] is False
            assert params["rotation_angle"] == 0

    def test_rotate_and_crop_without_borders_dimensions(self):
        """Test that rotation and crop maintains proper dimensions."""
        service = AugmentationService()
        img = Image.new("RGB", (200, 150))

        for angle in [5, 10, 15, 20]:
            result_img, params = service._rotate_and_crop_without_borders(img, angle)

            assert isinstance(result_img, Image.Image)
            assert params["rotation_angle"] == angle
            assert "crop_dimensions" in params
            # Cropped image should be smaller than original
            assert result_img.size[0] <= img.size[0]
            assert result_img.size[1] <= img.size[1]


class TestAugmentationServiceOverlay:
    """Tests for overlay transformations."""

    def test_apply_overlay_without_cache_returns_original(self):
        """Test overlay without cache returns original image."""
        service = AugmentationService()
        img = Image.new("RGB", (128, 128))

        result_img, params = service._apply_overlay_transform(img, dataset_cache=None)

        assert result_img is img
        assert params["overlay_applied"] is False

    @patch("random.randint")
    @patch("random.uniform")
    def test_apply_overlay_with_cache_success(self, mock_uniform, mock_randint):
        """Test successful overlay with cache."""
        mock_uniform.return_value = 0.3  # Scale factor
        mock_randint.side_effect = [45, 10, 20]  # angle, pos_x, pos_y

        service = AugmentationService()
        base_img = Image.new("RGB", (200, 200), color=(100, 100, 100))

        # Mock cache
        mock_cache = Mock()
        overlay_img = Image.new("RGB", (100, 100), color=(200, 200, 200))
        mock_cache.get_random_image_pil.return_value = overlay_img

        result_img, params = service._apply_overlay_transform(base_img, dataset_cache=mock_cache)

        assert params["overlay_applied"] is True
        assert params["overlay_angle"] == 45
        assert params["overlay_scale"] == 0.3
        assert params["overlay_position"] == (10, 20)
        assert result_img.size == base_img.size

    def test_apply_overlay_cache_returns_none(self):
        """Test overlay when cache returns None."""
        service = AugmentationService()
        img = Image.new("RGB", (128, 128))

        mock_cache = Mock()
        mock_cache.get_random_image_pil.return_value = None

        result_img, params = service._apply_overlay_transform(img, dataset_cache=mock_cache)

        assert result_img is img
        assert params["overlay_applied"] is False

    @patch("random.uniform")
    def test_apply_overlay_different_sizes(self, mock_uniform):
        """Test overlay with different size factors."""
        service = AugmentationService()
        base_img = Image.new("RGB", (300, 300))

        mock_cache = Mock()
        overlay_img = Image.new("RGB", (150, 150))
        mock_cache.get_random_image_pil.return_value = overlay_img

        for scale in [0.1, 0.3, 0.5]:
            mock_uniform.return_value = scale

            result_img, params = service._apply_overlay_transform(
                base_img, dataset_cache=mock_cache, min_overlay_size=scale, max_overlay_size=scale
            )

            assert params["overlay_scale"] == scale
            assert params["overlay_applied"] is True

    def test_apply_overlay_error_handling(self):
        """Test overlay error handling."""
        service = AugmentationService()
        img = Image.new("RGB", (128, 128))

        mock_cache = Mock()
        mock_cache.get_random_image_pil.side_effect = Exception("Cache error")

        result_img, params = service._apply_overlay_transform(img, dataset_cache=mock_cache)

        assert result_img is img
        assert params["overlay_applied"] is False


class TestAugmentationServiceApplyTransforms:
    """Tests for the main apply_transforms method."""

    def test_apply_transforms_pil_image_input(self):
        """Test transforms with PIL Image input."""
        service = AugmentationService()
        img = Image.new("RGB", (128, 128), color=(100, 150, 200))

        result_img, params = service.apply_transforms(img)

        assert isinstance(result_img, Image.Image)
        assert "horizontal_flip" in params
        assert "vertical_flip" in params
        assert "rotation_applied" in params

    def test_apply_transforms_bytes_input(self):
        """Test transforms with bytes input."""
        service = AugmentationService()
        img = Image.new("RGB", (64, 64))
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        img_bytes = buffer.getvalue()

        result_bytes, params = service.apply_transforms(img_bytes)

        assert isinstance(result_bytes, bytes)
        assert "horizontal_flip" in params
        assert "vertical_flip" in params

    def test_apply_transforms_path_input(self):
        """Test transforms with file path input."""
        import tempfile

        service = AugmentationService()

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            img = Image.new("RGB", (64, 64))
            img.save(tmp.name)

            result_img, params = service.apply_transforms(tmp.name)

            assert isinstance(result_img, Image.Image)
            assert "horizontal_flip" in params

    def test_apply_transforms_invalid_path_raises_error(self):
        """Test transforms with invalid file path."""
        service = AugmentationService()

        with pytest.raises(ValueError, match="Could not open image file"):
            service.apply_transforms("/nonexistent/file.png")

    def test_apply_transforms_output_bytes_true(self):
        """Test transforms with output_bytes=True."""
        service = AugmentationService()
        img = Image.new("RGB", (64, 64))

        result, params = service.apply_transforms(img, output_bytes=True, format="JPEG")

        assert isinstance(result, bytes)
        # Verify it's valid JPEG
        recovered_img = Image.open(io.BytesIO(result))
        assert recovered_img.format == "JPEG"

    def test_apply_transforms_with_all_options(self):
        """Test transforms with all options enabled."""
        service = AugmentationService(augmentation_prob=1.0)
        img = Image.new("RGB", (128, 128))

        mock_cache = Mock()
        overlay_img = Image.new("RGB", (64, 64))
        mock_cache.get_random_image_pil.return_value = overlay_img

        result_img, params = service.apply_transforms(img, output_bytes=False, dataset_cache=mock_cache)

        assert isinstance(result_img, Image.Image)
        assert "horizontal_flip" in params
        assert "vertical_flip" in params
        assert "rotation_applied" in params
        assert "overlay_applied" in params

    def test_apply_transforms_creates_copy(self):
        """Test that transforms work on a copy of the image."""
        service = AugmentationService()
        original_img = Image.new("RGB", (64, 64))

        result_img, _ = service.apply_transforms(original_img)

        assert result_img is not original_img

    def test_apply_transforms_invalid_type_raises_error(self):
        """Test transforms with invalid input type."""
        service = AugmentationService()

        with pytest.raises(TypeError, match="Expected PIL Image"):
            service.apply_transforms(123)  # Invalid type

    @pytest.mark.parametrize("format", ["PNG", "JPEG", "BMP"])
    def test_apply_transforms_different_formats(self, format):
        """Test transforms with different output formats."""
        service = AugmentationService()
        img = Image.new("RGB", (64, 64))

        result_bytes, params = service.apply_transforms(img, output_bytes=True, format=format)

        # Verify format
        recovered_img = Image.open(io.BytesIO(result_bytes))
        if format == "BMP":
            # BMP doesn't store format info the same way
            assert recovered_img.format in ["BMP", "DIB"]
        else:
            assert recovered_img.format == format


class TestAugmentationServiceIntegration:
    """Integration tests for AugmentationService."""

    @pytest.mark.integration
    def test_full_augmentation_pipeline(self):
        """Test complete augmentation pipeline."""
        service = AugmentationService(augmentation_prob=0.5)

        # Create test image with pattern
        img_array = np.zeros((100, 100, 3), dtype=np.uint8)
        for i in range(100):
            for j in range(100):
                img_array[i, j] = [i * 2, j * 2, (i + j)]

        img = Image.fromarray(img_array)

        # Mock cache with overlay image
        mock_cache = Mock()
        overlay_img = Image.new("RGB", (50, 50), color=(255, 0, 0))
        mock_cache.get_random_image_pil.return_value = overlay_img

        # Run multiple times to test randomness
        for _ in range(5):
            result_img, params = service.apply_transforms(img, dataset_cache=mock_cache)

            assert isinstance(result_img, Image.Image)
            assert all(key in params for key in ["horizontal_flip", "vertical_flip", "rotation_applied", "overlay_applied"])

    @pytest.mark.integration
    def test_augmentation_preserves_image_quality(self):
        """Test that augmentation preserves basic image properties."""
        service = AugmentationService()

        # Create high-quality test image
        img = Image.new("RGB", (256, 256))
        pixels = img.load()
        for i in range(256):
            for j in range(256):
                pixels[j, i] = (i, j, (i + j) % 256)

        result_img, _ = service.apply_transforms(img)

        # Check that result is valid
        assert result_img.mode == "RGB"
        assert result_img.size[0] > 0
        assert result_img.size[1] > 0

    @pytest.mark.integration
    @pytest.mark.parametrize("size", [(64, 64), (128, 256), (512, 512)])
    def test_augmentation_different_image_sizes(self, size):
        """Test augmentation with different image sizes."""
        service = AugmentationService()
        img = Image.new("RGB", size)

        result_img, params = service.apply_transforms(img)

        assert isinstance(result_img, Image.Image)
        # After rotation and crop, size should change
        if params["rotation_applied"]:
            assert result_img.size != size
