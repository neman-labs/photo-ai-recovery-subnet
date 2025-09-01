import io
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from imagerecovery.services.task_generation.task_generation_service import TaskGenerationService


class TestTaskGenerationServiceInitialization:
    """Tests for TaskGenerationService initialization."""

    def test_init_success(self):
        """Test successful initialization."""
        service = TaskGenerationService()
        assert service is not None


class TestTaskGenerationServiceDownscale:
    """Tests for downscale method."""

    def test_downscale_with_scale_factor(self):
        """Test downscaling with a scale factor."""
        service = TaskGenerationService()

        # Create test image
        original_img = Image.new("RGB", (256, 256), color=(100, 150, 200))

        # Downscale by factor of 4
        result = service.downscale(original_img, scale_factor=4)

        assert isinstance(result, Image.Image)
        assert result.size == (64, 64)

    def test_downscale_with_target_size(self):
        """Test downscaling with a specific target size."""
        service = TaskGenerationService()

        original_img = Image.new("RGB", (512, 512))
        target_size = (128, 128)

        result = service.downscale(original_img, target_size=target_size)

        assert isinstance(result, Image.Image)
        assert result.size == target_size

    def test_downscale_target_size_overrides_scale_factor(self):
        """Test that target_size overrides scale_factor when both are provided."""
        service = TaskGenerationService()

        original_img = Image.new("RGB", (400, 400))

        result = service.downscale(
            original_img,
            scale_factor=2,  # Would give (200, 200)
            target_size=(100, 100),  # Should override
        )

        assert result.size == (100, 100)

    def test_downscale_bytes_input(self):
        """Test downscaling with bytes input."""
        service = TaskGenerationService()

        # Create image and convert to bytes
        original_img = Image.new("RGB", (200, 200))
        buffer = io.BytesIO()
        original_img.save(buffer, format="PNG")
        img_bytes = buffer.getvalue()

        result = service.downscale(img_bytes, scale_factor=2)

        # Should return bytes by default when input is bytes (output_bytes=None)
        assert isinstance(result, bytes)

        # Verify the result
        result_img = Image.open(io.BytesIO(result))
        assert result_img.size == (100, 100)

    def test_downscale_pil_input_bytes_output(self):
        """Test downscaling PIL input with bytes output."""
        service = TaskGenerationService()

        original_img = Image.new("RGB", (160, 160))

        result = service.downscale(original_img, scale_factor=2, output_bytes=True, format="JPEG")

        assert isinstance(result, bytes)

        # Verify it's valid JPEG
        result_img = Image.open(io.BytesIO(result))
        assert result_img.format == "JPEG"
        assert result_img.size == (80, 80)

    def test_downscale_bytes_input_pil_output(self):
        """Test downscaling bytes input with PIL output."""
        service = TaskGenerationService()

        # Create image bytes
        original_img = Image.new("RGB", (300, 300))
        buffer = io.BytesIO()
        original_img.save(buffer, format="PNG")
        img_bytes = buffer.getvalue()

        result = service.downscale(img_bytes, scale_factor=3, output_bytes=False)

        assert isinstance(result, Image.Image)
        assert result.size == (100, 100)

    @pytest.mark.parametrize("scale_factor", [2, 3, 4, 8])
    def test_downscale_different_scale_factors(self, scale_factor):
        """Test downscaling with different scale factors."""
        service = TaskGenerationService()

        original_size = (480, 480)
        original_img = Image.new("RGB", original_size)

        result = service.downscale(original_img, scale_factor=scale_factor)

        expected_size = (original_size[0] // scale_factor, original_size[1] // scale_factor)
        assert result.size == expected_size

    @pytest.mark.parametrize("format", ["PNG", "JPEG", "BMP"])
    def test_downscale_different_formats(self, format):
        """Test downscaling with different output formats."""
        service = TaskGenerationService()

        original_img = Image.new("RGB", (128, 128))

        result = service.downscale(original_img, scale_factor=2, output_bytes=True, format=format)

        assert isinstance(result, bytes)

        # Verify format
        result_img = Image.open(io.BytesIO(result))
        if format == "BMP":
            assert result_img.format in ["BMP", "DIB"]
        else:
            assert result_img.format == format

    def test_downscale_non_divisible_dimensions(self):
        """Test downscaling with dimensions not evenly divisible by scale factor."""
        service = TaskGenerationService()

        # 333 is not divisible by 4
        original_img = Image.new("RGB", (333, 333))

        result = service.downscale(original_img, scale_factor=4)

        # Should be floor division: 333 // 4 = 83
        assert result.size == (83, 83)

    def test_downscale_rectangular_image(self):
        """Test downscaling rectangular (non-square) images."""
        service = TaskGenerationService()

        original_img = Image.new("RGB", (640, 480))

        result = service.downscale(original_img, scale_factor=2)

        assert result.size == (320, 240)

    def test_downscale_with_custom_target_size_rectangular(self):
        """Test downscaling to a custom rectangular target size."""
        service = TaskGenerationService()

        original_img = Image.new("RGB", (800, 600))
        target_size = (200, 150)

        result = service.downscale(original_img, target_size=target_size)

        assert result.size == target_size

    def test_downscale_preserves_image_mode(self):
        """Test that downscaling preserves the image mode."""
        service = TaskGenerationService()

        # Test with RGB
        rgb_img = Image.new("RGB", (100, 100))
        result_rgb = service.downscale(rgb_img, scale_factor=2)
        assert result_rgb.mode == "RGB"

        # Test with L (grayscale)
        gray_img = Image.new("L", (100, 100))
        result_gray = service.downscale(gray_img, scale_factor=2)
        assert result_gray.mode == "L"

    def test_downscale_small_scale_factor(self):
        """Test downscaling with scale factor of 1."""
        service = TaskGenerationService()

        original_img = Image.new("RGB", (256, 256))

        result = service.downscale(original_img, scale_factor=1)

        # Should return same size
        assert result.size == original_img.size


class TestTaskGenerationServiceQuality:
    """Tests for downscaling quality and interpolation."""

    def test_downscale_uses_bilinear_interpolation(self):
        """Test that downscaling uses bilinear interpolation."""
        service = TaskGenerationService()

        # Create a gradient image to test interpolation
        img_array = np.zeros((256, 256, 3), dtype=np.uint8)
        for i in range(256):
            for j in range(256):
                img_array[i, j] = [i, j, 128]

        original_img = Image.fromarray(img_array)

        with patch.object(Image.Image, "resize") as mock_resize:
            mock_resize.return_value = Image.new("RGB", (64, 64))

            service.downscale(original_img, scale_factor=4)

            # Verify bilinear interpolation was used
            mock_resize.assert_called_once_with((64, 64), Image.Resampling.BILINEAR)

    def test_downscale_quality_preservation(self):
        """Test that downscaling preserves relative image quality."""
        service = TaskGenerationService()

        # Create a detailed test image
        img_array = np.zeros((512, 512, 3), dtype=np.uint8)
        # Add some patterns
        for i in range(512):
            for j in range(512):
                img_array[i, j] = [(i * 2) % 256, (j * 2) % 256, ((i + j) * 2) % 256]

        original_img = Image.fromarray(img_array)

        # Downscale
        result = service.downscale(original_img, scale_factor=4)

        assert result.size == (128, 128)

        # Convert back to array and check it's not all zeros
        result_array = np.array(result)
        assert result_array.mean() > 0
        assert result_array.std() > 0  # Has variation


class TestTaskGenerationServiceIntegration:
    """Integration tests for TaskGenerationService."""

    @pytest.mark.integration
    def test_multiple_downscales_pipeline(self):
        """Test multiple sequential downscales."""
        service = TaskGenerationService()

        # Start with large image
        img = Image.new("RGB", (1024, 1024), color=(100, 150, 200))

        # First downscale
        img = service.downscale(img, scale_factor=2)
        assert img.size == (512, 512)

        # Second downscale
        img = service.downscale(img, scale_factor=2)
        assert img.size == (256, 256)

        # Third downscale with target size
        img = service.downscale(img, target_size=(64, 64))
        assert img.size == (64, 64)

    @pytest.mark.integration
    def test_downscale_with_real_image_data(self):
        """Test downscaling with realistic image data."""
        service = TaskGenerationService()

        # Create a realistic test image with patterns
        width, height = 800, 600
        img_array = np.zeros((height, width, 3), dtype=np.uint8)

        # Add gradient
        for i in range(height):
            for j in range(width):
                img_array[i, j] = [int(255 * i / height), int(255 * j / width), int(255 * (i + j) / (height + width))]

        original_img = Image.fromarray(img_array, "RGB")

        # Test various downscale operations
        result1 = service.downscale(original_img, scale_factor=2)
        assert result1.size == (400, 300)

        result2 = service.downscale(original_img, target_size=(200, 150))
        assert result2.size == (200, 150)

        # Test bytes output
        result3 = service.downscale(original_img, scale_factor=4, output_bytes=True, format="PNG")
        assert isinstance(result3, bytes)

        # Verify bytes result
        result3_img = Image.open(io.BytesIO(result3))
        assert result3_img.size == (200, 150)

    @pytest.mark.integration
    @pytest.mark.parametrize(
        "input_format,output_format",
        [
            ("PIL", "PIL"),
            ("PIL", "bytes"),
            ("bytes", "bytes"),
            ("bytes", "PIL"),
        ],
    )
    def test_downscale_format_combinations(self, input_format, output_format):
        """Test all combinations of input and output formats."""
        service = TaskGenerationService()

        # Create base image
        base_img = Image.new("RGB", (256, 256), color=(50, 100, 150))

        # Prepare input
        if input_format == "bytes":
            buffer = io.BytesIO()
            base_img.save(buffer, format="PNG")
            input_data = buffer.getvalue()
        else:
            input_data = base_img

        # Set output format
        output_bytes = output_format == "bytes"

        # Perform downscale
        result = service.downscale(input_data, scale_factor=2, output_bytes=output_bytes)

        # Verify output type
        if output_format == "bytes":
            assert isinstance(result, bytes)
            # Verify it's valid image data
            result_img = Image.open(io.BytesIO(result))
            assert result_img.size == (128, 128)
        else:
            assert isinstance(result, Image.Image)
            assert result.size == (128, 128)
