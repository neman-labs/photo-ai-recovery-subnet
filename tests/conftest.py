import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
from PIL import Image

# Moved ImageCache import to specific tests to avoid OpenCV import issues


@pytest.fixture(scope="session")
def test_images_dir():
    """Create a temporary directory with test images for the entire test session."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create subdirectories
        (temp_path / "original").mkdir()
        (temp_path / "degraded").mkdir()
        (temp_path / "restored").mkdir()

        # Create some test .dng files
        dng_files = ["test1.dng", "test2.dng", "test3.dng"]
        for filename in dng_files:
            (temp_path / filename).touch()

        yield temp_path


@pytest.fixture
def sample_rgb_image():
    """Create a sample RGB PIL Image for testing."""
    # Create a 256x256 RGB image with gradient pattern
    width, height = 256, 256
    image_array = np.zeros((height, width, 3), dtype=np.uint8)

    # Create gradient pattern
    for i in range(height):
        for j in range(width):
            image_array[i, j] = [i % 256, j % 256, (i + j) % 256]

    return Image.fromarray(image_array, "RGB")


@pytest.fixture
def sample_grayscale_image():
    """Create a sample grayscale PIL Image for testing."""
    width, height = 128, 128
    image_array = np.random.randint(0, 256, (height, width), dtype=np.uint8)
    return Image.fromarray(image_array, "L")


@pytest.fixture
def degraded_image_noisy(sample_rgb_image):
    """Create a degraded (noisy) version of the sample image."""
    image_array = np.array(sample_rgb_image)

    # Add random noise
    noise = np.random.normal(0, 25, image_array.shape).astype(np.int16)
    noisy_array = np.clip(image_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return Image.fromarray(noisy_array, "RGB")


@pytest.fixture
def mock_bittensor_metagraph():
    """Create a mock Bittensor metagraph for testing."""
    from tests.helpers import MetagraphStub

    return MetagraphStub(neurons_count=10)


@pytest.fixture
def mock_image_synapse():
    """Create a mock ImageSynapse for testing."""
    import base64
    from io import BytesIO

    from imagerecovery.protocol import ImageSynapse

    # Create a simple test image as base64
    test_image = Image.new("RGB", (64, 64), color="red")
    buffer = BytesIO()
    test_image.save(buffer, format="JPEG")
    image_b64 = base64.b64encode(buffer.getvalue()).decode()

    return ImageSynapse(
        image_to_restore=image_b64,
        task_name="RestorationComparison",
        target_metadata={"width": 64, "height": 64, "format": "JPEG"},
    )


@pytest.fixture
def mock_image_cache_ready():
    """Create a mock ImageCache in ready state for testing."""
    from imagerecovery.services.cache.cache import ImageCache

    with tempfile.TemporaryDirectory() as temp_dir:
        cache = ImageCache()
        cache.dataset_cache_path = temp_dir
        cache.image_files = [
            str(Path(temp_dir) / "test1.dng"),
            str(Path(temp_dir) / "test2.dng"),
            str(Path(temp_dir) / "test3.dng"),
        ]
        cache._ready = True

        # Create actual test files
        for file_path in cache.image_files:
            Path(file_path).touch()

        yield cache


@pytest.fixture
def mock_opencv_videocapture():
    """Mock OpenCV VideoCapture for DNG file testing."""
    with patch("imagerecovery.services.cache.cache.cv2.VideoCapture") as mock_cv2:
        mock_cap = Mock()
        mock_cap.read.return_value = (True, np.ones((100, 100, 3), dtype=np.uint8) * 255)
        mock_cap.release = Mock()
        mock_cv2.return_value = mock_cap
        yield mock_cv2


@pytest.fixture
def mock_huggingface_api():
    """Mock HuggingFace API for dataset operations."""
    with patch("imagerecovery.services.cache.cache.HfApi") as mock_api_class, patch(
        "imagerecovery.services.cache.cache.snapshot_download"
    ) as mock_download, patch("imagerecovery.services.cache.cache.try_to_load_from_cache") as mock_try_load:
        # Setup API mock
        mock_api = Mock()
        mock_api.list_repo_files.return_value = ["test1.dng", "test2.dng", "test3.dng"]
        mock_api_class.return_value = mock_api

        # Setup download mock
        mock_download.return_value = "/fake/cache/path"

        # Setup cache check mock
        mock_try_load.return_value = "_CACHED_FILE_NOT_FOUND"

        yield {"api_class": mock_api_class, "api_instance": mock_api, "download": mock_download, "try_load": mock_try_load}


@pytest.fixture
def image_bytes_rgb(sample_rgb_image):
    """Convert sample RGB image to bytes for testing."""
    from io import BytesIO

    buffer = BytesIO()
    sample_rgb_image.save(buffer, format="JPEG")
    return buffer.getvalue()


@pytest.fixture
def image_bytes_grayscale(sample_grayscale_image):
    """Convert sample grayscale image to bytes for testing."""
    from io import BytesIO

    buffer = BytesIO()
    sample_grayscale_image.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def base64_test_image(sample_rgb_image):
    """Convert sample image to base64 string for protocol testing."""
    import base64
    from io import BytesIO

    buffer = BytesIO()
    sample_rgb_image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode()


@pytest.fixture(autouse=True)
def reset_image_cache_singleton():
    """Automatically reset ImageCache singleton before each test."""
    try:
        from imagerecovery.services.cache.cache import ImageCache

        # Store original instance
        original_instance = ImageCache._instance

        # Reset for test
        ImageCache._instance = None

        yield

        # Restore original instance after test
        ImageCache._instance = original_instance
    except ImportError:
        # If ImageCache can't be imported, just yield
        yield


@pytest.fixture
def temp_cache_dir():
    """Create a temporary cache directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


# Test markers configuration
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "unit: Unit tests that test individual components")
    config.addinivalue_line("markers", "integration: Integration tests that test component interactions")
    config.addinivalue_line("markers", "slow: Tests that take longer than 5 seconds to run")
    config.addinivalue_line("markers", "requires_gpu: Tests that require GPU access")
    config.addinivalue_line("markers", "requires_network: Tests that require network access")
