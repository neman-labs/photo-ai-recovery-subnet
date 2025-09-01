import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
from PIL import Image

from imagerecovery.services.cache.cache import DEFAULT_CACHE_DIR, DEFAULT_DATASET_PATH, ImageCache


class TestImageCacheSingleton:
    """Tests for ImageCache singleton pattern."""

    def test_get_instance_returns_same_object(self):
        """Test that get_instance returns the same object."""
        # Reset singleton before test
        ImageCache._instance = None

        instance1 = ImageCache.get_instance()
        instance2 = ImageCache.get_instance()

        assert instance1 is instance2

    def test_get_instance_with_different_params_returns_same_object(self):
        """Test that get_instance with different parameters returns the same object."""
        # Reset singleton before test
        ImageCache._instance = None

        instance1 = ImageCache.get_instance("dataset1", "cache1")
        instance2 = ImageCache.get_instance("dataset2", "cache2")

        assert instance1 is instance2
        assert instance1.dataset_path == "dataset1"  # First parameters are preserved


class TestImageCacheInitialization:
    """Tests for ImageCache initialization."""

    def test_init_with_defaults(self):
        """Test initialization with default parameters."""
        cache = ImageCache()

        assert cache.dataset_path == DEFAULT_DATASET_PATH
        assert cache.cache_dir == DEFAULT_CACHE_DIR
        assert cache.dataset_cache_path is None
        assert cache.image_files == []
        assert cache._ready is False

    def test_init_with_custom_params(self):
        """Test initialization with custom parameters."""
        dataset_path = "custom/dataset"
        cache_dir = "/custom/cache"

        cache = ImageCache(dataset_path, cache_dir)

        assert cache.dataset_path == dataset_path
        assert cache.cache_dir == cache_dir
        assert cache.dataset_cache_path is None
        assert cache.image_files == []
        assert cache._ready is False


class TestImageCacheReadyStatus:
    """Tests for cache readiness checks."""

    def test_is_ready_false_when_not_initialized(self):
        """Test that is_ready returns False for uninitialized cache."""
        cache = ImageCache()
        assert cache.is_ready() is False

    def test_is_ready_true_when_ready_flag_set(self):
        """Test that is_ready returns True when _ready is set."""
        cache = ImageCache()
        cache._ready = True

        assert cache.is_ready() is True

    def test_is_ready_checks_cache_path_exists(self):
        """Test that is_ready checks cache path existence."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = ImageCache()
            cache.dataset_cache_path = temp_dir
            cache.image_files = ["test.dng"]

            # Path exists
            assert cache.is_ready() is True
            assert cache._ready is True

    def test_is_ready_false_when_cache_path_not_exists(self):
        """Test that is_ready returns False if cache path doesn't exist."""
        cache = ImageCache()
        cache.dataset_cache_path = "/nonexistent/path"
        cache.image_files = ["test.dng"]

        assert cache.is_ready() is False

    @patch.object(ImageCache, "_check_existing_cache")
    def test_is_ready_calls_check_existing_cache(self, mock_check):
        """Test that is_ready calls _check_existing_cache when cache is not ready."""
        cache = ImageCache()
        cache.is_ready()

        mock_check.assert_called_once()


class TestImageCacheHuggingFaceAPI:
    """Tests for HuggingFace API interactions."""

    @patch("imagerecovery.services.cache.cache.HfApi")
    def test_get_expected_file_count_success(self, mock_hf_api):
        """Test successful file count retrieval via API."""
        mock_api = Mock()
        mock_api.list_repo_files.return_value = ["file1.dng", "file2.dng", "file3.txt", "file4.DNG"]
        mock_hf_api.return_value = mock_api

        cache = ImageCache()
        count = cache._get_expected_file_count("test/dataset")

        assert count == 3  # Only .dng files (including .DNG)
        mock_api.list_repo_files.assert_called_once_with(repo_id="test/dataset", repo_type="dataset")

    @patch("imagerecovery.services.cache.cache.HfApi")
    def test_get_expected_file_count_api_error(self, mock_hf_api):
        """Test API error handling."""
        mock_hf_api.side_effect = Exception("API Error")

        cache = ImageCache()
        count = cache._get_expected_file_count("test/dataset")

        assert count == 0


class TestImageCacheFileVerification:
    """Tests for file verification in cache."""

    def test_load_and_verify_images_success(self):
        """Test successful file loading and verification."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test .dng files
            test_files = ["test1.dng", "test2.dng", "test3.dng"]
            for filename in test_files:
                (Path(temp_dir) / filename).touch()

            cache = ImageCache()
            cache.dataset_cache_path = temp_dir

            # Verify with expected count
            cache._load_and_verify_images(expected_count=3)

            assert len(cache.image_files) == 3
            assert all(f.endswith(".dng") for f in cache.image_files)

    def test_load_and_verify_images_no_files_raises_error(self):
        """Test that absence of files raises an error."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = ImageCache()
            cache.dataset_cache_path = temp_dir

            with pytest.raises(RuntimeError, match="No image files found"):
                cache._load_and_verify_images(expected_count=0)

    def test_load_and_verify_images_incomplete_dataset_raises_error(self):
        """Test that incomplete dataset raises an error."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create only 2 files out of expected 5
            (Path(temp_dir) / "test1.dng").touch()
            (Path(temp_dir) / "test2.dng").touch()

            cache = ImageCache()
            cache.dataset_cache_path = temp_dir

            with pytest.raises(RuntimeError, match="Dataset incomplete"):
                cache._load_and_verify_images(expected_count=5)

    def test_load_and_verify_images_extra_files_warning(self):
        """Test warning when there are extra files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create more files than expected
            for i in range(5):
                (Path(temp_dir) / f"test{i}.dng").touch()

            cache = ImageCache()
            cache.dataset_cache_path = temp_dir

            # Should not raise error, only warning
            cache._load_and_verify_images(expected_count=3)
            assert len(cache.image_files) == 5


class TestImageCacheDownload:
    """Tests for dataset download."""

    @patch("imagerecovery.services.cache.cache.snapshot_download")
    @patch.object(ImageCache, "_get_expected_file_count")
    @patch.object(ImageCache, "_load_and_verify_images")
    def test_download_dataset_success(self, mock_load_verify, mock_get_count, mock_snapshot):
        """Test successful dataset download."""
        mock_get_count.return_value = 5
        mock_snapshot.return_value = "/fake/cache/path"

        cache = ImageCache()
        # Set image_files to match expected_count so the condition passes
        cache.image_files = ["file1.dng", "file2.dng", "file3.dng", "file4.dng", "file5.dng"]
        result = cache.download_dataset()

        assert result is True
        assert cache._ready is True
        assert cache.dataset_cache_path == "/fake/cache/path"

        mock_get_count.assert_called_once_with(cache.dataset_path)
        mock_snapshot.assert_called_with(
            repo_id=cache.dataset_path,
            repo_type="dataset",
            cache_dir=cache.cache_dir,
            allow_patterns=["*.dng"],
            resume_download=True,
            max_workers=2,
            etag_timeout=30,
            force_download=False,
        )
        mock_load_verify.assert_called_once_with(5)

    @patch("imagerecovery.services.cache.cache.snapshot_download")
    def test_download_dataset_failure(self, mock_snapshot):
        """Test failed dataset download."""
        mock_snapshot.side_effect = Exception("Download failed")

        cache = ImageCache()
        result = cache.download_dataset()

        assert result is False
        assert cache._ready is False


class TestImageCacheExistingCache:
    """Tests for existing cache detection."""

    @patch("huggingface_hub.try_to_load_from_cache")
    def test_check_existing_cache_found(self, mock_try_load):
        """Test finding existing cache."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test files
            test_file = Path(temp_dir) / ".gitattributes"
            test_file.touch()
            (Path(temp_dir) / "test1.dng").touch()
            (Path(temp_dir) / "test2.dng").touch()

            mock_try_load.return_value = str(test_file)

            cache = ImageCache()
            cache._check_existing_cache()

            assert cache._ready is True
            assert cache.dataset_cache_path == temp_dir
            assert len(cache.image_files) == 2

    @patch("huggingface_hub.try_to_load_from_cache")
    def test_check_existing_cache_not_found(self, mock_try_load):
        """Test when existing cache is not found."""
        mock_try_load.return_value = "_CACHED_FILE_NOT_FOUND"

        cache = ImageCache()
        cache._check_existing_cache()

        assert cache._ready is False
        assert cache.dataset_cache_path is None

    @patch("huggingface_hub.try_to_load_from_cache")
    def test_check_existing_cache_exception(self, mock_try_load):
        """Test exception handling during cache check."""
        mock_try_load.side_effect = Exception("Cache check failed")

        cache = ImageCache()
        cache._check_existing_cache()  # Should not raise exception

        assert cache._ready is False


class TestImageCacheImageLoading:
    """Tests for image loading from cache."""

    @patch.object(ImageCache, "_load_dng_as_pil")
    def test_get_random_image_pil_success(self, mock_load_dng):
        """Test successful random image retrieval."""
        mock_image = Mock(spec=Image.Image)
        mock_load_dng.return_value = mock_image

        cache = ImageCache()
        cache._ready = True
        cache.image_files = ["test1.dng", "test2.dng"]

        result = cache.get_random_image_pil()

        assert result is mock_image
        mock_load_dng.assert_called_once()

    def test_get_random_image_pil_cache_not_ready(self):
        """Test image retrieval when cache is not ready."""
        cache = ImageCache()
        cache._ready = False

        result = cache.get_random_image_pil()

        assert result is None

    def test_get_random_image_pil_no_files(self):
        """Test image retrieval when no files are available."""
        cache = ImageCache()
        cache._ready = True
        cache.image_files = []

        result = cache.get_random_image_pil()

        assert result is None

    @patch.object(ImageCache, "_load_dng_as_pil")
    def test_get_random_image_pil_retry_on_failure(self, mock_load_dng):
        """Test retry mechanism on loading failure."""
        # First two attempts fail, third succeeds
        mock_load_dng.side_effect = [Exception("Load failed"), Exception("Load failed"), Mock(spec=Image.Image)]

        cache = ImageCache()
        cache._ready = True
        cache.image_files = ["test.dng"]

        result = cache.get_random_image_pil(max_retries=3)

        assert result is not None
        assert mock_load_dng.call_count == 3

    @patch.object(ImageCache, "_load_dng_as_pil")
    def test_get_random_image_pil_max_retries_exceeded(self, mock_load_dng):
        """Test when maximum retry attempts are exceeded."""
        mock_load_dng.side_effect = Exception("Load failed")

        cache = ImageCache()
        cache._ready = True
        cache.image_files = ["test.dng"]

        result = cache.get_random_image_pil(max_retries=2)

        assert result is None
        assert mock_load_dng.call_count == 2


class TestImageCacheDNGLoading:
    """Tests for DNG file loading."""

    @patch("imagerecovery.services.cache.cache.rawpy.imread")
    def test_load_dng_as_pil_success(self, mock_rawpy_imread):
        """Test successful DNG file loading."""
        # Mock rawpy
        mock_raw = MagicMock()
        mock_raw.__enter__.return_value = mock_raw
        mock_raw.__exit__.return_value = None
        mock_raw.postprocess.return_value = np.ones((100, 100, 3), dtype=np.uint8)
        mock_rawpy_imread.return_value = mock_raw

        cache = ImageCache()
        result = cache._load_dng_as_pil("test.dng")

        assert isinstance(result, Image.Image)
        assert result.size == (100, 100)
        mock_rawpy_imread.assert_called_once_with("test.dng")
        mock_raw.postprocess.assert_called_once_with(use_camera_wb=True, no_auto_bright=False, output_bps=8)

    @patch("imagerecovery.services.cache.cache.rawpy.imread")
    def test_load_dng_as_pil_read_failure(self, mock_rawpy_imread):
        """Test failed DNG file reading."""
        mock_rawpy_imread.side_effect = Exception("Failed to read DNG")

        cache = ImageCache()
        result = cache._load_dng_as_pil("test.dng")

        assert result is None
        mock_rawpy_imread.assert_called_once_with("test.dng")

    @patch("imagerecovery.services.cache.cache.rawpy.imread")
    def test_load_dng_as_pil_exception(self, mock_rawpy_imread):
        """Test exception handling during DNG loading."""
        mock_rawpy_imread.side_effect = Exception("rawpy error")

        cache = ImageCache()
        result = cache._load_dng_as_pil("test.dng")

        assert result is None

    def test_load_dng_as_pil_lru_cache(self):
        """Test that _load_dng_as_pil uses LRU caching."""
        cache = ImageCache()

        # Check that method has LRU cache attributes
        assert hasattr(cache._load_dng_as_pil, "cache_info")
        assert hasattr(cache._load_dng_as_pil, "cache_clear")


class TestImageCacheStats:
    """Tests for cache statistics."""

    def test_get_cache_stats_empty_cache(self):
        """Test statistics for empty cache."""
        cache = ImageCache("test/dataset", "/test/cache")
        stats = cache.get_cache_stats()

        expected = {
            "dataset_path": "test/dataset",
            "total_images": 0,
            "cache_path": None,
            "is_ready": False,
        }

        assert stats == expected

    def test_get_cache_stats_ready_cache(self):
        """Test statistics for ready cache."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = ImageCache("test/dataset", "/test/cache")
            cache.dataset_cache_path = temp_dir
            cache.image_files = ["test1.dng", "test2.dng", "test3.dng"]
            cache._ready = True

            stats = cache.get_cache_stats()

            expected = {
                "dataset_path": "test/dataset",
                "total_images": 3,
                "cache_path": temp_dir,
                "is_ready": True,
            }

            assert stats == expected


class TestImageCacheIntegration:
    """Integration tests for ImageCache."""

    @pytest.fixture
    def mock_complete_setup(self):
        """Fixture with complete mock setup for integration tests."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test files
            dng_files = ["test1.dng", "test2.dng", "test3.dng"]
            for filename in dng_files:
                (Path(temp_dir) / filename).touch()

            with patch("imagerecovery.services.cache.cache.snapshot_download") as mock_download, patch(
                "imagerecovery.services.cache.cache.HfApi"
            ) as mock_api, patch("imagerecovery.services.cache.cache.rawpy.imread") as mock_rawpy:
                # Setup mocks
                mock_download.return_value = temp_dir

                mock_hf_api = Mock()
                mock_hf_api.list_repo_files.return_value = dng_files
                mock_api.return_value = mock_hf_api

                # Mock rawpy
                mock_raw = MagicMock()
                mock_raw.__enter__.return_value = mock_raw
                mock_raw.__exit__.return_value = None
                mock_raw.postprocess.return_value = np.ones((100, 100, 3), dtype=np.uint8)
                mock_rawpy.return_value = mock_raw

                yield {
                    "temp_dir": temp_dir,
                    "dng_files": dng_files,
                    "mock_download": mock_download,
                    "mock_api": mock_hf_api,
                    "mock_rawpy": mock_rawpy,
                }

    def test_full_cache_workflow(self, mock_complete_setup):
        """Test complete cache workflow."""
        # Reset singleton
        ImageCache._instance = None

        # Create cache
        cache = ImageCache.get_instance()
        assert not cache.is_ready()

        # Download dataset
        success = cache.download_dataset()
        assert success is True
        assert cache.is_ready()

        # Check statistics
        stats = cache.get_cache_stats()
        assert stats["total_images"] == 3
        assert stats["is_ready"] is True

        # Get random image
        image = cache.get_random_image_pil()
        assert image is not None
        assert isinstance(image, Image.Image)

        # Check that singleton works
        cache2 = ImageCache.get_instance()
        assert cache2 is cache
