import random
import time
from functools import lru_cache
from pathlib import Path
from typing import Optional

import bittensor as bt
import cv2
import rawpy
from huggingface_hub import HfApi, snapshot_download
from PIL import Image

DEFAULT_DATASET_PATH = "NemanTeam/mit-adobe-5k"
DEFAULT_CACHE_DIR = str(Path.home() / ".cache" / "imagerecovery" / "datasets")


class ImageCache:
    """
    Dataset caching system for image processing tasks.
    """

    _instance = None

    @classmethod
    def get_instance(
        cls,
        dataset_path: str = DEFAULT_DATASET_PATH,
        cache_dir: Optional[str] = DEFAULT_CACHE_DIR,
    ):
        """Get or create singleton instance."""
        if cls._instance is None:
            cls._instance = cls(dataset_path, cache_dir)
        return cls._instance

    def __init__(
        self,
        dataset_path: str = DEFAULT_DATASET_PATH,
        cache_dir: Optional[str] = DEFAULT_CACHE_DIR,
    ):
        self.dataset_path = dataset_path
        self.cache_dir = cache_dir
        self.dataset_cache_path = None
        self.image_files = []
        self._ready = False

        bt.logging.info(f"ImageCache initialized for dataset: {dataset_path}")

    def _get_expected_file_count(self, dataset_path: str) -> int:
        """Get expected number of .dng files from HuggingFace API."""
        try:
            api = HfApi()
            files = api.list_repo_files(repo_id=dataset_path, repo_type="dataset")
            dng_files = [f for f in files if f.lower().endswith(".dng")]

            bt.logging.info(f"Dataset {dataset_path} contains {len(dng_files)} .dng files")
            return len(dng_files)

        except Exception as e:
            bt.logging.warning(f"Could not get expected file count from API: {e}")
            return 0  # Will skip integrity check if API fails

    def _load_and_verify_images(self, expected_count: int):
        """Load file list and verify download integrity."""
        cache_path = Path(self.dataset_cache_path)
        self.image_files = [str(f) for f in cache_path.rglob("*.dng")]

        downloaded_count = len(self.image_files)

        if downloaded_count == 0:
            raise RuntimeError(f"❌ No image files found in dataset {self.dataset_path}")

        bt.logging.info(f"Found {downloaded_count} image files in cache")

        # Verify integrity if we got expected count from API
        if expected_count > 0:
            if downloaded_count < expected_count:
                raise RuntimeError(
                    f"❌ Dataset incomplete: expected {expected_count} files, "
                    f"but only {downloaded_count} were downloaded"
                )
            if downloaded_count > expected_count:
                bt.logging.warning(f"⚠️ More files than expected: got {downloaded_count}, expected {expected_count}")
            else:
                bt.logging.info(f"✅ Dataset integrity verified: {downloaded_count}/{expected_count} files")
        else:
            bt.logging.info(f"📁 Skipping integrity check (API unavailable)")

    def is_ready(self) -> bool:
        """
        Check if the cache is ready for use.

        Returns:
            bool: True if dataset is downloaded and ready, False otherwise
        """
        if self._ready:
            return True

        # Check if we have cached dataset and images
        if self.dataset_cache_path and self.image_files:
            # Verify cache path still exists
            cache_path = Path(self.dataset_cache_path)
            if cache_path.exists():
                self._ready = True
                return True

        # Try to find existing cache
        self._check_existing_cache()

        return self._ready

    def download_dataset(self) -> bool:
        """
        Download the entire dataset using snapshot_download with retry logic.

        Returns:
            bool: True if download successful, False otherwise
        """
        max_retries = 3

        for attempt in range(max_retries):
            try:
                bt.logging.info(f"Starting dataset download: {self.dataset_path} (attempt {attempt + 1}/{max_retries})")

                # Get expected file count from HuggingFace API before downloading
                expected_count = self._get_expected_file_count(self.dataset_path)

                bt.logging.info("Downloading dataset from HuggingFace Hub...")
                self.dataset_cache_path = snapshot_download(
                    repo_id=self.dataset_path,
                    repo_type="dataset",
                    cache_dir=self.cache_dir,
                    allow_patterns=["*.dng"],
                    resume_download=True,  # Resume interrupted downloads
                    max_workers=2,  # Reduced for stability
                    etag_timeout=30,  # Increased timeout
                    force_download=False,  # Don't re-download existing files
                )

                self._load_and_verify_images(expected_count)

                if len(self.image_files) == expected_count:
                    self._ready = True
                    bt.logging.info(f"✅ Dataset download completed with {len(self.image_files)} images")
                    return True
                else:
                    bt.logging.warning(f"⚠️ Incomplete download: {len(self.image_files)}/{expected_count} files")
                    if attempt < max_retries - 1:
                        wait_time = 10 * (attempt + 1)
                        bt.logging.info(f"Waiting {wait_time} seconds before retry...")
                        time.sleep(wait_time)

            except Exception as e:
                bt.logging.error(f"❌ Download attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    wait_time = 10 * (attempt + 1)
                    bt.logging.info(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)

        bt.logging.error(f"Failed to download dataset after {max_retries} attempts")
        self._ready = False
        return False

    def _check_existing_cache(self) -> None:
        """
        Check if dataset is already cached locally.
        """
        try:
            # Try to find existing cache without downloading
            from huggingface_hub import try_to_load_from_cache

            # This will return cached path if exists, otherwise None
            cache_path = try_to_load_from_cache(
                repo_id=self.dataset_path,
                filename=".gitattributes",  # Any file that should exist
                repo_type="dataset",
                cache_dir=self.cache_dir or DEFAULT_CACHE_DIR,
            )

            if cache_path and cache_path != "_CACHED_FILE_NOT_FOUND":
                # Cache exists, get the parent directory
                self.dataset_cache_path = str(Path(cache_path).parent)

                # Load existing images
                cache_path_obj = Path(self.dataset_cache_path)
                self.image_files = [str(f) for f in cache_path_obj.rglob("*.dng")]

                if self.image_files:
                    self._ready = True
                    bt.logging.info(f"Found existing cache with {len(self.image_files)} images")

        except Exception as e:
            bt.logging.debug(f"No existing cache found: {e}")

    def get_random_image_pil(self, max_retries: int = 3) -> Optional[Image.Image]:
        """
        Get a random image from cache as PIL Image.

        Args:
            max_retries: Maximum number of retry attempts if image loading fails

        Returns:
            Optional[Image.Image]: A PIL Image if successful, None otherwise
        """
        if not self.is_ready():
            bt.logging.warning("Cache is not ready. Call download_dataset() first.")
            return None

        if not self.image_files:
            bt.logging.warning("No image files available in cache")
            return None

        for attempt in range(max_retries):
            try:
                file_path = random.choice(self.image_files)
                return self._load_dng_as_pil(file_path)
            except Exception as e:
                bt.logging.warning(f"Failed to load image (attempt {attempt+1}): {e}")

        bt.logging.error(f"Failed to load any image after {max_retries} attempts")
        return None

    @lru_cache(maxsize=128)
    def _load_dng_as_pil(self, file_path: str) -> Optional[Image.Image]:
        """Load DNG file as PIL Image using rawpy."""
        try:
            with rawpy.imread(file_path) as raw:
                rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=False, output_bps=8)
                return Image.fromarray(rgb)
        except Exception as e:
            bt.logging.error(f"Error processing DNG file {file_path}: {e}")
            return None

    def get_cache_stats(self) -> dict:
        """Get basic statistics about the cache."""
        return {
            "dataset_path": self.dataset_path,
            "total_images": len(self.image_files),
            "cache_path": str(self.dataset_cache_path) if self.dataset_cache_path else None,
            "is_ready": self.is_ready(),
        }
