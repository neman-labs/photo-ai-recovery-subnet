import base64
import logging
import os
from functools import lru_cache
from io import BytesIO
from typing import Optional

import bittensor as bt
from PIL import Image


def suppress_hf_logging():
    """Configure basic logging suppressors for HuggingFace libraries."""
    # Basic environment variables
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Configure loggers
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    logging.getLogger("datasets").setLevel(logging.ERROR)

    # Disable progress bars
    try:
        from huggingface_hub.utils import disable_progress_bars

        disable_progress_bars()
    except ImportError:
        pass

    bt.logging.info("HuggingFace logging suppressed")


def bytes_to_image(image_bytes: bytes) -> Image.Image:
    """
    Converts bytes to PIL Image object

    Args:
        image_bytes: Byte representation of an image

    Returns:
        Image.Image: PIL Image object
    """
    return Image.open(BytesIO(image_bytes))


def image_to_bytes(image: Image.Image, format: str = "PNG") -> bytes:
    """
    Converts PIL Image object to bytes

    Args:
        image: PIL Image object
        format: Image format (PNG, JPEG, etc.)

    Returns:
        bytes: Byte representation of the image
    """
    buffer = BytesIO()
    image.save(buffer, format=format)
    return buffer.getvalue()


@lru_cache(maxsize=32)
def cached_base64_decode(base64_string: str) -> bytes:
    """
    Cached base64 decoding to avoid repeated decoding of same strings.

    Args:
        base64_string: Base64 encoded string

    Returns:
        bytes: Decoded bytes
    """
    return base64.b64decode(base64_string)


def is_valid_base64(s: str) -> bool:
    """Check if a string is valid base64"""
    try:
        base64.b64decode(s)
        return True
    except Exception:
        return False
