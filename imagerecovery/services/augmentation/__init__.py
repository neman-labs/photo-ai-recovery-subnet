"""
Image augmentation service for creating unique image variations.

This service applies random transformations to images to prevent mining cheating
by ensuring each task image is unique and cannot be easily found in the original dataset.
"""

from .augmentation_service import AugmentationService

__all__ = ["AugmentationService"]
