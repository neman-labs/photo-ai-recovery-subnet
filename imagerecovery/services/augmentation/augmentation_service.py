import random
from typing import Any, Dict, Optional, Tuple, Union

import bittensor as bt
import numpy as np
from PIL import Image

from imagerecovery.utils import bytes_to_image, image_to_bytes

# Type alias for cache
CacheType = Any


class AugmentationService:
    """
    Service for applying random image transformations to create unique variations.
    """

    def __init__(self, augmentation_prob: float = 0.7):
        """
        Initialize the augmentation service.

        Args:
            augmentation_prob: Probability (0.0-1.0) of applying each optional transformation
                              (horizontal flip, vertical flip). Geometric transformations like
                              rotation and overlay are applied with their own logic.
        """
        if not 0.0 <= augmentation_prob <= 1.0:
            raise ValueError("augmentation_prob must be between 0.0 and 1.0")

        self.augmentation_prob = augmentation_prob

    def apply_transforms(
        self,
        image: Union[Image.Image, bytes, str],
        output_bytes: bool = False,
        format: str = "PNG",
        dataset_cache: Optional[CacheType] = None,
    ) -> Union[Tuple[Image.Image, Dict[str, Any]], Tuple[bytes, Dict[str, Any]]]:
        """
        Applies random transformations to an image.

        Args:
            image: Input image (PIL Image, bytes, or path to image file)
            output_bytes: Whether to return bytes (True) or PIL Image (False)
            format: Image format when returning bytes
            dataset_cache: Dataset cache for accessing additional images for overlay

        Returns:
            If output_bytes is False:
                Tuple (transformed PIL Image, transformation parameters)
            If output_bytes is True:
                Tuple (transformed image bytes, transformation parameters)
        """
        is_bytes = isinstance(image, bytes)
        is_path = isinstance(image, str)

        # Convert input to PIL Image
        if is_path:
            try:
                pil_image = Image.open(image)
            except Exception as e:
                bt.logging.error(f"Failed to open image file: {e}")
                raise ValueError(f"Could not open image file: {image}")
        elif is_bytes:
            pil_image = bytes_to_image(image)
        else:
            pil_image = image

        if not isinstance(pil_image, Image.Image):
            bt.logging.error(f"Expected PIL Image but got {type(pil_image)}")
            raise TypeError(f"Expected PIL Image but got {type(pil_image)}")

        img = pil_image.copy()
        transform_params = {}

        # Apply flip transformations (probabilistic)
        img, flip_params = self._apply_flips(img)
        transform_params.update(flip_params)

        # Apply rotation and crop (always applied)
        img, rotation_params = self._apply_rotate_crop_transform(img)
        transform_params.update(rotation_params)

        # Apply overlay (if cache is available)
        if dataset_cache is not None:
            img, overlay_params = self._apply_overlay_transform(img, dataset_cache)
            transform_params.update(overlay_params)

        # Return in requested format
        if output_bytes or is_bytes:
            return image_to_bytes(img, format=format), transform_params

        return img, transform_params

    def _apply_flips(self, image: Image.Image) -> Tuple[Image.Image, Dict[str, Any]]:
        """
        Apply horizontal and vertical flips based on probability.

        Args:
            image: PIL Image to flip

        Returns:
            Tuple containing (flipped image, flip parameters)
        """
        img = image.copy()
        transform_params = {}

        # Apply horizontal flip
        if random.random() < self.augmentation_prob:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            transform_params["horizontal_flip"] = True
        else:
            transform_params["horizontal_flip"] = False

        # Apply vertical flip
        if random.random() < self.augmentation_prob:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            transform_params["vertical_flip"] = True
        else:
            transform_params["vertical_flip"] = False

        return img, transform_params

    def _apply_rotate_crop_transform(
        self, image: Image.Image, max_angle: float = 10.0
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        """
        Applies random rotation to the image and crops to remove black borders.

        Uses the largest possible rectangle that fits within the rotated image
        to avoid black edges/corners.

        Args:
            image: PIL Image to rotate and crop
            max_angle: Maximum rotation angle in degrees

        Returns:
            Tuple containing (rotated and cropped image, rotation parameters)
        """
        try:
            angle = random.uniform(-max_angle, max_angle)
            cropped_image, transform_params = self._rotate_and_crop_without_borders(image, angle)
            transform_params["rotation_applied"] = True
            return cropped_image, transform_params

        except Exception as e:
            bt.logging.error(f"Error applying rotation and crop transform: {e!s}")
            return image, {"rotation_applied": False, "rotation_angle": 0}

    def _apply_overlay_transform(
        self,
        base_image: Image.Image,
        dataset_cache: Optional[CacheType] = None,
        min_overlay_size: float = 0.1,  # 10% of the base image size
        max_overlay_size: float = 0.5,
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        """
        Applies an overlay of another image from the dataset cache onto the base image.
        The resulting image will have the same resolution as the base image.

        Args:
            base_image: Base PIL Image to overlay onto
            dataset_cache: Cache containing images to use for overlay
            min_overlay_size: Minimum overlay size as fraction of base image
            max_overlay_size: Maximum overlay size as fraction of base image

        Returns:
            Tuple containing (overlaid image, overlay parameters)
        """
        transform_params = {"overlay_applied": False, "overlay_angle": 0, "overlay_scale": 0, "overlay_position": (0, 0)}

        if dataset_cache is None:
            return base_image, transform_params

        try:
            overlay_image = dataset_cache.get_random_image_pil()
            if overlay_image is None:
                bt.logging.warning("No images available in cache for overlay transform")
                return base_image, transform_params

            overlay_image = overlay_image.convert("RGB")

            # Rotate overlay image
            angle = random.randint(0, 360)
            rotated_overlay, overlay_transform_params = self._rotate_and_crop_without_borders(overlay_image, angle)

            # Scale overlay
            base_width, base_height = base_image.size
            scale_factor = random.uniform(min_overlay_size, max_overlay_size)

            overlay_width = int(base_width * scale_factor)
            overlay_height = int(overlay_width * rotated_overlay.height / rotated_overlay.width)

            resized_overlay = rotated_overlay.resize((overlay_width, overlay_height), Image.LANCZOS)

            # Position overlay randomly
            max_x = base_width - overlay_width
            max_y = base_height - overlay_height

            pos_x = random.randint(0, max(0, max_x))
            pos_y = random.randint(0, max(0, max_y))

            # Create result and paste overlay
            result_image = base_image.copy()
            result_image.paste(resized_overlay, (pos_x, pos_y))

            transform_params = {
                "overlay_applied": True,
                "overlay_angle": angle,
                "overlay_scale": scale_factor,
                "overlay_position": (pos_x, pos_y),
                "overlay_crop_dimensions": overlay_transform_params["crop_dimensions"],
            }

            return result_image, transform_params

        except Exception as e:
            bt.logging.error(f"Error applying overlay transform: {e!s}")
            return base_image, transform_params

    def _rotate_and_crop_without_borders(self, image: Image.Image, angle: float) -> Tuple[Image.Image, Dict[str, Any]]:
        """
        Helper method that rotates an image and crops it to remove black borders.

        Args:
            image: PIL Image to rotate and crop
            angle: Rotation angle in degrees

        Returns:
            Tuple containing:
                - The rotated and cropped image
                - Dictionary with transformation parameters
        """
        width, height = image.size
        rotated_image = image.rotate(angle, expand=True, resample=Image.BICUBIC)

        rads = abs(angle) * np.pi / 180
        cos_a = abs(np.cos(rads))
        sin_a = abs(np.sin(rads))

        # Calculate the size of the largest rectangle inside the rotated image
        # Formula for maximum inscribed rectangle
        if width <= height:
            crop_width = abs(int(width * cos_a - height * sin_a))
            crop_height = abs(int(height * cos_a - width * sin_a))
        else:
            crop_width = abs(int(width * cos_a - height * sin_a))
            crop_height = abs(int(height * cos_a - width * sin_a))

        crop_width = max(1, crop_width)
        crop_height = max(1, crop_height)
        rot_width, rot_height = rotated_image.size

        # Center the crop area
        left = (rot_width - crop_width) // 2
        top = (rot_height - crop_height) // 2
        right = left + crop_width
        bottom = top + crop_height

        cropped_image = rotated_image.crop((left, top, right, bottom))

        transform_params = {"rotation_angle": angle, "crop_dimensions": (crop_width, crop_height)}

        return cropped_image, transform_params
