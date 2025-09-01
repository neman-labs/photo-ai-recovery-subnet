from typing import Optional, Tuple, Union, cast

from PIL import Image

from imagerecovery.utils import bytes_to_image, image_to_bytes


class TaskGenerationService:
    """
    Service for generating tasks by transforming images for validator testing.
    """

    def __init__(self):
        """Initialize the task generation service."""

    def downscale(
        self,
        image: Union[Image.Image, bytes],
        scale_factor: int = 4,
        target_size: Optional[Tuple[int, int]] = None,
        output_bytes: Optional[bool] = None,
        format: str = "PNG",
    ) -> Union[Image.Image, bytes]:
        """
        Reduces the resolution of an image to create a downscaling task.

        Args:
            image: Input image (PIL Image or bytes)
            scale_factor: Resolution reduction factor
            target_size: Target size (if None, scale_factor is used)
            output_bytes: Whether to return bytes (True) or PIL Image (False).
                         If None, matches input format (bytes input -> bytes output)
            format: Image format to use when returning bytes

        Returns:
            Downscaled image (PIL Image or bytes based on output_bytes parameter)
        """
        # Convert bytes to PIL Image if needed
        is_bytes = isinstance(image, bytes)
        pil_image = bytes_to_image(cast("bytes", image)) if is_bytes else cast("Image.Image", image)

        # Perform the downscaling
        if target_size is not None:
            new_size = target_size
        else:
            width, height = pil_image.size
            new_size = (width // scale_factor, height // scale_factor)

        # Downscale the image using bilinear interpolation
        result_image = pil_image.resize(new_size, Image.Resampling.BILINEAR)

        # Return the result in the requested format
        # If output_bytes is explicitly set, use that; otherwise match input format
        should_return_bytes = output_bytes if output_bytes is not None else is_bytes
        if should_return_bytes:
            return image_to_bytes(result_image, format=format)
        return result_image
