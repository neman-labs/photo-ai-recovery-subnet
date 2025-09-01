import base64
from typing import Optional, Tuple, cast

import bittensor as bt
from PIL import Image

from imagerecovery.protocol import ImageSynapse
from imagerecovery.schemas import AugmentedImageMetadata, Metadata, OriginalImageMetadata
from imagerecovery.services.augmentation import AugmentationService
from imagerecovery.services.cache.cache import ImageCache
from imagerecovery.services.task_generation import TaskGenerationService
from imagerecovery.utils import image_to_bytes
from imagerecovery.validator.task.base import ValidatorTask


class RestorationComparisonTask(ValidatorTask):
    """
    Task for comparing restored image with the original.

    Creates a task for miners where the input data is a transformed
    image that needs to be restored to the original quality.
    """

    __slots__ = ["__metadata", "augmentation_params", "augmentation_type"]

    TASK_NAME: str = "RestorationComparison"
    REWARD_WEIGHT: float = 1.0
    FORWARD_PROBABILITY: float = 1.0

    def __init__(self, augmentation_type: str = "downscale", **augmentation_params):
        """
        Initialize the task

        Args:
            augmentation_type: Type of augmentation
            **augmentation_params: Augmentation parameters
        """
        self.augmentation_type = augmentation_type
        self.augmentation_params = augmentation_params
        self.__metadata = None

    async def prepare_synapse(self) -> Optional[Tuple[ImageSynapse, bytes]]:
        """
        Prepares a Synapse for the task using a random image from cache

        Returns:
            Tuple of (ImageSynapse object with the task data, original image bytes) or None in case of error
        """
        try:
            # Get cache instance and image
            cache = ImageCache.get_instance()
            original_image_pil = cache.get_random_image_pil()

            if original_image_pil is None:
                bt.logging.error("Failed to get image from cache")
                return None

            img_format = "PNG"

            # Apply random transformations
            bt.logging.info("Applying random transformations to source image")
            augmentation_service = AugmentationService(augmentation_prob=0.5)
            transformed_image_pil, transform_params = augmentation_service.apply_transforms(
                original_image_pil,  # Pass PIL Image directly instead of path
                output_bytes=False,
                dataset_cache=cache,
            )

            if self.augmentation_type == "downscale":
                bt.logging.info(
                    f"Applying augmentation '{self.augmentation_type}' with parameters: {self.augmentation_params}"
                )
                task_service = TaskGenerationService()
                augmented_image_bytes = task_service.downscale(
                    transformed_image_pil,
                    scale_factor=self.augmentation_params.get("scale_factor", 6),
                    target_size=self.augmentation_params.get("target_size"),
                    output_bytes=True,
                    format=img_format,
                )
            else:
                bt.logging.warning(f"Unknown augmentation type '{self.augmentation_type}'. Using original image.")
                pil_image = cast(Image.Image, transformed_image_pil)
                augmented_image_bytes = image_to_bytes(pil_image, format=img_format)

            # Get original image bytes for comparison
            pil_image = cast(Image.Image, transformed_image_pil)
            original_image_bytes = image_to_bytes(pil_image, format=img_format)

            # Convert binary image to base64 before adding to synapse
            augmented_image_base64 = base64.b64encode(cast("bytes", augmented_image_bytes)).decode("utf-8")

            metadata = {"width": int(pil_image.width), "height": int(pil_image.height), "format": img_format}

            synapse = ImageSynapse(
                image_to_restore=augmented_image_base64,
                task_name=self.TASK_NAME,
                restored_image="",  # To be filled by miner
                target_metadata=metadata,
            )

            self.__metadata = Metadata(
                original_image_metadata=OriginalImageMetadata(
                    width=int(pil_image.width),
                    height=int(pil_image.height),
                    format=img_format,
                ),
                augmented_image_metadata=AugmentedImageMetadata(
                    augmentation_type=self.augmentation_type,
                    augmentation_params={**self.augmentation_params, "random_transforms": transform_params},
                ),
            )

            bt.logging.info(f"Successfully created synapse for task {self.TASK_NAME}")
            return synapse, original_image_bytes
        except Exception as e:
            bt.logging.error(f"Error preparing task {self.TASK_NAME}: {e}")
            return None

    def metadata(self) -> dict:
        """Returns task metadata"""
        base_metadata = super().metadata()
        if self.__metadata:
            return {**base_metadata, **self.__metadata.dict()}
        return base_metadata

    async def save_dataset(self, *args, **kwargs) -> None:
        """Save task dataset to disk or database."""
        bt.logging.info("Dataset saving not implemented for this task type")
