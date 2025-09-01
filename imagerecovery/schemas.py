from typing import Any, Dict

from pydantic import BaseModel


class OriginalImageMetadata(BaseModel):
    """Metadata for the original image"""

    width: int
    height: int
    format: str


class AugmentedImageMetadata(BaseModel):
    """Metadata for the augmented image"""

    augmentation_type: str
    augmentation_params: Dict[str, Any]


class Metadata(BaseModel):
    """Metadata for the restoration comparison task"""

    original_image_metadata: OriginalImageMetadata
    augmented_image_metadata: AugmentedImageMetadata
