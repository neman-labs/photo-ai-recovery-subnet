# The MIT License (MIT)
# Copyright © 2023 Yuma Rao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


from typing import Any, Dict

import bittensor as bt
import pydantic


class ImageSynapse(bt.Synapse):
    """
    A protocol representation for image restoration tasks.
    This protocol helps in handling request and response communication between
    the miner and the validator for image restoration tasks.

    Attributes:
        image_to_restore (str): The degraded/low-quality image as a base64-encoded string
                                that needs to be restored by the miner. This is an immutable field.
                                This field is not included in the response from miner.

        task_name (str): Identifier for the type of restoration task. This is an immutable field.

        target_metadata (Dict[str, Any]): Metadata about the original high-quality image,
                                         including its width, height, and format. This information
                                         helps miners generate better restorations by knowing
                                         the target dimensions. This is an immutable field.
                                         Keys include 'width', 'height', and 'format'.

        restored_image (str): The restored high-quality image provided by the miner as a
                              base64-encoded string response. Initially empty, filled by the miner.
    """

    # Input fields (from validator to miner)
    image_to_restore: str = pydantic.Field(
        ...,
        title="Image to Restore (Base64)",
        description="The base64-encoded image that needs to be restored. Immutable.",
        frozen=True,
    )

    task_name: str = pydantic.Field(
        "RestorationComparison",
        title="Task Name",
        description="The name of the task. Immutable.",
        frozen=True,
    )

    target_metadata: Dict[str, Any] = pydantic.Field(
        ...,
        title="Target Metadata",
        description="Dictionary with metadata about original image including width, height and format keys.",
        frozen=True,
    )

    # Output fields (from miner to validator)
    restored_image: str = pydantic.Field(
        "",
        title="Restored Image (Base64)",
        description="The base64-encoded restored image provided by the miner. Mutable.",
    )

    def deserialize(self) -> str:
        """
        Deserialize output. This method retrieves the response from
        the miner, deserializes it and returns the restored image as a base64 string.

        Returns:
            str: The deserialized response containing the base64-encoded restored image.
        """
        return self.restored_image
