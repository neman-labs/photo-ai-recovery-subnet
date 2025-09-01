# The MIT License (MIT)
# Copyright © 2023

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

import base64
from typing import List

import bittensor as bt
import numpy as np

from imagerecovery.services.image_quality import ImageQualityService
from imagerecovery.utils import cached_base64_decode, is_valid_base64
from imagerecovery.validator.task import ValidatorTask


class RewardCalculator:
    """
    Calculates rewards based on responses from miners, performing quality assessment.
    """

    def __init__(
        self,
        image_quality_service: ImageQualityService,
    ):
        """
        Initialize RewardCalculator with dependencies.

        Args:
            image_quality_service: Service for calculating image quality metrics
        """
        self.image_quality_service = image_quality_service

    def get_rewards(
        self,
        responses: list,
        original_image_bytes: bytes,
        uids: List[int],
        axons: list,
        task: ValidatorTask,
    ) -> np.ndarray:
        """
        Calculate rewards for miners based on responses.

        Args:
            responses: List of responses from miners (base64-encoded strings)
            original_image_bytes: Original image bytes for quality comparison
            uids: List of miner UIDs
            axons: List of miner axons
            task: The validator task being evaluated

        Returns:
            numpy.ndarray: Array of rewards
        """
        rewards = []

        for response, uid, axon in zip(responses, uids, axons):
            hotkey = axon.hotkey

            # Decode base64 response
            decoded_response = cached_base64_decode(response) if response and is_valid_base64(response) else None

            # Calculate quality score from decoded response
            quality_score = self._calculate_quality_score(decoded_response, original_image_bytes, uid)

            # Apply task weight
            final_score = quality_score * task.REWARD_WEIGHT

            rewards.append(final_score)

            # Log the calculation details
            self._log_reward_details(uid, hotkey, final_score)

        return np.array(rewards)

    def _calculate_quality_score(self, response: bytes, original_image_bytes: bytes, miner_uid: int) -> float:
        """
        Calculate quality score for a single response.

        Args:
            response: Decoded response from miner (bytes for image restoration tasks)
            original_image_bytes: Original image bytes for comparison

        Returns:
            float: Quality score in range [0.0, 1.0]
        """
        if isinstance(response, bytes) and response:
            try:
                return self.image_quality_service.calculate_score(
                    reference_img_bytes=original_image_bytes, restored_img_bytes=response, miner_uid=miner_uid
                )
            except Exception as e:
                bt.logging.error(f"Error calculating quality score: {e}")
                return 0.0
        return 0.0

    def _log_reward_details(
        self,
        uid: int,
        hotkey: str,
        final_score: float,
    ) -> None:
        bt.logging.info(f"[REWARD][UID:{uid}][HOTKEY:{hotkey}] " f"reward={final_score:.4f}")
