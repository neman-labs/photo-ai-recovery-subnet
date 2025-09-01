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

import time
from contextlib import suppress

import bittensor as bt
import numpy as np
import wandb

from imagerecovery.base.utils import uids
from imagerecovery.base.validator import BaseValidatorNeuron
from imagerecovery.services.cache.cache import ImageCache
from imagerecovery.services.image_quality import ImageQualityService
from imagerecovery.validator.reward import RewardCalculator
from imagerecovery.validator.task import ValidatorTask, select_task


async def forward_validator_task(self: BaseValidatorNeuron):
    bt.logging.info("Preparing validator task...")

    # Get cache instance and ensure it's ready
    cache = ImageCache.get_instance()
    if not cache.is_ready():
        bt.logging.info("Dataset cache not ready, starting download...")
        success = cache.download_dataset()
        if not success:
            bt.logging.error("Failed to download dataset, skipping forward pass")
            return
        bt.logging.info("Dataset cache ready")
    else:
        bt.logging.debug("Dataset cache already ready")

    miner_uids = uids.get_available_miners(self, k=self.config.neuron.sample_size)
    bt.logging.info(f"Miners: {miner_uids.tolist()}")
    axons = [self.metagraph.axons[uid] for uid in miner_uids]

    if len(miner_uids) == 0:
        bt.logging.info("No miners available")
        return

    task: ValidatorTask = select_task(self.tasks)
    bt.logging.info(f"Selected task: {task.TASK_NAME}")

    try:
        synapse, original_image_bytes = await task.prepare_synapse()
    except Exception as e:
        bt.logging.error(f"Failed to prepare synapse: {e}")
        return

    start = time.perf_counter()
    responses = await self.dendrite(
        axons=axons,
        synapse=synapse,
        deserialize=True,
        timeout=30,
    )

    # Log the results for monitoring purposes.
    bt.logging.info(f"Received responses in {time.perf_counter() - start:.2f} seconds")

    uids_list = miner_uids.tolist() if isinstance(miner_uids, np.ndarray) else list(miner_uids)

    # Calculate rewards using RewardCalculator
    reward_calculator = RewardCalculator(
        image_quality_service=ImageQualityService(),
    )
    rewards = reward_calculator.get_rewards(
        responses=responses,
        original_image_bytes=original_image_bytes,
        uids=uids_list,
        axons=axons,
        task=task,
    )

    bt.logging.info(f"Scored responses: {rewards.tolist()}")

    # Update historical data
    self.update_historical_data(task, uids_list, rewards)

    # Update the validator's scores with the rewards
    self.update_scores(rewards, uids_list)
    self.save_miner_history()

    if not self.config.wandb.off:
        wandb_logging_context = {
            "rewards": rewards.tolist(),
            "miner_uids": miner_uids.tolist(),
            "scores": self.scores.tolist(),
            "task_metadata": task.metadata(),
        }
        with suppress(Exception):
            wandb.log(wandb_logging_context)
