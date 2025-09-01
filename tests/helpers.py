import os
from datetime import datetime, timedelta
from random import randint, random

import bittensor as bt
import numpy as np
from substrateinterface import Keypair

from imagerecovery.base.utils.config import add_args, add_validator_args, argparse
from imagerecovery.protocol import ImageSynapse

kp = Keypair.create_from_mnemonic(Keypair.generate_mnemonic())


class MetagraphStub:
    def __init__(self, neurons_count):
        self.netuid = 0
        self.network, self.chain_endpoint = "mock", "mock"
        self.subtensor = None

        self.version = np.array([1], dtype=np.int64)
        self.n = np.array([neurons_count], dtype=np.int64)
        self.block = np.array([0], dtype=np.int64)
        self.ranks = np.array([0] * neurons_count, dtype=np.float32)
        self.trust = np.array([0] * neurons_count, dtype=np.float32)
        self.consensus = np.array([0] * neurons_count, dtype=np.float32)
        self.validator_trust = np.array([0] * neurons_count, dtype=np.float32)
        self.incentive = np.array([0] * neurons_count, dtype=np.float32)
        self.emission = np.array([0] * neurons_count, dtype=np.float32)
        self.dividends = np.array([0] * neurons_count, dtype=np.float32)
        self.active = np.array([0] * neurons_count, dtype=np.int64)
        self.last_update = np.array([0] * neurons_count, dtype=np.int64)
        self.validator_permit = np.array([0] * neurons_count, dtype=bool)
        self.weights = np.array([0] * neurons_count, dtype=np.float32)
        self.bonds = np.array([0] * neurons_count, dtype=np.int64)
        self.uids = np.array(list(range(neurons_count)), dtype=np.int64)
        self.alpha_stake = np.array([0] * neurons_count, dtype=np.int64)
        self.tao_stake = np.array([0] * neurons_count, dtype=np.int64)
        self.stake = np.array([0] * neurons_count, dtype=np.int64)
        self.S = np.array([0] * neurons_count, dtype=np.int64)
        self.axons = [AxonStub() for _ in range(neurons_count)]
        self.hotkeys = [f"hotkey_{i}" for i in range(neurons_count)]


class AxonStub:
    def __init__(self):
        self.is_serving = True


class ValidatorStub:
    """
    Stub the validator for testing your forward() method
    """

    def __init__(self, metagraph):
        self.metagraph = metagraph
        self.REDIS_SCHEDULED_FLIGHTS_SET_KEY = "mock"
        self._init_config()

    def _init_config(self):
        parser = argparse.ArgumentParser()
        add_args(None, parser)
        add_validator_args(None, parser)
        self.config = bt.config(parser)

    async def dendrite(
        self,
        axons,
        synapse: ImageSynapse,
        timeout=60,
        deserialize=True,
    ):
        ...

    async def forward(self):
        ...

    def set_weights(self):
        ...
