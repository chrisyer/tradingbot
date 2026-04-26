"""Multi-asset wrapper environment for training a shared agent across symbols."""

import random
from dataclasses import dataclass

import numpy as np
import gymnasium as gym

from env.xauusd_env import XAUUSDTradingEnv


@dataclass
class AssetDataset:
    symbol: str
    features: np.ndarray
    returns: np.ndarray


class MultiAssetTradingEnv(gym.Env):
    """
    Wraps multiple asset datasets and samples one per episode.

    Observation/action spaces are inherited from the first child env.
    Reward logic is delegated to `XAUUSDTradingEnv` to stay consistent.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        datasets: list[AssetDataset],
        window: int = 64,
        cost_per_trade: float = 0.0001,
        max_episode_steps: int | None = None,
        seed: int = 42,
    ):
        if not datasets:
            raise ValueError("datasets must not be empty")

        self.datasets = datasets
        self.window = window
        self.cost_per_trade = cost_per_trade
        self.max_episode_steps = max_episode_steps
        self.rng = random.Random(seed)

        self.current_symbol = datasets[0].symbol
        self.current_env = self._make_child_env(datasets[0])

        self.observation_space = self.current_env.observation_space
        self.action_space = self.current_env.action_space

    def _make_child_env(self, dataset: AssetDataset) -> XAUUSDTradingEnv:
        return XAUUSDTradingEnv(
            features=dataset.features,
            returns=dataset.returns,
            window=self.window,
            cost_per_trade=self.cost_per_trade,
            max_episode_steps=self.max_episode_steps,
        )

    def _sample_dataset(self) -> AssetDataset:
        return self.rng.choice(self.datasets)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        sampled = self._sample_dataset()
        self.current_symbol = sampled.symbol
        self.current_env = self._make_child_env(sampled)
        obs, info = self.current_env.reset(seed=seed, options=options)
        info["symbol"] = self.current_symbol
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.current_env.step(action)
        info["symbol"] = self.current_symbol
        return obs, reward, terminated, truncated, info
