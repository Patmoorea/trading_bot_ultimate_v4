# src/environments/trading_env.py
import logging
import gymnasium as gym  # Changez import gym en import gymnasium as gym
import numpy as np


class TradingEnv(gym.Env):
    def __init__(self, trading_pairs, timeframes):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.logger.info(
            f"Initialisation TradingEnv avec {trading_pairs} et {timeframes}"
        )

        self.trading_pairs = trading_pairs
        self.timeframes = timeframes

        # Définition des espaces
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(42,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(3)  # HOLD, BUY, SELL

    def reset(self, seed=None):  # Fixed indentation
        super().reset(seed=seed)
        observation = np.zeros(42)
        info = {}
        return observation, info

    def step(self, action):  # Fixed indentation
        observation = np.zeros(42)
        reward = 0
        terminated = False
        truncated = False
        info = {}
        return observation, reward, terminated, truncated, info
