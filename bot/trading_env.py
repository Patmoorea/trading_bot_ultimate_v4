import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import logging

class TradingEnv(gym.Env):
    def __init__(self, trading_pairs, timeframes):
        super().__init__()
        self.trading_pairs = trading_pairs
        self.timeframes = timeframes
        self.logger = logging.getLogger(__name__)

        # PATCH: Définition dynamique de la dimension d'input attendue par PPO
        N_FEATURES = 8    # Nombre de features par pas
        N_STEPS = 63      # Nombre de steps (historique)
        self.N_FEATURES = N_FEATURES
        self.N_STEPS = N_STEPS

        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf,
            shape=(N_FEATURES * N_STEPS * len(trading_pairs),),
            dtype=np.float32
        )
        print(f"[DEBUG TradingEnv] observation_space.shape = {self.observation_space.shape}")

        self.action_space = spaces.Box(
            low=0,
            high=1,
            shape=(len(trading_pairs),),
            dtype=np.float32
        )

        # Paramètres d'apprentissage
        self.max_steps = 1000
        self.current_step = 0
        self.portfolio_value = 10000  # Valeur initiale
        self.positions = {}
        self.reward_scale = 1.0
        self.position_history = []
        self.done_penalty = -1.0

        # Métriques
        self.metrics = {
            "episode_rewards": [],
            "portfolio_values": [],
            "positions": [],
            "actions": []
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.portfolio_value = 10000
        self.positions = {}
        self.position_history = []
        return self._get_observation(), {}

    def step(self, action):
        try:
            self.current_step += 1

            # Validation de l'action
            if not self.action_space.contains(action):
                self.logger.warning(f"Action invalide: {action}")
                action = np.clip(action, self.action_space.low, self.action_space.high)

            # Calcul de la récompense
            reward = self._calculate_reward(action)

            # Mise à jour de l'état
            self._update_state()

            # Vérification des conditions de fin
            done = self._check_done()
            truncated = False

            # Mise à jour des métriques
            self._update_metrics(action, reward)

            return self._get_observation(), reward, done, truncated, self._get_info()

        except Exception as e:
            self.logger.error(f"Erreur dans step: {e}")
            return self._get_observation(), 0, True, False, {}

    def _calculate_reward(self, action):
        """Calcule la récompense basée sur le PnL et le risque"""
        try:
            # Calcul du PnL simulé
            pnl = sum([pos * act for pos, act in zip(self.positions.values(), action)])

            # Pénalité pour le risque (exemple simple)
            risk_penalty = np.std(action) * 0.1

            # Reward final
            reward = (pnl - risk_penalty) * self.reward_scale

            return float(reward)

        except Exception as e:
            self.logger.error(f"Erreur calcul reward: {e}")
            return 0.0

    def _update_state(self):
        """Mise à jour de l'état"""
        try:
            # Simulation d'un état pour le moment
            self.state = np.random.normal(0, 1, self.observation_space.shape)
        except Exception as e:
            self.logger.error(f"Erreur mise à jour state: {e}")

    def _check_done(self):
        """Vérifie les conditions de fin d'épisode"""
        # Fin si max steps atteint
        if self.current_step >= self.max_steps:
            return True

        # Fin si portfolio vide
        if self.portfolio_value <= 0:
            return True

        return False

    def _update_metrics(self, action, reward):
        """Mise à jour des métriques"""
        self.metrics["episode_rewards"].append(reward)
        self.metrics["portfolio_values"].append(self.portfolio_value)
        self.metrics["positions"].append(self.positions.copy())
        self.metrics["actions"].append(action.copy())

    def _get_observation(self):
        """Retourne l'état actuel"""
        # PATCH: retourne un vecteur de la bonne dimension
        return np.zeros(self.observation_space.shape)

    def _get_info(self):
        """Retourne les informations additionnelles"""
        return {
            "portfolio_value": self.portfolio_value,
            "current_step": self.current_step,
            "positions": self.positions,
            "metrics": self.metrics
        }

    def render(self):
        """Affichage des métriques"""
        print(f"\nPortfolio Value: {self.portfolio_value:.2f}")
        print(f"Step: {self.current_step}/{self.max_steps}")
        print(f"Active Positions: {len(self.positions)}")
        if self.metrics["episode_rewards"]:
            print(f"Last Reward: {self.metrics['episode_rewards'][-1]:.2f}")