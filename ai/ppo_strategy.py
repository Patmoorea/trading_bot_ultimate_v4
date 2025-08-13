import torch
import logging
import numpy as np
from typing import Dict, Any, Optional, Union
from stable_baselines3 import PPO
from ..strategies.base import BaseStrategy


class PPOStrategy(BaseStrategy):
    def __init__(self, config: dict):
        super().__init__(config)
        self.logger = logging.getLogger(__name__)

        # Vérification de l'environnement
        self.env = config.get("env")
        if self.env is None:
            self.logger.error("Environnement manquant dans la configuration")
            raise ValueError("Environnement manquant dans la configuration")

        # Dimensions
        # PATCH: input_dim doit matcher le nombre de paires dynamiquement
        self.input_dim = config.get("input_dim")
        if self.input_dim is None:
            N_FEATURES = 8
            N_STEPS = 63
            num_pairs = len(getattr(self.env, "trading_pairs", []))
            self.input_dim = N_FEATURES * N_STEPS * num_pairs
        self.action_dim = len(self.env.trading_pairs)

        # Configuration PPO
        self.ppo_config = {
            "policy": "MlpPolicy",
            "learning_rate": config.get("learning_rate", 3e-4),
            "n_steps": config.get("n_steps", 2048),
            "batch_size": config.get("batch_size", 64),
            "n_epochs": config.get("n_epochs", 10),
            "gamma": config.get("gamma", 0.99),
            "gae_lambda": config.get("gae_lambda", 0.95),
            "clip_range": config.get("clip_range", 0.2),
            "verbose": config.get("verbose", 1),
            "policy_kwargs": {"net_arch": [64, 64], "activation_fn": torch.nn.ReLU},
        }
        print(f"[DEBUG PPO] input_dim attendu: {self.input_dim}, nombre de paires : {len(self.env.trading_pairs)}")
        try:
            self.model = PPO(env=self.env, **self.ppo_config)
            self.logger.info("✅ Modèle PPO initialisé")
        except Exception as e:
            self.logger.error(f"❌ Erreur initialisation PPO: {str(e)}")
            self.model = None

    def get_action(self, state: np.ndarray) -> Dict[str, Any]:
        """
        Méthode explicite get_action requise par le bot
        """
        try:
            if self.model is None:
                return {"action": "HOLD", "confidence": 0.0}

            # Préparation de l'état
            processed_state = self._preprocess_state(state)

            # Prédiction
            action, _ = self.model.predict(processed_state, deterministic=True)

            # Calcul de la confiance
            confidence = self._calculate_confidence(processed_state)

            return {
                "action": self._convert_action(action),
                "confidence": confidence,
                "raw_action": action,
            }

        except Exception as e:
            self.logger.error(f"Erreur get_action: {str(e)}")
            return {"action": "HOLD", "confidence": 0.0}

    def _preprocess_state(self, state: np.ndarray) -> np.ndarray:
        try:
            if isinstance(state, (list, tuple)):
                state = np.array(state, dtype=np.float32)
            state = state.flatten()
            if state.shape[0] != self.input_dim:
                # pad ou truncate si besoin
                if state.shape[0] > self.input_dim:
                    state = state[:self.input_dim]
                else:
                    pad_width = (0, self.input_dim - state.shape[0])
                    state = np.pad(state, pad_width, mode="constant")
            return state
        except Exception as e:
            self.logger.error(f"Erreur prétraitement: {str(e)}")
            return np.zeros(self.input_dim)

    def _calculate_confidence(self, state: np.ndarray) -> float:
        """Calcul du score de confiance compatible avec Categorical/Normal SB3"""
        try:
            if self.model is None:
                return 0.0

            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                dist = self.model.policy.get_distribution(state_tensor)
                # Pour Categorical (actions discrètes)
                if hasattr(dist.distribution, "probs"):
                    probs = dist.distribution.probs
                    return float(torch.max(probs))
                # Pour Normal (actions continues)
                elif hasattr(dist.distribution, "mean"):
                    std = dist.distribution.stddev
                    confidence = 1.0 / (1.0 + float(std.mean()))
                    return confidence
                else:
                    return 0.0

        except Exception as e:
            self.logger.error(f"Erreur calcul confiance: {str(e)}")
            return 0.0

    def _convert_action(self, action: Union[int, np.ndarray]) -> str:
        """Conversion action -> décision"""
        try:
            if isinstance(action, np.ndarray):
                action = action.flatten()[
                    0
                ]  # PATCH: .flatten()[0] pour éviter les erreurs array > 1
            actions = {0: "HOLD", 1: "BUY", 2: "SELL"}
            return actions.get(int(action), "HOLD")
        except Exception as e:
            self.logger.error(f"Erreur conversion action: {str(e)}")
            return "HOLD"

    async def analyze_market(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyse du marché"""
        try:
            state = self._prepare_observation(market_data)
            action_dict = self.get_action(state)

            return {
                "action": action_dict["action"],
                "confidence": action_dict["confidence"],
                "size": self.config.get("position_size", 1000),
                "metadata": {
                    "raw_action": action_dict["raw_action"],
                    "timestamp": market_data.get("timestamp"),
                },
            }

        except Exception as e:
            self.logger.error(f"Erreur analyze_market: {str(e)}")
            return {"action": "HOLD", "confidence": 0.0}

    def _prepare_observation(self, market_data: Dict[str, Any]) -> np.ndarray:
        try:
            features = []

            # Log des dimensions pour déboguer
            for key, value in market_data.items():
                self.logger.debug(
                    f"Dimension de {key}: {np.array(value).shape if isinstance(value, (list, np.ndarray)) else 'scalaire'}"
                )

            # OHLCV
            if "ohlcv" in market_data:
                ohlcv = np.array(market_data["ohlcv"], dtype=np.float32)
                if len(ohlcv) > 0:
                    features.append(ohlcv[-1])  # Dernier état

            # Indicateurs
            if "indicators" in market_data:
                indicators = np.array(
                    list(market_data["indicators"].values()), dtype=np.float32
                )
                features.append(indicators)

            # Métriques
            if "market_metrics" in market_data:
                metrics = np.array(
                    list(market_data["market_metrics"].values()), dtype=np.float32
                )
                features.append(metrics)

            if not features:
                return np.zeros(self.input_dim)

            # Combinaison des features
            state = np.concatenate(features).flatten()
            return self._preprocess_state(state)

        except Exception as e:
            self.logger.error(f"Erreur prepare_observation: {str(e)}")
            return np.zeros(self.input_dim)
