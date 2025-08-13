from stable_baselines3 import PPO
from transformers import GPT2Model
import tensorflow as tf
from typing import Dict, Any
import numpy as np

class HybridAI:
    def __init__(self):
        self.technical_model = self._build_technical_model()
        self.decision_model = self._build_decision_model()

    def _build_technical_model(self):
        """CNN-LSTM 18 couches"""
        inputs = tf.keras.Input(shape=(100, 5, 4))
        x = tf.keras.layers.Conv2D(64, (3,3), activation='relu')(inputs)
        x = tf.keras.layers.LSTM(128, return_sequences=True)(x)
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        return tf.keras.Model(inputs=inputs, outputs=outputs)

    def _build_decision_model(self):
        """PPO avec politique custom"""
        return PPO(
            policy="MlpPolicy",
            policy_kwargs={
                "net_arch": [dict(pi=[256, 256], vf=[256, 256])]
            },
            n_steps=2048,
            batch_size=64
        )

    def predict(self, analysis: Dict[str, Any]) -> Dict[str, float]:
        """Prédiction combinée"""
        tech_pred = self.technical_model.predict(analysis['technical'])
        decision_pred = self.decision_model.predict(analysis['features'])
        return {
            'direction': float(tech_pred[0]),
            'confidence': float(decision_pred[0])
        }
