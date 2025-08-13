import tensorflow as tf
from stable_baselines3 import PPO
class HybridTradingModel:
    def __init__(self):
        self.technical_model = self.build_cnn_lstm()
        self.decision_model = PPO("MlpPolicy", ...)
