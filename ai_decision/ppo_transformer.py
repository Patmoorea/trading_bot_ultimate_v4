import torch
from stable_baselines3 import PPO
from torch.nn import TransformerEncoderLayer
class PPOTradingAgent:
    def __init__(self, env):
        self.policy_kwargs = {
            "activation_fn": torch.nn.ReLU,
            "net_arch": [dict(pi=[512, 512], vf=[512, 512])],
            "transformer_layer": TransformerEncoderLayer(
                d_model=512,
                nhead=8,
                dim_feedforward=2048,
                batch_first=True
            )
        }
        self.model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=self.policy_kwargs,
            n_steps=2048,
            batch_size=128,
            n_epochs=10,
            learning_rate=3e-4,
            verbose=1
        )
    def train(self, timesteps=1_000_000):
        self.model.learn(total_timesteps=timesteps)
        self.model.save("ppo_trading_transformer")
