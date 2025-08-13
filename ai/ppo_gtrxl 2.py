import tensorflow as tf
from typing import Dict, List, Tuple
import numpy as np

class GTrXLBlock(tf.keras.layers.Layer):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model)
        self.ff = tf.keras.Sequential([
            tf.keras.layers.Dense(d_model * 4, activation='relu'),
            tf.keras.layers.Dense(d_model)
        ])
        self.ln1 = tf.keras.layers.LayerNormalization()
        self.ln2 = tf.keras.layers.LayerNormalization()
        
    def call(self, x: tf.Tensor) -> tf.Tensor:
        attn_out = self.mha(x, x)
        x = self.ln1(x + attn_out)
        ff_out = self.ff(x)
        return self.ln2(x + ff_out)

class PPOGTrXL(tf.keras.Model):
    def __init__(self, 
                 state_dim: int,
                 action_dim: int,
                 seq_len: int = 64,
                 d_model: int = 512,
                 num_layers: int = 6,
                 num_heads: int = 8):
        super().__init__()
        
        self.state_encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(d_model, activation='relu'),
            tf.keras.layers.LayerNormalization()
        ])
        
        self.transformer_blocks = [
            GTrXLBlock(d_model, num_heads) 
            for _ in range(num_layers)
        ]
        
        self.policy_head = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(action_dim, activation='softmax')
        ])
        
        self.value_head = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        
    def call(self, states: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        x = self.state_encoder(states)
        
        for block in self.transformer_blocks:
            x = block(x)
            
        policy = self.policy_head(x)
        value = self.value_head(x)
        
        return policy, value
        
    def train_step(self, data: Tuple) -> Dict[str, float]:
        states, actions, advantages, returns, old_policies = data
        
        with tf.GradientTape() as tape:
            policies, values = self(states)
            
            # Policy loss
            ratio = tf.exp(tf.math.log(policies + 1e-10) - 
                         tf.math.log(old_policies + 1e-10))
            min_advantage = tf.where(
                advantages > 0,
                (1 + 0.2) * advantages,
                (1 - 0.2) * advantages,
            )
            policy_loss = -tf.reduce_mean(
                tf.minimum(ratio * advantages, min_advantage)
            )
            
            # Value loss
            value_loss = tf.reduce_mean(tf.square(returns - values))
            
            # Entropy bonus
            entropy = tf.reduce_mean(
                tf.reduce_sum(-policies * tf.math.log(policies + 1e-10), axis=-1)
            )
            
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
            
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        
        return {
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'entropy': entropy
        }
