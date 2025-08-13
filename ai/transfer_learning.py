from typing import Dict, List
import tensorflow as tf
import numpy as np
from .ppo_gtrxl import PPOGTrXL
class TransferLearningManager:
    def __init__(self):
        self.models: Dict[str, PPOGTrXL] = {}
        self.correlation_matrix: Dict[str, Dict[str, float]] = {}
        self.performance_metrics: Dict[str, List[float]] = {}
    def add_model(self, pair: str, model: PPOGTrXL):
        self.models[pair] = model
        self.performance_metrics[pair] = []
    def update_correlation(self, pair1: str, pair2: str, correlation: float):
        if pair1 not in self.correlation_matrix:
            self.correlation_matrix[pair1] = {}
        self.correlation_matrix[pair1][pair2] = correlation
    def update_performance(self, pair: str, metric: float):
        self.performance_metrics[pair].append(metric)
    async def transfer_knowledge(self, 
                               source_pair: str, 
                               target_pair: str,
                               transfer_ratio: float = 0.5):
        if source_pair not in self.models or target_pair not in self.models:
            raise ValueError("Source or target model not found")
        correlation = self.correlation_matrix.get(source_pair, {}).get(target_pair, 0)
        if correlation < 0.5:  # Seuil minimum de corrélation
            print(f"Correlation too low between {source_pair} and {target_pair}")
            return
        # Transfer des poids avec ratio adaptatif
        source_weights = self.models[source_pair].get_weights()
        target_weights = self.models[target_pair].get_weights()
        new_weights = []
        for sw, tw in zip(source_weights, target_weights):
            merged = sw * transfer_ratio + tw * (1 - transfer_ratio)
            new_weights.append(merged)
        self.models[target_pair].set_weights(new_weights)
    def get_best_source_model(self, target_pair: str) -> str:
        if target_pair not in self.correlation_matrix:
            return None
        best_correlation = -1
        best_source = None
        for source, correlation in self.correlation_matrix[target_pair].items():
            if correlation > best_correlation and source in self.models:
                recent_performance = np.mean(self.performance_metrics[source][-10:])
                if recent_performance > 0:  # Only transfer from profitable models
                    best_correlation = correlation
                    best_source = source
        return best_source
