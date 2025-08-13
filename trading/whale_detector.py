import numpy as np
from typing import List, Dict
import pandas as pd
class EnhancedWhaleDetector:
    def __init__(self, min_whale_size: float = 100_000):
        self.min_whale_size = min_whale_size
        self.historical_movements: List[Dict] = []
        self.known_patterns: Dict[str, np.ndarray] = {}
    def analyze_order(self, order: Dict) -> Dict:
        if float(order['amount']) < self.min_whale_size:
            return {'is_whale': False}
        pattern = self._extract_pattern(order)
        similarity = self._compare_patterns(pattern)
        impact = self._calculate_impact(order)
        return {
            'is_whale': True,
            'pattern_match': similarity,
            'estimated_impact': impact,
            'risk_level': self._assess_risk(impact, similarity)
        }
    def _extract_pattern(self, order: Dict) -> np.ndarray:
        # Conversion des données en pattern analysable
        features = [
            float(order['amount']),
            float(order['price']),
            order.get('taker') == True,
            len(order.get('related_orders', [])),
        ]
        return np.array(features)
    def _compare_patterns(self, pattern: np.ndarray) -> float:
        if not self.known_patterns:
            return 0.0
        similarities = []
        for known in self.known_patterns.values():
            similarity = 1 - (np.linalg.norm(pattern - known) / 
                            (np.linalg.norm(pattern) + np.linalg.norm(known)))
            similarities.append(similarity)
        return max(similarities)
    def _calculate_impact(self, order: Dict) -> float:
        amount = float(order['amount'])
        price = float(order['price'])
        market_volume = order.get('24h_volume', amount * 100)  # fallback
        return (amount * price) / market_volume
    def _assess_risk(self, impact: float, similarity: float) -> str:
        if impact > 0.1 and similarity > 0.8:
            return 'HIGH'
        elif impact > 0.05 or similarity > 0.6:
            return 'MEDIUM'
        return 'LOW'
