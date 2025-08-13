from typing import Dict, List
import numpy as np
from datetime import datetime, timedelta
class EnhancedCircuitBreaker:
    def __init__(self):
        self.triggers: Dict[str, List[Dict]] = {
            'price': [],
            'volume': [],
            'spread': [],
            'volatility': [],
            'liquidity': []
        }
        self.cooldown_periods: Dict[str, timedelta] = {
            'price': timedelta(minutes=5),
            'volume': timedelta(minutes=15),
            'spread': timedelta(minutes=3),
            'volatility': timedelta(minutes=10),
            'liquidity': timedelta(minutes=7)
        }
    def check_conditions(self, market_data: Dict) -> List[str]:
        triggered = []
        # Vérification prix
        if self._check_price_volatility(market_data):
            triggered.append('price')
        # Vérification volume
        if self._check_volume_spike(market_data):
            triggered.append('volume')
        # Vérification spread
        if self._check_spread_explosion(market_data):
            triggered.append('spread')
        # Vérification liquidité
        if self._check_liquidity_drain(market_data):
            triggered.append('liquidity')
        return triggered
    def _check_price_volatility(self, data: Dict) -> bool:
        recent_prices = data.get('recent_prices', [])
        if len(recent_prices) < 10:
            return False
        volatility = np.std(recent_prices) / np.mean(recent_prices)
        return volatility > 0.02  # 2% threshold
    def _check_volume_spike(self, data: Dict) -> bool:
        current_volume = data.get('current_volume', 0)
        avg_volume = data.get('average_volume', current_volume)
        return current_volume > avg_volume * 3  # 300% spike
    def _check_spread_explosion(self, data: Dict) -> bool:
        current_spread = data.get('current_spread', 0)
        avg_spread = data.get('average_spread', current_spread)
        return current_spread > avg_spread * 5  # 500% spike
    def _check_liquidity_drain(self, data: Dict) -> bool:
        current_liquidity = data.get('orderbook_depth', 0)
        normal_liquidity = data.get('average_depth', current_liquidity)
        return current_liquidity < normal_liquidity * 0.3  # 70% drop
