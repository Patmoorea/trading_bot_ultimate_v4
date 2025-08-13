import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
@dataclass
class OrderFlowConfig:
    tick_size: float
    depth_levels: int = 10
    refresh_rate: int = 100  # ms
class OrderFlowAnalysis:
    def __init__(self, config: OrderFlowConfig):
        self.config = config
        self.order_history = []
        self.trade_history = []
    def analyze_orderflow(
        self,
        orderbook: pd.DataFrame,
        trades: pd.DataFrame
    ) -> Dict[str, any]:
        """Analyse complète de l'orderflow"""
        # 1. Imbalance Analysis
        imbalance = self._calculate_imbalance(orderbook)
        # 2. Trade Flow Analysis
        trade_flow = self._analyze_trades(trades)
        # 3. Order Book Heatmap
        heatmap = self._generate_heatmap(orderbook)
        # 4. Large Orders Detection
        whales = self._detect_whales(trades)
        return {
            'imbalance': imbalance,
            'trade_flow': trade_flow,
            'heatmap': heatmap,
            'whales': whales
        }
    def _calculate_imbalance(self, orderbook: pd.DataFrame) -> Dict[str, float]:
        """Calcul d'imbalance bid/ask"""
        bids = orderbook[orderbook['side'] == 'bid']
        asks = orderbook[orderbook['side'] == 'ask']
        bid_volume = bids['volume'].sum()
        ask_volume = asks['volume'].sum()
        total_volume = bid_volume + ask_volume
        if total_volume == 0:
            return {'bid_ratio': 0.5, 'ask_ratio': 0.5}
        return {
            'bid_ratio': bid_volume / total_volume,
            'ask_ratio': ask_volume / total_volume,
            'imbalance_score': (bid_volume - ask_volume) / total_volume
        }
    def _analyze_trades(self, trades: pd.DataFrame) -> Dict[str, float]:
        """Analyse détaillée des trades"""
        # Agressive orders
        aggressive_buys = trades[trades['aggressor_side'] == 'buy']
        aggressive_sells = trades[trades['aggressor_side'] == 'sell']
        # Volume Analysis
        buy_volume = aggressive_buys['volume'].sum()
        sell_volume = aggressive_sells['volume'].sum()
        # Trade Flow Score
        total_volume = buy_volume + sell_volume
        if total_volume == 0:
            return {'flow_score': 0}
        flow_score = (buy_volume - sell_volume) / total_volume
        return {
            'flow_score': flow_score,
            'aggressive_buy_ratio': buy_volume / total_volume if total_volume > 0 else 0,
            'aggressive_sell_ratio': sell_volume / total_volume if total_volume > 0 else 0,
            'total_trades': len(trades),
            'average_trade_size': trades['volume'].mean()
        }
    def _generate_heatmap(self, orderbook: pd.DataFrame) -> pd.DataFrame:
        """Génère une heatmap du carnet d'ordres"""
        # Prix min/max pour le range
        min_price = orderbook['price'].min()
        max_price = orderbook['price'].max()
        # Création des niveaux de prix
        price_levels = np.linspace(min_price, max_price, self.config.depth_levels)
        # Initialisation heatmap
        heatmap = pd.DataFrame(index=price_levels, columns=['bid_volume', 'ask_volume'])
        # Remplissage
        for price in price_levels:
            price_range = orderbook[
                (orderbook['price'] >= price - self.config.tick_size) &
                (orderbook['price'] < price + self.config.tick_size)
            ]
            heatmap.loc[price, 'bid_volume'] = \
                price_range[price_range['side'] == 'bid']['volume'].sum()
            heatmap.loc[price, 'ask_volume'] = \
                price_range[price_range['side'] == 'ask']['volume'].sum()
        return heatmap
    def _detect_whales(self, trades: pd.DataFrame) -> List[Dict[str, any]]:
        """Détection des ordres whale"""
        whale_threshold = self.config.tick_size * 1000  # Exemple
        whales = []
        for _, trade in trades.iterrows():
            if trade['volume'] * trade['price'] >= whale_threshold:
                whales.append({
                    'timestamp': trade['timestamp'],
                    'price': trade['price'],
                    'volume': trade['volume'],
                    'side': trade['side'],
                    'impact_score': self._calculate_impact(trade)
                })
        return whales
    def _calculate_impact(self, trade: pd.Series) -> float:
        """Calcule l'impact potentiel d'un ordre whale"""
        # Formule basique : volume * prix / moyenne_volume
        return trade['volume'] * trade['price'] / np.mean(
            [t['volume'] * t['price'] for t in self.trade_history[-100:]]
        ) if self.trade_history else 1.0
