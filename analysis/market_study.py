import pandas as pd
import numpy as np
from typing import Dict, List
from dataclasses import dataclass
from sklearn.cluster import KMeans
from hmmlearn import hmm
@dataclass
class MarketStudyConfig:
    study_period: str = "1M"  # 1 mois
    min_data_points: int = 1000
    regime_count: int = 5
    confidence_threshold: float = 0.75
class MarketStudySystem:
    def __init__(self, config: MarketStudyConfig = MarketStudyConfig()):
        self.config = config
        self.market_regimes = self._init_regime_detector()
        self.current_plan = None
    def _init_regime_detector(self):
        """Initialise le détecteur de régimes"""
        return {
            'hmm': hmm.GaussianHMM(
                n_components=self.config.regime_count,
                covariance_type="full"
            ),
            'kmeans': KMeans(
                n_clusters=self.config.regime_count,
                random_state=42
            )
        }
    def analyze_market(self, data: pd.DataFrame) -> Dict:
        """Analyse complète du marché"""
        if len(data) < self.config.min_data_points:
            raise ValueError("Insufficient data for analysis")
        results = {
            'regime': self._detect_regime(data),
            'volatility': self._analyze_volatility(data),
            'trends': self._analyze_trends(data),
            'support_resistance': self._find_levels(data),
            'correlation': self._analyze_correlation(data)
        }
        self.current_plan = self._generate_trading_plan(results)
        return results
    def _detect_regime(self, data: pd.DataFrame) -> Dict:
        """Détecte le régime de marché actuel"""
        # Préparation des features
        features = np.column_stack([
            data['returns'].values,
            data['volatility'].values
        ])
        # Détection HMM
        self.market_regimes['hmm'].fit(features)
        hmm_state = self.market_regimes['hmm'].predict(features)[-1]
        # Clustering KMeans
        kmeans_state = self.market_regimes['kmeans'].fit_predict(features)[-1]
        # Mapping des régimes
        regime_map = {
            0: 'High Volatility Bull',
            1: 'Low Volatility Bull',
            2: 'High Volatility Bear',
            3: 'Low Volatility Bear',
            4: 'Sideways'
        }
        return {
            'current_regime': regime_map[hmm_state],
            'confidence': float(self.market_regimes['hmm'].score(features)),
            'alternative_regime': regime_map[kmeans_state]
        }
    def _analyze_volatility(self, data: pd.DataFrame) -> Dict:
        """Analyse la volatilité"""
        returns = data['close'].pct_change()
        return {
            'current_vol': float(returns.std() * np.sqrt(252)),
            'historical_vol': float(returns.rolling(20).std() * np.sqrt(252)),
            'regime_change_prob': self._calc_regime_change_prob(returns)
        }
    def _analyze_trends(self, data: pd.DataFrame) -> Dict:
        """Analyse les tendances"""
        trends = {}
        for period in [20, 50, 200]:
            ma = data['close'].rolling(period).mean()
            trends[f'ma_{period}'] = {
                'value': float(ma.iloc[-1]),
                'direction': 'up' if ma.iloc[-1] > ma.iloc[-2] else 'down',
                'strength': float(abs(ma.iloc[-1] - ma.iloc[-2]) / ma.iloc[-2])
            }
        return trends
    def _find_levels(self, data: pd.DataFrame) -> Dict:
        """Trouve les niveaux de support et résistance"""
        # Utilisation de KMeans pour identifier les clusters de prix
        prices = data['close'].values.reshape(-1, 1)
        n_clusters = min(5, len(prices) // 100)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(prices)
        levels = sorted(kmeans.cluster_centers_.flatten())
        current_price = float(data['close'].iloc[-1])
        return {
            'supports': [l for l in levels if l < current_price],
            'resistances': [l for l in levels if l > current_price],
            'strength': self._calculate_level_strength(levels, data)
        }
    def _generate_trading_plan(self, analysis: Dict) -> Dict:
        """Génère un plan de trading basé sur l'analyse"""
        return {
            'regime_strategy': self._get_regime_strategy(analysis['regime']),
            'entry_points': self._calculate_entry_points(analysis),
            'position_sizing': self._calculate_position_sizing(analysis),
            'risk_parameters': self._define_risk_parameters(analysis)
        }
    def _get_regime_strategy(self, regime: Dict) -> Dict:
        """Définit la stratégie en fonction du régime"""
        strategies = {
            'High Volatility Bull': {
                'approach': 'aggressive_long',
                'timeframes': ['15m', '1h'],
                'indicators': ['RSI', 'MACD', 'BB']
            },
            'Low Volatility Bull': {
                'approach': 'conservative_long',
                'timeframes': ['1h', '4h'],
                'indicators': ['EMA', 'ATR', 'OBV']
            },
            'High Volatility Bear': {
                'approach': 'defensive',
                'timeframes': ['5m', '15m'],
                'indicators': ['Stoch RSI', 'CMF', 'KC']
            },
            'Low Volatility Bear': {
                'approach': 'ranging',
                'timeframes': ['1h', '4h'],
                'indicators': ['BB', 'ATR', 'MFI']
            },
            'Sideways': {
                'approach': 'mean_reversion',
                'timeframes': ['5m', '15m'],
                'indicators': ['BB', 'RSI', 'Volume']
            }
        }
        return strategies.get(regime['current_regime'], strategies['Sideways'])
