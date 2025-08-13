import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from typing import Dict, List
import talib
class MarketRegimeDetector:
    def __init__(self, n_regimes: int = 5):
        self.n_regimes = n_regimes
        self.hmm = GaussianMixture(n_components=n_regimes)
        self.kmeans = KMeans(n_clusters=n_regimes)
        self.regime_labels = {
            0: 'High Volatility Bull',
            1: 'Low Volatility Bull',
            2: 'High Volatility Bear',
            3: 'Low Volatility Bear',
            4: 'Sideways'
        }
    def detect(self, data: pd.DataFrame) -> Dict:
        """Détecte le régime de marché actuel"""
        # Calcul des features
        features = self._calculate_features(data)
        # Combinaison HMM + K-Means
        hmm_pred = self.hmm.predict(features)
        kmeans_pred = self.kmeans.predict(features)
        # Fusion des prédictions
        regime = self._combine_predictions(hmm_pred, kmeans_pred)
        # Caractéristiques du régime
        characteristics = self._get_regime_characteristics(
            data, 
            features,
            regime
        )
        return {
            'regime': self.regime_labels[regime],
            'characteristics': characteristics,
            'confidence': self._calculate_confidence(
                features,
                regime,
                hmm_pred,
                kmeans_pred
            )
        }
    def _calculate_features(self, data: pd.DataFrame) -> np.ndarray:
        """Calcule les features pour la détection"""
        # Rendements
        returns = np.log(data['close'] / data['close'].shift(1))
        # Volatilité
        volatility = returns.rolling(window=20).std()
        # Momentum
        momentum = returns.rolling(window=10).mean()
        # Volume
        volume_ma = data['volume'].rolling(window=20).mean()
        rel_volume = data['volume'] / volume_ma
        # Tendance
        sma20 = talib.SMA(data['close'], timeperiod=20)
        sma50 = talib.SMA(data['close'], timeperiod=50)
        trend = sma20 / sma50 - 1
        features = np.column_stack([
            returns,
            volatility,
            momentum,
            rel_volume,
            trend
        ])
        return features
    def _combine_predictions(self, hmm_pred: np.ndarray, kmeans_pred: np.ndarray) -> int:
        """Combine les prédictions HMM et K-Means"""
        # Matrice de confusion entre les deux modèles
        confusion = np.zeros((self.n_regimes, self.n_regimes))
        for i, j in zip(hmm_pred, kmeans_pred):
            confusion[i, j] += 1
        # Assignation optimale
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(-confusion)
        # Retourne le régime le plus probable
        regime_votes = np.bincount(hmm_pred)
        regime = np.argmax(regime_votes)
        return regime
    def _get_regime_characteristics(
        self,
        data: pd.DataFrame,
        features: np.ndarray,
        regime: int
    ) -> Dict:
        """Calcule les caractéristiques du régime"""
        regime_data = data[features[:, -1] == regime]
        return {
            'avg_volatility': float(regime_data['close'].pct_change().std()),
            'avg_volume': float(regime_data['volume'].mean()),
            'avg_return': float(regime_data['close'].pct_change().mean()),
            'duration': len(regime_data),
            'strength': self._calculate_regime_strength(features, regime)
        }
    def _calculate_confidence(
        self,
        features: np.ndarray,
        regime: int,
        hmm_pred: np.ndarray,
        kmeans_pred: np.ndarray
    ) -> float:
        """Calcule la confiance dans la détection"""
        # Probabilités HMM
        hmm_probs = self.hmm.predict_proba(features)
        hmm_conf = hmm_probs[:, regime].mean()
        # Distance aux centroïdes K-Means
        kmeans_distances = self.kmeans.transform(features)
        kmeans_conf = 1 / (1 + kmeans_distances[:, regime].mean())
        # Accord entre modèles
        model_agreement = (hmm_pred == kmeans_pred).mean()
        # Confiance finale
        confidence = 0.4 * hmm_conf + 0.4 * kmeans_conf + 0.2 * model_agreement
        return float(confidence)
    def _calculate_regime_strength(self, features: np.ndarray, regime: int) -> float:
        """Calcule la force du régime actuel"""
        # Distance aux autres régimes
        regime_features = features[features[:, -1] == regime]
        other_features = features[features[:, -1] != regime]
        if len(regime_features) == 0 or len(other_features) == 0:
            return 0.0
        from scipy.spatial.distance import cdist
        distances = cdist(regime_features, other_features).mean()
        # Normalisation
        strength = 1 / (1 + np.exp(-distances))
        return float(strength)
