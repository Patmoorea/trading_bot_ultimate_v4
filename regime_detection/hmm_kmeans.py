"""
Market Regime Detector
Version: 2.0.0
Détection des régimes de marché utilisant HMM + K-means
avec optimisation pour Apple Silicon M1/M4
"""
import numpy as np
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta
import joblib
import os
logger = logging.getLogger(__name__)
class MarketRegimeDetector:
    """
    Détecteur de régimes de marché hybride HMM + K-means
    avec support de cache et optimisation M1/M4
    """
    def __init__(self, 
                 n_regimes: int = 3,
                 lookback_period: int = 90,
                 cache_dir: str = "./models/regime_detection"):
        """
        Initialize le détecteur de régimes
        Args:
            n_regimes: Nombre de régimes à détecter
            lookback_period: Période d'historique en jours
            cache_dir: Répertoire pour le cache des modèles
        """
        self.n_regimes = n_regimes
        self.lookback_period = lookback_period
        self.cache_dir = cache_dir
        # Création du répertoire de cache
        os.makedirs(cache_dir, exist_ok=True)
        # Initialisation des modèles
        self.scaler = StandardScaler()
        self.hmm_model = hmm.GaussianHMM(
            n_components=n_regimes,
            covariance_type="full",
            n_iter=100,
            random_state=42
        )
        self.kmeans = KMeans(
            n_clusters=n_regimes,
            random_state=42
        )
        # Cache pour les prédictions
        self.cache = {}
        self._last_training = None
        self.regime_characteristics = None
    def _extract_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        Extrait les features des données de marché
        Args:
            data: DataFrame avec OHLCV
        Returns:
            np.ndarray: Features extraites
        """
        # Calcul des rendements
        returns = data['close'].pct_change().fillna(0)
        # Volatilité mobile
        volatility = returns.rolling(window=20).std().fillna(0)
        # Volume normalisé
        volume_ma = data['volume'].rolling(window=20).mean()
        norm_volume = (data['volume'] / volume_ma).fillna(1)
        # Extraction de features techniques
        features = pd.DataFrame({
            'returns': returns,
            'volatility': volatility,
            'volume_ratio': norm_volume,
            'hl_range': (data['high'] - data['low']) / data['close'],
            'price_ma_ratio': data['close'] / data['close'].rolling(20).mean().fillna(method='bfill')
        })
        return self.scaler.fit_transform(features)
    def fit(self, data: pd.DataFrame) -> None:
        """
        Entraîne le modèle sur les données historiques
        Args:
            data: DataFrame avec colonnes OHLCV
        """
        try:
            features = self._extract_features(data)
            # Entraînement HMM
            self.hmm_model.fit(features)
            hmm_states = self.hmm_model.predict(features)
            # Entraînement K-means
            self.kmeans.fit(features)
            kmeans_clusters = self.kmeans.predict(features)
            # Fusion des états
            self.regime_characteristics = self._analyze_regimes(
                data, hmm_states, kmeans_clusters
            )
            # Sauvegarde du modèle
            self._save_model()
            self._last_training = datetime.utcnow()
            logger.info("✅ Entraînement du détecteur de régimes réussi")
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'entraînement: {str(e)}")
            raise
    def predict(self, data: pd.DataFrame) -> Dict:
        """
        Prédit le régime actuel du marché
        Args:
            data: DataFrame avec dernières données OHLCV
        Returns:
            Dict avec regime_id, probabilités et caractéristiques
        """
        try:
            # Vérification du cache
            key = data.index[-1]
            if key in self.cache:
                return self.cache[key]
            # Extraction features
            features = self._extract_features(data)
            # Prédictions
            hmm_state = self.hmm_model.predict(features)[-1]
            kmeans_cluster = self.kmeans.predict(features)[-1]
            # Probabilités des états
            state_probs = np.exp(self.hmm_model.score_samples(features))[0]
            # Construction résultat
            regime = {
                'timestamp': key,
                'regime_id': hmm_state,
                'cluster_id': kmeans_cluster,
                'probabilities': {
                    f'regime_{i}': float(p) 
                    for i, p in enumerate(state_probs)
                },
                'characteristics': self.regime_characteristics[hmm_state]
                if self.regime_characteristics else None
            }
            # Mise en cache
            self.cache[key] = regime
            return regime
        except Exception as e:
            logger.error(f"❌ Erreur prédiction: {str(e)}")
            return None
    def _analyze_regimes(self, 
                        data: pd.DataFrame,
                        hmm_states: np.ndarray,
                        kmeans_clusters: np.ndarray) -> Dict:
        """
        Analyse les caractéristiques de chaque régime
        Returns:
            Dict avec statistiques par régime
        """
        regime_stats = {}
        for regime in range(self.n_regimes):
            mask = hmm_states == regime
            regime_data = data[mask]
            if len(regime_data) == 0:
                continue
            returns = regime_data['close'].pct_change()
            regime_stats[regime] = {
                'avg_return': float(returns.mean()),
                'volatility': float(returns.std()),
                'avg_volume': float(regime_data['volume'].mean()),
                'avg_range': float((regime_data['high'] - regime_data['low']).mean()),
                'duration': int(mask.sum()),
                'last_seen': regime_data.index[-1].isoformat(),
                'cluster_association': int(kmeans_clusters[mask].mode()[0])
            }
        return regime_stats
    def _save_model(self) -> None:
        """Sauvegarde le modèle dans le cache"""
        try:
            path = os.path.join(self.cache_dir, "regime_detector.joblib")
            joblib.dump({
                'hmm': self.hmm_model,
                'kmeans': self.kmeans,
                'scaler': self.scaler,
                'characteristics': self.regime_characteristics,
                'last_training': self._last_training
            }, path)
            logger.info(f"✅ Modèle sauvegardé: {path}")
        except Exception as e:
            logger.error(f"❌ Erreur sauvegarde modèle: {str(e)}")
    def load_model(self) -> bool:
        """Charge le modèle depuis le cache"""
        try:
            path = os.path.join(self.cache_dir, "regime_detector.joblib")
            if not os.path.exists(path):
                return False
            cached = joblib.load(path)
            self.hmm_model = cached['hmm']
            self.kmeans = cached['kmeans']
            self.scaler = cached['scaler']
            self.regime_characteristics = cached['characteristics']
            self._last_training = cached['last_training']
            logger.info(f"✅ Modèle chargé: {path}")
            return True
        except Exception as e:
            logger.error(f"❌ Erreur chargement modèle: {str(e)}")
            return False
    def needs_training(self, max_age_days: int = 7) -> bool:
        """Vérifie si le modèle doit être réentraîné"""
        if not self._last_training:
            return True
        age = datetime.utcnow() - self._last_training
        return age > timedelta(days=max_age_days)
