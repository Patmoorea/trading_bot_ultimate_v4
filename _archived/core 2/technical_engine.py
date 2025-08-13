import numpy as np
import tensorflow as tf
from datetime import datetime
class TechnicalEngine:
    def __init__(self):
        print("⚡ Moteur Technique Initialisé (GPU Optimisé)")
    def _clean_data(self, data):
        """Filtre les données pour ne garder que les valeurs numériques"""
        if isinstance(data, (list, np.ndarray)):
            return np.array([x for x in data if isinstance(x, (int, float))], dtype=np.float32)
        return np.array([], dtype=np.float32)
    def compute(self, market_data):
        '''Analyse les données marché réelles
        Args:
            market_data: DataFrame avec colonnes ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        '''
        try:
            clean_data = self._clean_data(data)
            if len(clean_data) == 0:
                return {"error": "No numeric data found"}
            data_tensor = tf.convert_to_tensor(clean_data)
            with tf.device('/GPU:0'):
                trend = tf.reduce_mean(data_tensor)
            return {
                "trend": float(trend.numpy()),
                "signal": "BUY" if trend > 2 else "SELL",
                "confidence": 0.95
            }
        except Exception as e:
            print(f"Erreur de calcul: {e}")
            return {"error": str(e)}
