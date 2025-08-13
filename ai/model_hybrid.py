"""
Advanced Hybrid Deep Learning Model
Version: 1.0.0
Created: 2025-07-02 19:41:13
Author: Patmoorea
"""

from typing import Dict
import numpy as np
from datetime import datetime, timezone
from src.ai.hybrid.extended_model import ExtendedHybridModel
from src.ai_models.hybrid.cnn_lstm_enhanced import EnhancedCNNLSTM


class DeepLearningModel:
    """
    Modèle hybride combinant approches stratégique et tactique
    pour le trading automatique multi-timeframes.
    """
    
    def __init__(self):
        # Modèle complet pour les décisions stratégiques
        self.strategic_model = ExtendedHybridModel()
        
        # Modèle rapide pour les décisions tactiques
        self.tactical_model = EnhancedCNNLSTM()
        
        # Timestamp d'initialisation
        self.init_time = datetime.now(timezone.utc)
        self.last_update = self.init_time
        
    async def predict(self, data: Dict) -> Dict:
        # Décisions stratégiques (long terme, avec toutes les sources)
        strategic_pred = await self.strategic_model.process_multi_timeframe(data)
        
        # Décisions tactiques (court terme, données techniques)
        tactical_pred = self.tactical_model.predict(data['technical'])
        
        # Mise à jour du timestamp
        self.last_update = datetime.now(timezone.utc)
        
        # Fusion des prédictions selon l'horizon temporel
        return {
            'strategic': strategic_pred,
            'tactical': tactical_pred,
            'final_decision': self._merge_decisions(
                strategic_pred,
                tactical_pred,
                timeframe=data.get('timeframe', '1h')
            )
        }
    
    def _merge_decisions(self, strategic, tactical, timeframe):
        # Pondération différente selon le timeframe
        timeframe_weights = {
            '1m':  (0.1, 0.9),  # 10% stratégique, 90% tactique
            '5m':  (0.2, 0.8),
            '15m': (0.3, 0.7),
            '1h':  (0.5, 0.5),
            '4h':  (0.7, 0.3),
            '1d':  (0.9, 0.1)   # 90% stratégique, 10% tactique
        }
        
        s_weight, t_weight = timeframe_weights.get(timeframe, (0.5, 0.5))
        return {
            'position': s_weight * strategic['final_prediction'] + 
                       t_weight * tactical,
            'confidence': (s_weight * strategic['confidence'] + 
                         t_weight * self.tactical_model.get_confidence())
        }
    
    def get_status(self) -> Dict:
        """
        Retourne le statut actuel du modèle
        """
        return {
            'initialized_at': self.init_time.strftime('%Y-%m-%d %H:%M:%S'),
            'last_update': self.last_update.strftime('%Y-%m-%d %H:%M:%S'),
            'strategic_status': self.strategic_model.get_status() if hasattr(self.strategic_model, 'get_status') else 'OK',
            'tactical_status': self.tactical_model.get_status() if hasattr(self.tactical_model, 'get_status') else 'OK'
        }
