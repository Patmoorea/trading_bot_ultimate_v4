from typing import Dict, List, Optional
import numpy as np
import tensorflow as tf
from .cnn_lstm import CNNLSTMModel
from .ppo_transformer import PPOTransformer
from .quantum_processor import QuantumProcessor

class ExtendedHybridModel:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.cnn_lstm = CNNLSTMModel(
            input_shape=(100, 5, 4),
            lstm_units=512
        )
        self.ppo_transformer = PPOTransformer(
            n_layers=6,
            d_model=512
        )
        self.quantum_processor = QuantumProcessor()
        self._initialize_models()

    def _default_config(self) -> Dict:
        return {
            'learning_rate': 1e-4,
            'batch_size': 32,
            'epochs': 100,
            'validation_split': 0.2
        }

    def _initialize_models(self) -> None:
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.config['learning_rate']
        )
        self.models_initialized = True

    async def process_multi_timeframe(self, data: Dict) -> Dict:
        features = await self.extract_features(data)
        sentiment = await self.process_sentiment(data)
        predictions = await self.merge_predictions(features, sentiment)
        return self._post_process_predictions(predictions)

    async def extract_features(self, data: Dict) -> np.ndarray:
        technical_features = self.cnn_lstm.extract_features(data['technical'])
        market_features = self.ppo_transformer.process_market_data(data['market'])
        quantum_features = await self.quantum_processor.analyze(data['quantum'])
        
        return np.concatenate([
            technical_features,
            market_features,
            quantum_features
        ], axis=1)

    async def process_sentiment(self, data: Dict) -> np.ndarray:
        news_sentiment = await self.analyze_news(data.get('news', []))
        market_sentiment = self.analyze_market_sentiment(data['market'])
        onchain_sentiment = await self.analyze_onchain_data(data.get('onchain', {}))
        
        return np.mean([
            news_sentiment,
            market_sentiment,
            onchain_sentiment
        ], axis=0)

    async def merge_predictions(self, 
                              features: np.ndarray, 
                              sentiment: np.ndarray) -> Dict:
        technical_pred = self.cnn_lstm.predict(features)
        market_pred = self.ppo_transformer.predict(features)
        quantum_adj = await self.quantum_processor.adjust_predictions(
            technical_pred,
            market_pred,
            sentiment
        )
        
        return {
            'final_prediction': quantum_adj,
            'confidence': self._calculate_confidence(
                technical_pred,
                market_pred,
                quantum_adj
            ),
            'components': {
                'technical': technical_pred,
                'market': market_pred,
                'quantum': quantum_adj
            }
        }

    def _calculate_confidence(self, *predictions) -> float:
        weights = [0.4, 0.4, 0.2]  # Poids pour chaque composant
        weighted_pred = np.average(predictions, weights=weights, axis=0)
        variance = np.var(predictions, axis=0)
        return float(np.mean(weighted_pred) / (1 + np.mean(variance)))

    def _post_process_predictions(self, predictions: Dict) -> Dict:
        return {
            'prediction': float(predictions['final_prediction']),
            'confidence': float(predictions['confidence']),
            'components': {
                k: float(v) 
                for k, v in predictions['components'].items()
            },
            'timestamp': utils.get_current_timestamp(),
            'metadata': {
                'model_version': self.__class__.__version__,
                'config': self.config
            }
        }
