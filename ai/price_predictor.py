"""
Price Predictor Module
"""
import os
import logging
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
class PricePredictor:
    def __init__(self, symbol: str, timeframe: str):
        """Initialize price predictor"""
        self.symbol = symbol
        self.timeframe = timeframe
        self.sequence_length = 60
        self.features = ['close', 'volume', 'high', 'low']
        # Create model directory structure
        symbol_base = symbol.split('/')[0]
        self.model_dir = os.path.join('models', symbol_base)
        self.model_name = f"{symbol.replace('/', '_')}_{timeframe}_model.keras"  # Ajout de .keras ici
        self.model_path = os.path.join(self.model_dir, self.model_name)
        # Initialize scalers
        self.feature_scaler = MinMaxScaler()
        self.price_scaler = MinMaxScaler()
        # Create model directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)
        # Try to load existing model or create new one
        try:
            self.model = load_model(self.model_path)
            logging.info(f"Loaded existing model from {self.model_path}")
        except:
            self.model = self._create_model()
            logging.info("Created new model")
    def _create_model(self) -> Sequential:
        """Create and compile the LSTM model"""
        model = Sequential([
            LSTM(100, return_sequences=True, input_shape=(self.sequence_length, len(self.features))),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model
    def prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for training or prediction"""
        if len(data) < self.sequence_length + 1:
            raise ValueError(f"Not enough data points. Need at least {self.sequence_length + 1}")
        # Scale features
        feature_data = data[self.features].values
        scaled_features = self.feature_scaler.fit_transform(feature_data)
        # Scale prices separately for better prediction
        price_data = data[['close']].values
        scaled_prices = self.price_scaler.fit_transform(price_data)
        X, y = [], []
        for i in range(len(scaled_features) - self.sequence_length):
            X.append(scaled_features[i:(i + self.sequence_length)])
            y.append(scaled_prices[i + self.sequence_length])
        return np.array(X), np.array(y)
    def train(self, data: pd.DataFrame, epochs: int = 50, batch_size: int = 32) -> Dict[str, list]:
        """Train the model on historical data"""
        X, y = self.prepare_data(data)
        history = self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
        self.model.save(self.model_path)
        return history.history
    def predict(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Make price predictions"""
        X, _ = self.prepare_data(data)
        if len(X) == 0:
            raise ValueError("Not enough data for prediction")
        # Make prediction
        scaled_prediction = self.model.predict(X[-1:], verbose=0)
        # Inverse transform the prediction
        prediction = self.price_scaler.inverse_transform(scaled_prediction)
        # Calculate confidence based on recent prediction accuracy
        recent_predictions = self.model.predict(X[-10:], verbose=0)
        recent_predictions = self.price_scaler.inverse_transform(recent_predictions)
        recent_actuals = data['close'].iloc[-10:].values.reshape(-1, 1)
        prediction_errors = np.abs(recent_predictions - recent_actuals) / recent_actuals
        confidence = float(1 - np.mean(prediction_errors))
        return {
            'predictions': prediction.flatten().tolist(),
            'confidence': max(0.0, min(1.0, confidence)),  # Ensure confidence is between 0 and 1
            'timestamp': pd.Timestamp.utcnow().isoformat()
        }
