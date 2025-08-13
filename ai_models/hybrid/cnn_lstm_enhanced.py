import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, LSTM, Dense, Add, LayerNormalization
import numpy as np
class EnhancedCNNLSTM:
    def __init__(self, input_shape=(100, 5, 4)):
        self.input_shape = input_shape
        self.model = self._build_model()
    def _build_residual_block(self, x, filters):
        """Bloc résiduel avec normalisation"""
        shortcut = x
        x = Conv1D(filters, 3, padding='same')(x)
        x = LayerNormalization()(x)
        x = tf.nn.relu(x)
        x = Conv1D(filters, 3, padding='same')(x)
        x = LayerNormalization()(x)
        x = Add()([shortcut, x])
        return tf.nn.relu(x)
    def _build_model(self):
        """Architecture 18 couches avec connexions résiduelles"""
        inputs = Input(shape=self.input_shape)
        # Couche initiale
        x = Conv1D(128, 3, padding='same')(inputs)
        x = LayerNormalization()(x)
        # 18 couches avec connexions résiduelles
        for _ in range(6):
            x = self._build_residual_block(x, 128)
        # LSTM bidirectionnel
        x = tf.keras.layers.Bidirectional(LSTM(128, return_sequences=True))(x)
        x = tf.keras.layers.Bidirectional(LSTM(64))(x)
        # Couches de sortie
        outputs = Dense(3, activation='softmax')(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-4),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
