import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, LSTM, Dense, Add
class CNN_LSTM_Advanced:
    def __init__(self, input_shape):
        self.model = self._build_model(input_shape)
    def _build_model(self, input_shape):
        inputs = Input(shape=input_shape)
        # Bloc 1
        x = Conv1D(64, 3, activation='relu', padding='same')(inputs)
        x1 = Conv1D(64, 3, activation='relu', padding='same')(x)
        x = Add()([x, x1])  # Connexion résiduelle
        # ... 16 autres couches avec 4 blocs LSTM
        outputs = Dense(1, activation='sigmoid')(x)
        return tf.keras.Model(inputs, outputs)
