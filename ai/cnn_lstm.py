# Ajout de l'importation manquante pour TensorFlow dans cnn_lstm.py
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, LSTM, Dense
from tensorflow.keras.models import Sequential
class CNNLSTM:
    def __init__(self, input_shape):
        self.input_shape = input_shape
    def build_model(self):
        model = Sequential()
        model.add(Conv1D(64, 3, activation='relu', input_shape=self.input_shape))
        model.add(LSTM(64, activation='relu'))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model
