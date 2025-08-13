from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense
class CNNLSTM:
    def __init__(self, input_shape):
        self.input_shape = input_shape
    def build_model(self):
        model = Sequential()
        model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=self.input_shape))
        model.add(LSTM(64, return_sequences=False))
        model.add(Dense(1, activation='sigmoid'))
        return model
