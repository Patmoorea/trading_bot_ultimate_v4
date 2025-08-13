from src.ai.cnn_lstm import CNNLSTM
from tensorflow.keras.layers import Dropout
class EnhancedCNNLSTM(CNNLSTM):
    def __init__(self, input_shape):
        super().__init__(input_shape)
    def enhanced_build_model(self, metrics=None):
        model = self.build_model()
        model.add(Dropout(0.2))
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=metrics)
        return model
