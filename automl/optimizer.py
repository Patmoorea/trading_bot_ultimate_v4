import optuna
from tensorflow.keras import layers
class HyperparameterOptimizer:
    def __init__(self):
        self.study = optuna.create_study(direction='maximize')
    def optimize_cnn_lstm(self, trial):
        # Architecture optimisée pour M1/M4
        params = {
            'conv_layers': trial.suggest_int('conv_layers', 1, 3),
            'lstm_units': trial.suggest_int('lstm_units', 64, 256),
            'learning_rate': trial.suggest_float('lr', 1e-5, 1e-3, log=True)
        }
        return params
