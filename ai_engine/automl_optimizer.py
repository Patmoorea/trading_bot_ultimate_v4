import optuna
from optuna.integration import TFKerasPruningCallback
import tensorflow as tf
from .cnn_lstm_model import create_cnn_lstm_model
class HyperparameterOptimizer:
    def __init__(self, X_train, y_train, X_val, y_val, n_trials=200):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.n_trials = n_trials
    def optimize(self):
        study = optuna.create_study(
            direction='maximize',
            pruner=optuna.pruners.HyperbandPruner()
        )
        study.optimize(self._objective, n_trials=self.n_trials)
        return study.best_params
    def _objective(self, trial):
        # Hyperparameters to optimize
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
            'conv1_filters': trial.suggest_categorical('conv1_filters', [32, 64, 128]),
            'conv2_filters': trial.suggest_categorical('conv2_filters', [64, 128, 256]),
            'lstm_units': trial.suggest_categorical('lstm_units', [128, 256, 512]),
            'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
            'l2_reg': trial.suggest_float('l2_reg', 1e-6, 1e-3, log=True)
        }
        # Model creation
        model = create_cnn_lstm_model()
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=params['learning_rate']),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        # Training with pruning
        history = model.fit(
            self.X_train,
            self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=50,
            batch_size=256,
            callbacks=[TFKerasPruningCallback(trial, 'val_accuracy')],
            verbose=0
        )
        return history.history['val_accuracy'][-1]
