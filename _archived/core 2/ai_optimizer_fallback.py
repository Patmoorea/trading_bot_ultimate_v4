import tensorflow as tf
import optuna
import numpy as np
from tensorflow.keras.optimizers import Adam
def run_optimization(model, X_train, y_train, X_val, y_val, n_trials=5):
    """Version avec fallback CPU intégré"""
    def objective(trial):
        try:
            # Configuration pour chaque essai
            lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
            model.compile(
                optimizer=Adam(learning_rate=lr),
                loss='mse'
            )
            # Entraînement avec gestion d'erreur
            try:
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=2,
                    batch_size=32,
                    verbose=0
                )
                return history.history['val_loss'][-1]
            except BaseException:
                # Fallback CPU si échec GPU
                with tf.device('/CPU:0'):
                    history = model.fit(
                        X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=2,
                        batch_size=32,
                        verbose=0
                    )
                return history.history['val_loss'][-1]
        except Exception as e:
            print(f"Échec complet de l'essai: {str(e)}")
            return float('inf')
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    return study.best_params
