import tensorflow as tf
import optuna
from tensorflow.keras.optimizers import Adam
from src.core_merged.gpu_config import configure_gpu
def run_optimization(model, X_train, y_train, X_val, y_val, n_trials=5):
    """Version robuste de l'optimisation"""
    configure_gpu()  # Réapplique la configuration
    def objective(trial):
        try:
            lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
            model.compile(
                optimizer=Adam(learning_rate=lr),
                loss='mse',
                metrics=['mae']
            )
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=2,
                batch_size=32,
                verbose=0
            )
            return history.history['val_loss'][-1]
        except Exception as e:
            print(f"Erreur pendant l'essai: {str(e)}")
            return float('inf')  # Retourne une mauvaise valeur si échec
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    return study.best_params
