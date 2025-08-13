import tensorflow as tf
import optuna
from tensorflow.keras.optimizers import Adam
def objective(model, X_train, y_train, X_val, y_val, trial):
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    # Réinitialisation des poids pour chaque essai
    for layer in model.layers:
        if hasattr(layer, 'kernel_initializer'):
            layer.kernel.initializer.run()
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss='mse',
        metrics=['mae']
    )
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=2,  # Réduit pour les tests
        batch_size=32,
        verbose=0
    )
    return history.history['val_loss'][-1]
def run_optimization(model, X_train, y_train, X_val, y_val, n_trials=5):
    study = optuna.create_study(direction='minimize')
    study.optimize(
        lambda trial: objective(model, X_train, y_train, X_val, y_val, trial),
        n_trials=n_trials,
        catch=(tf.errors.NotFoundError,)
    )
    return study.best_params
