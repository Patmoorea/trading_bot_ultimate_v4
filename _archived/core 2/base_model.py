"""Module de base pour tous les modèles"""
from tensorflow.keras.models import Model
class BaseModel:
    def __init__(self):
        self._model = None
        self.optimizer = None
        self.loss = None
    def compile(self, optimizer='adam', loss='mse'):
        """Configure le modèle pour l'entraînement"""
        if self._model is None:
            self._build_default_model()
        self._model.compile(optimizer=optimizer, loss=loss)
    def _build_default_model(self):
        """Crée un modèle simple si aucun n'existe"""
        from tensorflow.keras.layers import Input, Dense
        inputs = Input(shape=(10,))
        outputs = Dense(1)(inputs)
        self._model = Model(inputs, outputs)
    def predict(self, X):
        """Fait des prédictions"""
        if self._model is None:
            self.compile()
        return self._model.predict(X)
def compile(self, optimizer='adam', loss='mse'):
    """Configure le modèle pour l'entraînement (ajout compatible)"""
    if not hasattr(self, '_is_compiled'):
        self._optimizer = optimizer
        self._loss = loss
        self._is_compiled = True
        print(f"Modèle compilé avec {optimizer}")
    return self
