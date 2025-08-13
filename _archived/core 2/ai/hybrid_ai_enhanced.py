"""Module d'IA hybride amélioré"""
import numpy as np
from core.base_model import BaseModel
class HybridAIEnhanced:
    def __init__(self):
        self.models = {
            'primary': BaseModel(),
            'fallback': BaseModel()
        }
    def predict(self, X, model_type='primary'):
        """Prédiction avec fallback automatique"""
        try:
            model = self.models[model_type]
            return model.predict(X)
        except Exception as e:
            print(f"Prediction failed: {str(e)}, using fallback")
            return np.full((len(X), 1), 0.5)  # Valeur neutre
def predict(self, X, **kwargs):
    """Prédiction compatible avec fallback (ajout sécurisé)"""
    if not hasattr(self, '_fallback_model'):
        from core.base_model import BaseModel
        self._fallback_model = BaseModel().compile()
    try:
        return self._fallback_model.predict(X)
    except Exception as e:
        print(f"Erreur de prédiction: {str(e)}")
        import numpy as np
        return np.zeros((len(X), 1))  # Fallback
