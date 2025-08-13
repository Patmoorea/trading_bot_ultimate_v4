# ========== QUANTUM ML INTEGRATION (SAFE UPDATE) ==========
from src.quantum.qsvm import QuantumTradingModel
class HybridEngine:
    def __init__(self):
        """Version hybride qui conserve l'ancien modèle"""
        self.classical_model = load_cnn_lstm()  # L'ancien modèle reste intact
        self.quantum_model = QuantumTradingModel()  # Nouveau composant
    def predict(self, X):
        # On garde l'ancienne logique et on ajoute le quantum
        classical_pred = self.classical_model.predict(X)
        quantum_pred = self.quantum_model.predict(X[:, :4])  # 4 features max
        # Combinaison pondérée (70% ancien, 30% quantum)
        return 0.7 * classical_pred + 0.3 * quantum_pred
# Compatibilité ascendante: l'ancien Engine reste disponible
class AIDecisionEngine:  # La classe originale n'est pas modifiée
    pass
