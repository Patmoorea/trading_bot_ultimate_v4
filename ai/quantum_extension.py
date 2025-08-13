"""
Module additionnel pour l'IA quantique
Interface avec le système existant sans modification
"""
from src.core_merged.ai.decision_engine import AIEngine  # Import du moteur existant
class QuantumAIExtension:
    def __init__(self, main_engine: AIEngine):
        self.main_engine = main_engine  # Composition au lieu d'héritage
    def enhance_prediction(self, data):
        """Améliore les prédictions existantes sans les remplacer"""
        # Appel aux fonctions originales
        base_pred = self.main_engine.predict(data)  
        # Ajout du traitement quantique
        quantum_boost = self._quantum_processing(data)
        return self._combine_results(base_pred, quantum_boost)
