class HybridAI:
    """Classe de base pour l'IA de trading"""
    def __init__(self):
        self.learning_rate = 0.001
    def predict(self, input_data):
        """Fait une prédiction de base"""
        return 0.5
class HybridAIEnhanced(HybridAI):
    """Version améliorée avec traitement multi-timeframe"""
    def get_latency(self, test_data=None):
        """Mesure la latence avec données optionnelles"""
        import time
        test_data = test_data or [[0] * 6] * 100
        start = time.time()
        self.predict(test_data)
        return (time.time() - start) * 1000
