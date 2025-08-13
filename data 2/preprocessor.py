import numpy as np
class DataPreprocessor:
    def prepare_ai_input(self, df):
        """Convertit les données pour le modèle IA"""
        return np.array([df['close'].values])
