from sklearn.svm import SVC
import numpy as np
class QuantumSVM:
    def __init__(self):
        self.model = SVC(kernel="rbf", gamma=2, probability=True)
    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]  # Retourne les scores
