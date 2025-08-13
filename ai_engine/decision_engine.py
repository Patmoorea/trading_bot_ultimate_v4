import numpy as np
from sklearn.linear_model import SGDClassifier
class DecisionEngine:
    def __init__(self):
        # Modèle simple mais efficace
        self.model = SGDClassifier(loss='log_loss')
        # Entraînement initial fictif
        X = np.random.rand(100, 2)  # close, volume
        y = np.random.randint(0, 3, 100)  # 0=hold, 1=buy, 2=sell
        self.model.fit(X, y)
    def make_decision(self, market_data):
        try:
            inputs = np.array([[
                market_data.get('close', 0),
                market_data.get('volume', 0)
            ]])
            pred = self.model.predict_proba(inputs)[0]
            actions = ['hold', 'buy', 'sell']
            idx = pred.argmax()
            return {
                'action': actions[idx],
                'confidence': float(pred.max())
            }
        except Exception as e:
            print(f"Decision error: {e}")
            return {'action': 'hold', 'confidence': 0.5}
if __name__ == '__main__':
    engine = DecisionEngine()
    print(engine.make_decision({'close': 50000, 'volume': 1000}))
