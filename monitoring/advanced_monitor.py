import matplotlib.pyplot as plt
class ArbitrageVisualizer:
    """Visualisation des opportunités sans modifier le code existant"""
    def __init__(self, arbitrage_module):
        self.arbitrage = arbitrage_module
    def plot_spread_history(self, pair: str, history: list):
        """Affiche l'historique des spreads"""
        plt.figure(figsize=(12, 6))
        plt.plot(history)
        plt.title(f"Spread historique pour {pair}")
        plt.ylabel("Spread (%)")
        plt.xlabel("Temps")
        plt.grid(True)
        plt.show()
