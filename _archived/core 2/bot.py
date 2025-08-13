class TradingBot:
    """Classe de base du trading bot"""
    def __init__(self):
        self.running = False
    async def fetch_data(self, symbol, timeframe):
        """Méthode de base pour récupérer les données"""
        # Implémentation factice pour les tests
        import pandas as pd
        return pd.DataFrame({
            'open': [1, 2, 3],
            'high': [1, 2, 3],
            'low': [1, 2, 3],
            'close': [1, 2, 3],
            'volume': [100, 200, 300]
        })
    def run(self):
        """Méthode principale d'exécution"""
        self.running = True
        return True
