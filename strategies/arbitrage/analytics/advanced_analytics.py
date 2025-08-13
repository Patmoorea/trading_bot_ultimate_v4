class ArbitrageAnalytics:
    """Nouvelle fonctionnalité : Analyse historique des spreads"""
    def __init__(self):
        from datetime import datetime
        import pandas as pd
        self.data = pd.DataFrame(columns=['timestamp', 'spread'])
    def record_spread(self, spread):
        """Enregistre les spreads pour analyse"""
        self.data.loc[len(self.data)] = new_data
    def get_stats(self):
        """Retourne les statistiques sur 24h"""
        return {
            'max': self.data['spread'].max(),
            'min': self.data['spread'].min(),
            'mean': self.data['spread'].mean()
        }
