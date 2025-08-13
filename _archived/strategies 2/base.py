class BaseStrategy:
    """Classe de base pour toutes les stratégies"""
    def __init__(self, config: dict):
        self.config = config
        self.exchange = None  # Exchange par défaut
    def initialize(self):
        """Initialisation optionnelle"""
        pass
    def cleanup(self):
        """Nettoyage des ressources"""
        pass
