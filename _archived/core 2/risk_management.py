class RiskManager:
    """Gestionnaire de risque complet"""
    def __init__(self, max_drawdown=0.05):
        self.max_drawdown = max_drawdown
    def calculate_position_size(self, capital, risk_percent):
        """Calcule la taille de position"""
        return capital * risk_percent
    def calculate_stop_loss(self, entry_price, risk_percent):
        """Calcule le stop-loss"""
        return entry_price * (1 - risk_percent)
    def validate_risk_parameters(self, risk):
        """Valide les paramètres de risque"""
        return 0 < risk <= self.max_drawdown
class CircuitBreakerExtension:
    """Extension pour la gestion des circuit breakers"""
    def __init__(self, threshold=0.1):
        self.threshold = threshold
        self.triggered = False
    def check_market_condition(self, market_data):
        """Vérifie si les conditions pour un circuit breaker sont remplies"""
        if market_data.get('drop_percent', 0) > self.threshold:
            self.triggered = True
        return self.triggered
    def reset(self):
        """Réinitialise le circuit breaker"""
        self.triggered = False
class ProfessionalRiskManager:
    PARAMS = {
        'max_drawdown': 0.05,
        'daily_stop_loss': 0.02,
        'position_sizing': 'volatility_based',
        'circuit_breaker': {
            'market_crash': True,
            'liquidity_shock': True,
            'black_swan': True
        }
    }
    def check_circuit_breakers(self, market_data):
        """Vérifie les conditions de circuit breaker"""
        if self._detect_market_crash(market_data):
            self.trigger_emergency_protocol()
