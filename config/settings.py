from datetime import datetime, timezone
class Settings:
    # Informations utilisateur
    # Modes et environnements
    TRADING_MODE = "LIVE"
    LOG_LEVEL = "DEBUG"
    # Paramètres trading
    MIN_PROFIT = 0.005  # 0.5%
    MAX_SLIPPAGE = 0.001
    RISK_PERCENTAGE = 0.02  # 2% par trade
    # Paramètres de sécurité
    MAX_TRADES_PER_HOUR = 10
    MAX_POSITION_SIZE = 0.1
    STOP_LOSS_DEFAULT = 0.02
    @classmethod
    def is_trading_allowed(cls):
        return cls.TRADING_MODE == "LIVE"
