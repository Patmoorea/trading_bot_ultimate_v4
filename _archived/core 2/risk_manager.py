class RiskManager:
    def __init__(self):
        print("⚡ Risk Manager Initialisé")
    def evaluate_risk(self, trade_signal):
        return {
            "approved": True,
            "risk_level": "low",
            "max_position": 0.1  # 10% du capital
        }
