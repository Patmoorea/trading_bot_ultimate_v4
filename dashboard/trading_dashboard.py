from datetime import datetime
class TradingMetrics:
    def __init__(self):
        self.daily_pnl = 0.0
        self.total_pnl = 0.0
        self.win_rate = 0.0
        self.drawdown = 0.0
        self.sharpe_ratio = 0.0
        self.trades_count = 0
        self.active_positions = 0
class NotificationManager:
    def __init__(self):
        self.notifications = []
    def add_alert(self, message, level="info", expiry=None):
        self.notifications.append({
            'message': message,
            'level': level,
            'expiry': expiry,
            'timestamp': datetime.utcnow()
        })
        return len(self.notifications)
    def get_active_alerts(self):
        current_time = datetime.utcnow()
        return [n for n in self.notifications if n['expiry'] > current_time]
class EnhancedTradingDashboard:
    def __init__(self):
        self.metrics = TradingMetrics()
        self.notifications = NotificationManager()
    def update_metrics(self, metrics_dict):
        for key, value in metrics_dict.items():
            setattr(self.metrics, key, value)
        return self.metrics
