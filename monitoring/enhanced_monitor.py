from typing import Dict, Any
import psutil
import numpy as np
from datetime import datetime
import telegram
import asyncio
class EnhancedMonitor:
    def __init__(self):
        self.metrics = {}
        self.alerts = []
        self.telegram_bot = None
        self.setup_telegram()
    def setup_telegram(self):
        """Configuration du bot Telegram"""
        self.telegram_bot = telegram.Bot(token='YOUR_BOT_TOKEN')
    async def send_alert(self, message: str, priority: str = "normal"):
        """Envoi d'alertes via Telegram"""
        if priority == "high":
            message = "🚨 " + message
        elif priority == "medium":
            message = "⚠️ " + message
        await self.telegram_bot.send_message(
            chat_id='YOUR_CHAT_ID',
            text=message
        )
    def monitor_system(self) -> Dict[str, float]:
        """Surveillance système"""
        return {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent,
            'network_io': psutil.net_io_counters()._asdict()
        }
    def monitor_trading(self, portfolio: Dict) -> Dict[str, Any]:
        """Surveillance trading"""
        return {
            'total_value': sum(portfolio.values()),
            'positions': len(portfolio),
            'top_holdings': sorted(portfolio.items(), key=lambda x: x[1])[-5:],
            'timestamp': datetime.utcnow().isoformat()
        }
    def monitor_performance(self, trades: list) -> Dict[str, float]:
        """Métriques de performance"""
        returns = [t['profit_loss'] for t in trades]
        return {
            'total_trades': len(trades),
            'win_rate': sum(1 for r in returns if r > 0) / len(returns),
            'avg_return': np.mean(returns),
            'sharpe': np.mean(returns) / np.std(returns) if len(returns) > 1 else 0,
            'max_drawdown': min(returns) if returns else 0
        }
    def check_alerts(self, metrics: Dict[str, Any]):
        """Vérification des alertes"""
        if metrics['cpu_percent'] > 90:
            asyncio.create_task(
                self.send_alert("CPU usage critique!", "high")
            )
        if metrics['memory_percent'] > 85:
            asyncio.create_task(
                self.send_alert("Mémoire critique!", "high")
            )
