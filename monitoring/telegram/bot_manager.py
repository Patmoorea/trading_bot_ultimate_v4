from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
import asyncio
from typing import Dict, Any
import json
from dataclasses import dataclass
@dataclass
class TelegramConfig:
    token: str
    chat_id: str
    alert_levels: Dict[str, float] = {
        "critical": 0.8,
        "important": 0.6,
        "info": 0.3
    }
class TelegramManager:
    def __init__(self, config: TelegramConfig):
        self.config = config
        self.app = Application.builder().token(config.token).build()
        self._setup_handlers()
    def _setup_handlers(self):
        """Configure les handlers de commandes"""
        self.app.add_handler(CommandHandler("status", self._cmd_status))
        self.app.add_handler(CommandHandler("position", self._cmd_position))
        self.app.add_handler(CommandHandler("performance", self._cmd_performance))
        self.app.add_handler(CommandHandler("alerts", self._cmd_alerts))
    async def send_alert(self, **kwargs):
        """Envoie une alerte formatée"""
        message = self._format_alert(kwargs)
        await self.app.bot.send_message(
            chat_id=self.config.chat_id,
            text=message,
            parse_mode='HTML'
        )
    def _format_alert(self, data: Dict) -> str:
        """Formate une alerte pour Telegram"""
        template = """
🚨 <b>{action} Signal</b> - {symbol}
📊 Timeframe: {timeframe}
🎯 Raison: {reason}
✅ Confiance: {confidence:.2%}
💰 Prix: {price}
📈 Stop Loss: {stop_loss}
📉 Take Profit: {take_profit}
⚡️ Risk/Reward: {risk_reward:.2f}
"""
        return template.format(**data)
    async def _cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Répond à la commande /status"""
        status = self._get_system_status()
        await update.message.reply_text(
            self._format_status(status),
            parse_mode='HTML'
        )
    async def _cmd_position(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Répond à la commande /position"""
        positions = self._get_current_positions()
        await update.message.reply_text(
            self._format_positions(positions),
            parse_mode='HTML'
        )
    def _format_status(self, status: Dict) -> str:
        """Formate le statut système"""
        template = """
🤖 <b>État du Système</b>
🖥️ CPU: {cpu_usage:.1f}%
💾 RAM: {ram_usage:.1f}GB
🌡️ GPU: {gpu_temp}°C
⚡️ Latence: {latency}ms
📊 Trades Aujourd'hui: {trades_today}
💰 P&L 24h: {daily_pnl:+.2f}%
"""
        return template.format(**status)
    def _format_positions(self, positions: List[Dict]) -> str:
        """Formate les positions actuelles"""
        if not positions:
            return "📭 Aucune position ouverte"
        template = """
📍 <b>Positions Actuelles</b>
"""
        for pos in positions:
            template += f"""
{pos['symbol']}:
└ Type: {pos['type']}
└ Taille: {pos['size']}
└ Prix d'entrée: {pos['entry_price']}
└ PnL: {pos['pnl']:+.2f}%
"""
        return template
