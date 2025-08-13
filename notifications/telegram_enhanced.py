from telegram import Bot
from typing import Dict, Optional
import logging
import asyncio
class EnhancedTelegramBot:
    def __init__(self, token: str, chat_id: str):
        self.bot = Bot(token)
        self.chat_id = chat_id
        self.alert_types = {
            "TRADE": "🤖",
            "RISK": "⚠️",
            "INFO": "ℹ️",
            "PROFIT": "💰",
            "LOSS": "🔻"
        }
    async def send_trade_alert(self, trade: dict):
        try:
            msg = f"""
{self.alert_types['TRADE']} Signal Trading
📊 Paire: {trade['symbol']} 
💎 Action: {trade['action']}
💰 Prix: {trade['price']}
🎯 Confiance: {trade['confidence']}%
🛑 Stop Loss: {trade['stop_loss']}
✨ Take Profit: {trade['take_profit']}
📈 Analyse Technique: {trade.get('technical_analysis', 'N/A')}
🌐 Sentiment Marché: {trade.get('market_sentiment', 'N/A')}
⚡ Volatilité: {trade.get('volatility', 'N/A')}
            """
            await self.bot.send_message(self.chat_id, msg)
        except Exception as e:
            logging.error(f"Erreur envoi Telegram: {str(e)}")
    async def send_market_update(self, data: Dict):
        msg = f"""
{self.alert_types['INFO']} Mise à jour Marché
📊 {data['symbol']}
📈 Tendance: {data['trend']}
💰 Volume 24h: {data['volume']}
🌊 Liquidité: {data['liquidity']}
        """
        await self.bot.send_message(self.chat_id, msg)
    async def send_risk_alert(self, alert: Dict):
        msg = f"""
{self.alert_types['RISK']} Alerte Risque
⚠️ Type: {alert['type']}
📊 Impact: {alert['impact']}
🔍 Détails: {alert['details']}
        """
        await self.bot.send_message(self.chat_id, msg)
    async def send_performance_update(self, perf: Dict):
        msg = f"""
{self.alert_types['PROFIT' if perf['daily_pnl'] > 0 else 'LOSS']} Bilan Performance
📈 PnL Quotidien: {perf['daily_pnl']}%
💼 Portfolio Total: {perf['total_balance']}
🎯 Trades Gagnants: {perf['win_rate']}%
        """
        await self.bot.send_message(self.chat_id, msg)
