"""
Notifications Telegram complètes avec alertes de trading
"""
import requests
from config import Config
class TelegramNotifier:
    def __init__(self):
        self.base_url = f"https://api.telegram.org/bot{Config.TELEGRAM_BOT_TOKEN}"
    def send_message(self, text):
        if not Config.TELEGRAM_BOT_TOKEN:
            return
        try:
            requests.post(
                f"{self.base_url}/sendMessage",
                json={
                    "chat_id": Config.TELEGRAM_CHAT_ID,
                    "text": text,
                    "parse_mode": "Markdown"
                }
            )
        except Exception as e:
            print(f"Erreur Telegram: {str(e)}")
    def send_trade_alert(self, symbol, action, price, reason):
        emoji = "🟢" if action.lower() == "buy" else "🔴"
        message = (
            f"{emoji} *Trade Executed*\n"
            f"• Pair: `{symbol}`\n"
            f"• Action: `{action.upper()}`\n"
            f"• Price: `{price:.2f}`\n"
            f"• Reason: `{reason}`"
        )
        self.send_message(message)
    def send_news_alert(self, sentiment, top_news):
        emoji = "📈" if sentiment > 0 else "📉"
        message = (
            f"{emoji} *Market Sentiment Update*\n"
            f"• Score: `{sentiment:.2f}`\n"
            f"• Top News: `{top_news[:50]}...`"
        )
        self.send_message(message)
def send_trade_alert(symbol, action, confidence, news_impact=None):
    """Version améliorée avec intégration news"""
    message = f"📈 {symbol} {action.upper()}\n"
    message += f"Confiance: {confidence:.2f}\n"
    if news_impact:
        message += f"Impact News: {news_impact}/5\n"
    bot.send_message(config.TELEGRAM_CHAT_ID, message)
