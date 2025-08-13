from typing import Dict
from .telegram_notifications import TelegramNotifier
def send_arbitrage_alert(opportunity: Dict) -> bool:
    """Envoie une alerte réelle via Telegram"""
    notifier = TelegramNotifier()
    message = (
        f"🚨 Opportunité d'arbitrage détectée\n"
        f"• Exchange: `{opportunity.get('exchange', 'N/A')}`\n"
        f"• Spread: `{opportunity.get('spread', 0):.2f}%`\n"
        f"• Bid: `{opportunity.get('bid', 0)}`\n"
        f"• Ask: `{opportunity.get('ask', 0)}`"
    )
    return notifier.send(message)
