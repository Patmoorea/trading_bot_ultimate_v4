import asyncio
import aiohttp
from typing import Optional
from src.config.settings import Settings
class TelegramNotifier:
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
    async def send_message(self, message: str) -> bool:
        async with aiohttp.ClientSession() as session:
            try:
                url = f"{self.base_url}/sendMessage"
                params = {
                    'chat_id': self.chat_id,
                    'text': message,
                    'parse_mode': 'HTML'
                }
                async with session.post(url, params=params) as response:
                    return response.status == 200
            except Exception as e:
                print(f"Erreur envoi Telegram: {str(e)}")
                return False
    def send_trade_alert(self, trade_info: dict):
        message = (
            f"🚨 <b>Alert Trading</b>\n"
            f"Symbol: {trade_info['symbol']}\n"
            f"Type: {trade_info['type']}\n"
            f"Prix: {trade_info['price']}\n"
            f"Volume: {trade_info['volume']}"
        )
        asyncio.create_task(self.send_message(message))
