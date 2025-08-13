from telegram import Bot
from .realtime_processing import RealTimeNewsMonitor
class NewsTelegramBridge(RealTimeNewsMonitor):
    """Interface entre le traitement des news et Telegram"""
    def __init__(self, bot_token):
        super().__init__()
        self.bot = Bot(token=bot_token)
        self.chat_id = None  # À configurer
    async def trigger_alert(self, news_item):
        """Envoie les alertes news vers Telegram"""
        message = self._format_alert(news_item)
        await self.bot.send_message(
            chat_id=self.chat_id,
            text=message,
            parse_mode='Markdown'
        )
        self.last_alert = news_item['id']
    def _format_alert(self, news):
        """Formate le message Telegram"""
        return f"""
        🚨 **ALERTE MARKET** 🚨
        *{news['title']}*
        Impact: {self.evaluate_impact(news)}
        Sentiment: {self.analyze_sentiment(news['content'])}
        """
