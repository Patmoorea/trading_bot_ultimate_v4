from datetime import datetime
from typing import Dict
import asyncio
from src.news_processor.core import CachedNewsSentimentAnalyzer
from src.notifications.telegram_bot import TelegramBot
class NewsAlertSystem:
    def __init__(self):
        self.news_analyzer = CachedNewsSentimentAnalyzer()
        self.telegram_bot = TelegramBot()
        self.last_alert = {}
    async def process_and_alert(self, news_data: Dict):
        """Traite les news et envoie des alertes si nécessaire"""
        try:
            # Analyse du sentiment
            analysis = await self.news_analyzer.analyze([news_data['text']])
            if not analysis:
                return
            impact = analysis[0]
            # Vérification du seuil d'alerte
            if impact['confidence'] >= 0.7:
                # Évite les doublons en moins de 5 minutes
                news_id = news_data.get('id', '')
                if news_id in self.last_alert:
                    time_diff = datetime.utcnow() - self.last_alert[news_id]
                    if time_diff.total_seconds() < 300:
                        return
                # Création du message d'alerte
                message = self._format_alert_message(news_data, impact)
                # Envoi via Telegram
                await self.telegram_bot.send_message(message)
                # Mise à jour du timestamp
                self.last_alert[news_id] = datetime.utcnow()
        except Exception as e:
            print(f"❌ Erreur traitement alerte: {e}")
    def _format_alert_message(self, news_data: Dict, impact: Dict) -> str:
        """Formate le message d'alerte"""
        return f"""
📰 *Alerte News Importante*
{'='*30}
📑 Titre: {news_data['title']}
📊 Impact: {'🟢' if impact['sentiment'] == 'bullish' else '🔴'} {impact['confidence']:.2%}
🔍 Source: {news_data['source']}
{'='*30}
"""
