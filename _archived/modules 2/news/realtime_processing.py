import asyncio
from datetime import datetime
from . import EnhancedNewsProcessor
class RealTimeNewsMonitor(EnhancedNewsProcessor):
    """Extension pour le traitement temps réel"""
    POLL_INTERVAL = 60  # secondes
    def __init__(self):
        super().__init__()
        self.active_alerts = set()
    async def monitor_breaking_news(self):
        """Surveille les news en continu"""
        while True:
            try:
                await self._monitoring_cycle()
            except Exception as e:
                print(f"Monitoring error: {e}")
                await asyncio.sleep(10)
    async def _monitoring_cycle(self):
        news = self.fetch_latest()
        critical = self.filter_critical_news(news)
        for item in critical:
            if item['id'] not in self.active_alerts:
                await self.process_new_alert(item)
        await asyncio.sleep(self.POLL_INTERVAL)
    async def process_new_alert(self, news_item):
        """Traite une nouvelle alerte"""
        self.active_alerts.add(news_item['id'])
        if hasattr(self, 'trigger_alert'):
            await self.trigger_alert(news_item)
