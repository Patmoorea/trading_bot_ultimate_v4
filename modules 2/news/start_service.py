import asyncio
from .telegram_integration import NewsTelegramBridge
def start_news_monitoring(config):
    """Lance le service complet"""
    service = NewsTelegramBridge(
        bot_token=config['TELEGRAM_TOKEN'],
        chat_id=config['TELEGRAM_CHAT_ID']
    )
    loop = asyncio.get_event_loop()
    loop.create_task(service.monitor_breaking_news())
    return service
