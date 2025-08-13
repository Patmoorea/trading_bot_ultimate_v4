# src/notifications/telegram_bot.py VERSION 1.0.0
import os
import logging
import asyncio
from typing import List, Dict, Any, Optional
from decimal import Decimal, InvalidOperation
import aiohttp
from dotenv import load_dotenv
load_dotenv()
logger = logging.getLogger(__name__)
class TelegramBot:
    def __init__(self, queue_size: int = 100):
        self.token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = os.getenv('TELEGRAM_CHAT_ID')
        self.enabled = bool(self.token and self.chat_id)
        self.signal_queue = asyncio.Queue(maxsize=queue_size)
        self._running = False
        self._rate_limits = {}
        self._rate_limit_delay = 0.01
        self._queue_task = None
        self._batch_size = 10
        self.session = None
        if not self.enabled:
            logger.warning("⚠️ Configuration Telegram incomplète")
    async def _init_session(self):
        if not self.session:
            self.session = aiohttp.ClientSession()
    async def _close_session(self):
        if self.session:
            await self.session.close()
            self.session = None
    async def send_message(self, 
                          message: str, 
                          parse_mode: str = 'HTML',
                          silent: bool = False) -> bool:
        if not self.enabled:
            return False
        try:
            await self._init_session()
            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            data = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": parse_mode,
                "disable_notification": silent
            }
            async with self.session.post(url, json=data) as response:
                success = response.status == 200
                if success:
                    logger.debug(f"Message envoyé: {message[:50]}...")
                else:
                    logger.error(f"Erreur Telegram {response.status}: {await response.text()}")
                return success
        except Exception as e:
            logger.error(f"Erreur envoi Telegram: {str(e)}")
            return False
    async def send_trade_alert(self,
                             symbol: str,
                             action: str,
                             price: float,
                             volume: float,
                             reason: Optional[str] = None,
                             confidence: Optional[float] = None) -> bool:
        """
        Envoie une alerte de trading enrichie
        """
        message = (
            f"🚨 <b>Alert Trading</b>\n"
            f"Symbol: {symbol}\n"
            f"Action: {action}\n"
            f"Prix: {price}\n"
            f"Volume: {volume}"
        )
        if reason:
            message += f"\nRaison: {reason}"
        if confidence:
            message += f"\nConfiance: {confidence:.2%}"
        return await self.send_message(message)
    def _validate_decimal(self, value: Any) -> bool:
        try:
            decimal_value = Decimal(str(value))
            return decimal_value > 0 and not decimal_value.is_nan() and not decimal_value.is_infinite()
        except (InvalidOperation, ValueError, TypeError):
            return False
    async def start(self):
        """Démarre le processeur de queue"""
        if self._running:
            return
        self._running = True
        self._queue_task = asyncio.create_task(self._process_queue())
    async def stop(self):
        """Arrête le processeur de queue"""
        self._running = False
        if self._queue_task:
            await self._queue_task
            self._queue_task = None
        await self._close_session()
    async def _process_queue(self):
        """Traite les messages en attente"""
        while self._running:
            try:
                for _ in range(min(self._batch_size, self.signal_queue.qsize())):
                    message = await self.signal_queue.get()
                    await self.send_message(message)
                    self.signal_queue.task_done()
                await asyncio.sleep(self._rate_limit_delay)
            except Exception as e:
                logger.error(f"Erreur traitement queue: {str(e)}")
                await asyncio.sleep(1)
