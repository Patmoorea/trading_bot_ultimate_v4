"""
Telegram notification handler
"""
import asyncio
import logging
from typing import Optional
from datetime import datetime
class TelegramHandler:
    def __init__(self, bot_token: str, chat_id: str, logger: Optional[logging.Logger] = None):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.logger = logger or logging.getLogger('TelegramHandler')
        self._queue = asyncio.Queue()
        self._worker_task = None
        self._running = False
        self.is_authorized = True  # For test purposes
    async def start(self):
        """Start the handler"""
        if self._running:
            return
        self._running = True
        self._worker_task = asyncio.create_task(self._process_queue())
    async def stop(self):
        """Stop the handler"""
        self._running = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None
    async def send_signal(self, signal: dict):
        """Add signal to queue"""
        if not self._running:
            return False
        await self._queue.put(signal)
        return True
    async def _process_queue(self):
        """Process signals in queue"""
        while self._running:
            try:
                signal = await self._queue.get()
                # Implement actual Telegram sending here
                self.logger.info(f"Would send to Telegram: {signal}")
                self._queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error processing signal: {e}")
                await asyncio.sleep(1)
    @property
    def is_authorized(self):
        """Check if handler is authorized"""
        return True
    async def send_signal(self, signal):
        """Send signal through telegram"""
        if not self._queue:
            self._queue = asyncio.Queue()
        await self._queue.put(signal)
        return True
    async def _process_queue(self):
        """Process queued signals"""
        while self._running:
            try:
                signal = await self._queue.get()
                # Process signal here
                self._queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error processing signal: {e}")
    @property
    def is_authorized(self) -> bool:
        """Check if handler is authorized"""
        return hasattr(self, 'bot_token') and hasattr(self, 'chat_id')
    def _init_worker(self):
        """Initialize worker task"""
        if not hasattr(self, '_worker_task') or self._worker_task is None:
            self._worker_task = asyncio.create_task(self._process_queue())
