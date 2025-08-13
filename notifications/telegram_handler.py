"""
Telegram Handler Module
"""
import asyncio
import logging
import re
from decimal import Decimal, InvalidOperation
from datetime import datetime, timezone
from typing import Dict, Any, List
logger = logging.getLogger(__name__)
class TelegramHandler:
        self.bot_token = bot_token
        self.signal_queue = asyncio.Queue(maxsize=queue_size)
        self._running = False
        self._rate_limits = {}
        self._rate_limit_delay = 0.01  # Réduit à 10ms
        self._queue_task = None
        self._batch_size = 10  # Traitement par lots
    def _validate_decimal(self, value: Any) -> bool:
        try:
            decimal_value = Decimal(str(value))
            return decimal_value > 0 and not decimal_value.is_nan() and not decimal_value.is_infinite()
        except (InvalidOperation, ValueError, TypeError):
            return False
    def _validate_pair(self, pair: str) -> bool:
        if not isinstance(pair, str):
            return False
        pattern = r'^[A-Z0-9]+/[A-Z0-9]+$'
        return bool(re.match(pattern, pair))
    def _validate_action(self, action: str) -> bool:
        return action in {'BUY', 'SELL'}
    def _validate_signal(self, signal: Dict[str, Any]) -> bool:
        try:
            required_fields = {'pair', 'action', 'price', 'amount'}
            if not all(field in signal for field in required_fields):
                return False
            if not self._validate_pair(signal['pair']):
                return False
            if not self._validate_action(signal['action']):
                return False
            if not self._validate_decimal(signal['price']):
                return False
            if not self._validate_decimal(signal['amount']):
                return False
            return True
        except Exception as e:
            logger.error(f"Error validating signal: {e}")
            return False
    async def start(self):
        if self._running:
            return
        self._running = True
        self._queue_task = asyncio.create_task(self._process_queue())
    async def stop(self):
        if not self._running:
            return
        self._running = False
        if self._queue_task:
            # Attendre que la queue soit vide
            try:
                await asyncio.wait_for(self.signal_queue.join(), timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("Queue processing timeout during stop")
            finally:
                if not self._queue_task.done():
                    self._queue_task.cancel()
                try:
                    await self._queue_task
                except asyncio.CancelledError:
                    pass
            self._queue_task = None
        now = asyncio.get_running_loop().time()
                return False
        return True
        if not self._running:
            return False
            return False
            return False
        if not self._validate_signal(signal):
            return False
        try:
            enriched_signal = {
                **signal,
            }
            await self.signal_queue.put(enriched_signal)
            return True
        except Exception as e:
            logger.error(f"Error sending signal: {e}")
            return False
    async def _process_queue(self):
        while self._running:
            try:
                # Traitement par lots pour plus d'efficacité
                batch = []
                for _ in range(self._batch_size):
                    if self.signal_queue.empty():
                        break
                    batch.append(await self.signal_queue.get())
                if batch:
                    # Simuler le traitement des signaux
                    await asyncio.sleep(0.01 * len(batch))
                    # Marquer les signaux comme traités
                    for _ in batch:
                        self.signal_queue.task_done()
                else:
                    await asyncio.sleep(0.01)
            except Exception as e:
                logger.error(f"Error processing queue: {e}")
                await asyncio.sleep(0.1)
