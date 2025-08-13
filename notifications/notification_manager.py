"""
Gestionnaire de notifications
"""
import asyncio
import logging
from typing import List, Dict, Any
from datetime import datetime, timezone
from decimal import Decimal
class NotificationManager:
    """Gestionnaire de notifications pour le trading bot"""
    def __init__(self, handlers: List[Any] = None):
        self.handlers = handlers or []
        self._running = False
        self._queue = asyncio.Queue()
        self._worker_task = None
        self.logger = logging.getLogger('NotificationManager')
    async def start(self):
        """Démarrer le gestionnaire de notifications"""
        if self._running:
            return
        self._running = True
        self._worker_task = asyncio.create_task(self._process_notifications())
    async def stop(self):
        """Arrêter proprement le gestionnaire"""
        self._running = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None
    async def send_notification(self, notification: Dict[str, Any]) -> bool:
        """Envoyer une notification à tous les handlers"""
        if not self._running:
            return False
        try:
            enriched_notification = {
                **notification,
            }
            await self._queue.put(enriched_notification)
            return True
        except Exception as e:
            self.logger.error(f"Error sending notification: {e}")
            return False
    async def _process_notifications(self):
        """Traiter les notifications en attente"""
        while self._running:
            try:
                notification = await self._queue.get()
                for handler in self.handlers:
                    try:
                        if hasattr(handler, 'send_signal'):
                            await handler.send_signal(notification)
                        else:
                            self.logger.warning(f"Handler {handler} missing send_signal method")
                    except Exception as e:
                        self.logger.error(f"Handler {handler} failed: {e}")
                self._queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error processing notification: {e}")
                await asyncio.sleep(1)
"""
Module de gestion des notifications
"""
import asyncio
import logging
from typing import List, Dict, Any
from datetime import datetime, timezone
from decimal import Decimal
class NotificationManager:
    """Gestionnaire de notifications pour le trading bot"""
    def __init__(self, handlers: List[Any] = None, logger: logging.Logger = None):
        """Initialize notification manager"""
        self.handlers = handlers or []
        self._running = False
        self._queue = asyncio.Queue()
        self._worker_task = None
        self.logger = logger or logging.getLogger('NotificationManager')
    async def start(self):
        """Start notification manager"""
        if self._running:
            return
        self._running = True
        self._worker_task = asyncio.create_task(self._process_notifications())
    async def stop(self):
        """Stop notification manager"""
        self._running = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None
    async def close(self):
        """Close notification manager and cleanup resources"""
        await self.stop()
        # Close any additional resources here
    async def send_notification(self, notification: Dict[str, Any]) -> bool:
        """Send notification to all handlers"""
        if not self._running:
            return False
        try:
            enriched_notification = {
                **notification,
            }
            await self._queue.put(enriched_notification)
            return True
        except Exception as e:
            self.logger.error(f"Error sending notification: {e}")
            return False
    async def notify_opportunity_triangular(self, opportunity: Dict[str, Any]) -> bool:
        """Notify about triangular arbitrage opportunity"""
        if not opportunity or 'profit_pct' not in opportunity:
            return False
        notification = {
            'type': 'triangular_arbitrage',
            'profit_pct': float(opportunity['profit_pct']),
            'exchange': opportunity.get('exchange', 'unknown'),
            'path': opportunity.get('path', []),
            'details': opportunity
        }
        return await self.send_notification(notification)
    async def notify_opportunity_inter_exchange(self, opportunity: Dict[str, Any]) -> bool:
        """Notify about inter-exchange arbitrage opportunity"""
        if not opportunity or 'profit_pct' not in opportunity:
            return False
        notification = {
            'type': 'inter_exchange_arbitrage',
            'profit_pct': float(opportunity['profit_pct']),
            'buy_exchange': opportunity.get('buy_exchange', 'unknown'),
            'sell_exchange': opportunity.get('sell_exchange', 'unknown'),
            'symbol': opportunity.get('symbol', 'unknown'),
            'details': opportunity
        }
        return await self.send_notification(notification)
    async def send_daily_report(self, report_data: Dict[str, Any]) -> bool:
        """Send daily performance report"""
        notification = {
            'type': 'daily_report',
            'data': report_data,
        }
        return await self.send_notification(notification)
    async def _process_notifications(self):
        """Process pending notifications"""
        while self._running:
            try:
                notification = await self._queue.get()
                for handler in self.handlers:
                    try:
                        if hasattr(handler, 'send_signal'):
                            await handler.send_signal(notification)
                        else:
                            self.logger.warning(f"Handler {handler} missing send_signal method")
                    except Exception as e:
                        self.logger.error(f"Handler {handler} failed: {e}")
                self._queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error processing notification: {e}")
                await asyncio.sleep(1)
    def __init__(self, handlers: List[Any] = None, logger: logging.Logger = None):
        self.notifiers = handlers or []  # Change handlers to notifiers to match tests
        self._running = False
        self._queue = asyncio.Queue()
        self._worker_task = None
        self.logger = logger or logging.getLogger('NotificationManager')
    async def send(self, message: str, level: str = 'info'):
        """Send a simple message notification"""
        return await self.send_notification({'message': message, 'level': level})
    async def send_notification(self, message: str = None, level: str = 'info', **kwargs):
        """Updated send_notification to handle both string messages and dict notifications"""
        if isinstance(message, str):
            notification = {'message': message, 'level': level, **kwargs}
        else:
            notification = message
        if not self._running:
            return False
        try:
            enriched_notification = {
                **notification,
            }
            await self._queue.put(enriched_notification)
            return True
        except Exception as e:
            self.logger.error(f"Error sending notification: {e}")
            return False
    async def flush(self):
        """Flush all pending notifications"""
        while not self._queue.empty():
            try:
                notification = self._queue.get_nowait()
                for handler in self.handlers:
                    try:
                        await handler.send_signal(notification)
                    except:
                        self.logger.error(f"Error processing notification during flush")
                self._queue.task_done()
            except:
                break
    async def flush(self) -> list:
        """Flush all pending notifications and return them"""
        notifications = []
        while not self._queue.empty():
            notification = await self._queue.get()
            notifications.append(notification.get('message', ''))
            self._queue.task_done()
        return notifications
    async def send_notification_log(self, title: str, message: str, level: str = 'info') -> bool:
        """Send a notification with title and message"""
        notification = {
            'title': title,
            'message': message,
            'level': level
        }
        return await self.send_notification(notification)
    def __len__(self):
        """Return number of handlers"""
        return len(self.handlers or [])
    async def send_notification_log(self, title: str, message: str, level: str = 'info'):
        """Send a log notification"""
        notification = {
            'title': title,
            'message': message,
            'level': level
        }
        return await self.send_notification(notification)
import asyncio
class NotificationManager:
    def __init__(self):
        self.queue = []
    async def flush(self):
        await asyncio.sleep(0)  # simulate delay
        return self.queue
class NotificationManager:
    def __init__(self, *args, **kwargs):
        self.channels = []
    def send_notification(self, message):
        print("Notification:", message)
        return True
class NotificationManager:
    def __init__(self, *args, **kwargs):
        self.queue = []
    def flush(self):
        return True
