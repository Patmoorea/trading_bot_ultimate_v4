import asyncio
from ws_optimized import OptimizedWSClient
import signal
class GracefulExit:
    def __init__(self):
        self.shutdown = False
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)
    def exit_gracefully(self, signum, frame):
        self.shutdown = True
async def main():
    exit_handler = GracefulExit()
    client = OptimizedWSClient()
    try:
        await client.connect([
            'wss://stream.binance.com:9443/ws/btcusdt@kline_1m',
            'wss://stream.binance.com:9443/ws/ethusdt@kline_1m'
        ])
        while not exit_handler.shutdown:
            await asyncio.sleep(1)
    finally:
        print("Arrêt propre du client WebSocket")
if __name__ == '__main__':
    asyncio.run(main())
