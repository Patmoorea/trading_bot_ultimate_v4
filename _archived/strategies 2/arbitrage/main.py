import asyncio
import time
from decimal import Decimal
from utils.logger import get_logger
from .service import ArbitrageEngine
from .config import SETTINGS
logger = get_logger()
class ArbitrageBot:
    def __init__(self):
        self.engine = ArbitrageEngine()
        self.total_profit = Decimal(0)
    async def run(self):
        logger.info("Starting arbitrage bot...")
        try:
            while True:
                start_time = time.time()
                try:
                    opportunities = await self.engine.scan_opportunities()
                    if opportunities:
                        for opp in sorted(opportunities, key=lambda x: x.profit, reverse=True):
                            logger.info(
                                f"Opportunity: {opp.symbol} "
                                f"Buy@{opp.buy_exchange} {opp.buy_price:.8f} "
                                f"Sell@{opp.sell_exchange} {opp.sell_price:.8f} "
                                f"Profit: {opp.profit:.2%}"
                            )
                    else:
                        logger.info("No arbitrage opportunities found")
                except Exception as e:
                    logger.error(f"Error scanning opportunities: {str(e)}")
                elapsed = time.time() - start_time
                await asyncio.sleep(max(0, SETTINGS['price_expiry'] - elapsed))
        except asyncio.CancelledError:
            logger.info("Stopping bot...")
        finally:
            await self.engine.close()
            logger.info("Bot stopped successfully")
if __name__ == "__main__":
    bot = ArbitrageBot()
    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
