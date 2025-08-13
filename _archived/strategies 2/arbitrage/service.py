import os
import asyncio
from decimal import Decimal, getcontext  # Ajout de l'import Decimal
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from dotenv import load_dotenv
from ccxt.async_support import gateio, bingx, okx
from connectors.binance import BinanceConnector
from connectors.blofin import BlofinConnector
from utils.logger import get_logger
from .config import PAIRS, SETTINGS, FEES
load_dotenv()
getcontext().prec = 8
logger = get_logger()
@dataclass
class ArbitrageOpportunity:
    symbol: str
    buy_exchange: str
    sell_exchange: str
    buy_price: Decimal
    sell_price: Decimal
    volume: Decimal
    profit: Decimal
    timestamp: float
class ExchangeManager:
    def __init__(self):
        self.exchanges = {
            'binance': BinanceConnector(),
            'gateio': gateio({
                'apiKey': os.getenv('GATEIO_API_KEY'),
                'secret': os.getenv('GATEIO_API_SECRET'),
                'options': {'defaultType': 'spot'},
                'enableRateLimit': True
            }),
            'bingx': bingx({
                'apiKey': os.getenv('BINGX_API_KEY'),
                'secret': os.getenv('BINGX_API_SECRET'),
                'enableRateLimit': True
            }),
            'okx': okx({
                'apiKey': os.getenv('OKX_API_KEY'),
                'secret': os.getenv('OKX_API_SECRET'),
                'password': os.getenv('OKX_PASSPHRASE'),
                'enableRateLimit': True
            }),
            'blofin': BlofinConnector()
        }
    async def get_order_book(self, exchange: str, symbol: str) -> Tuple[Decimal, Decimal]:
        try:
            if exchange in ['binance', 'blofin']:
                return await self.exchanges[exchange].get_order_book(symbol)
            orderbook = await self.exchanges[exchange].fetch_order_book(symbol)
            bid = Decimal(str(orderbook['bids'][0][0])) if len(orderbook['bids']) > 0 else Decimal(0)
            ask = Decimal(str(orderbook['asks'][0][0])) if len(orderbook['asks']) > 0 else Decimal('Infinity')
            return bid, ask
        except Exception as e:
            logger.error(f"Failed to fetch {symbol} from {exchange}: {str(e)}")
            return Decimal(0), Decimal('Infinity')
class ArbitrageEngine:
    def __init__(self):
        self.exchange_manager = ExchangeManager()
        self.active_orders = {}
    async def scan_opportunities(self) -> List[ArbitrageOpportunity]:
        opportunities = []
        for symbol in PAIRS:
            prices = {}
            for exchange, pair in PAIRS[symbol].items():
                bid, ask = await self.exchange_manager.get_order_book(exchange, pair)
                if bid > 0 and ask < Decimal('Infinity'):
                    prices[exchange] = (bid, ask)
                    logger.debug(f"{symbol} prices on {exchange}: bid={bid:.8f}, ask={ask:.8f}")
            if len(prices) < 2:
                continue
            best_bid_exchange, (best_bid, _) = max(prices.items(), key=lambda x: x[1][0])
            best_ask_exchange, (_, best_ask) = min(prices.items(), key=lambda x: x[1][1])
            if best_bid_exchange != best_ask_exchange:
                effective_bid = best_bid * (1 - FEES[best_bid_exchange]['taker'])
                effective_ask = best_ask * (1 + FEES[best_ask_exchange]['taker'])
                profit = (effective_bid - effective_ask) / effective_ask
                if profit >= SETTINGS['profit_threshold']:
                    volume = min(
                        SETTINGS['max_order_value'] / best_ask,
                        Decimal('1')  # Max 1 unit of asset for safety
                    )
                    opportunities.append(
                        ArbitrageOpportunity(
                            symbol=symbol,
                            buy_exchange=best_ask_exchange,
                            sell_exchange=best_bid_exchange,
                            buy_price=best_ask,
                            sell_price=best_bid,
                            volume=volume,
                            profit=profit,
                            timestamp=time.time()
                        )
                    )
        return opportunities
    async def close(self):
        """Close all exchange connections"""
        await self.exchange_manager.close()
