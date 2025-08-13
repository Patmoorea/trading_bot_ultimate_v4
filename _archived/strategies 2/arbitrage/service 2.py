import os
import asyncio
from decimal import Decimal, getcontext
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
            if exchange == 'binance' or exchange == 'blofin':
                return await self.exchanges[exchange].get_order_book(symbol)
            else:
                orderbook = await self.exchanges[exchange].fetch_order_book(symbol)
                bid_price = Decimal(str(orderbook['bids'][0][0])) if len(orderbook['bids']) > 0 else Decimal(0)
                ask_price = Decimal(str(orderbook['asks'][0][0])) if len(orderbook['asks']) > 0 else Decimal('Infinity')
                return bid_price, ask_price
        except Exception as e:
            logger.warning(f"Échec carnet {exchange} {symbol}: {str(e)}")
            return Decimal(0), Decimal('Infinity')
class ArbitrageEngine:
    def __init__(self):
        self.exchange_manager = ExchangeManager()
    async def scan_opportunities(self) -> List[ArbitrageOpportunity]:
        opportunities = []
        for symbol in PAIRS:
            prices = {}
            valid_exchanges = []
            # Récupération des prix
            for exchange, pair in PAIRS[symbol].items():
                bid, ask = await self.exchange_manager.get_order_book(exchange, pair)
                if bid > 0 and ask < Decimal('Infinity'):
                    prices[exchange] = (bid, ask)
                    valid_exchanges.append(exchange)
            if len(valid_exchanges) < 2:
                continue
            # Calcul des meilleurs prix
            best_bid = max(((ex, data[0]) for ex, data in prices.items()), key=lambda x: x[1])
            best_ask = min(((ex, data[1]) for ex, data in prices.items()), key=lambda x: x[1])
            if best_bid[0] != best_ask[0]:
                # Calcul du profit après frais
                sell_fee = FEES[best_bid[0]]['taker']
                buy_fee = FEES[best_ask[0]]['taker']
                effective_bid = best_bid[1] * (Decimal(1) - sell_fee)
                effective_ask = best_ask[1] * (Decimal(1) + buy_fee)
                profit = (effective_bid - effective_ask) / effective_ask
                if profit >= SETTINGS['profit_threshold']:
                    volume = min(SETTINGS['max_order_value'] / effective_ask, 
                                Decimal('0.1'))  # Limite à 0.1 BTC/ETH pour les tests
                    opportunities.append(
                        ArbitrageOpportunity(
                            symbol=symbol,
                            buy_exchange=best_ask[0],
                            sell_exchange=best_bid[0],
                            buy_price=best_ask[1],
                            sell_price=best_bid[1],
                            volume=volume,
                            profit=profit,
                            timestamp=time.time()
                        )
                    )
        return opportunities
