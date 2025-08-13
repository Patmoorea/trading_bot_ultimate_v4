from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Dict, Optional
class BaseExchange(ABC):
    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
    @abstractmethod
    def get_ticker(self, symbol: str) -> Dict:
        """Get ticker information for a symbol"""
        pass
    @abstractmethod
    def get_balance(self) -> Dict:
        """Get account balance"""
        pass
    @abstractmethod
    def place_order(self, symbol: str, side: str, amount: Decimal, 
                   price: Optional[Decimal] = None) -> Dict:
        """Place a new order"""
        pass
