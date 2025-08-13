from typing import Dict, List, Optional, Tuple
import numpy as np
from datetime import datetime
import logging
import hmac
import hashlib
import time
import random
from decimal import Decimal
class AntiFrontrunning:
    def __init__(self,
                 min_delay_ms: int = 20,
                 max_delay_ms: int = 200,
                 chunk_size: float = 0.1,
                 price_noise: float = 0.0001):
        self.min_delay = min_delay_ms
        self.max_delay = max_delay_ms
        self.chunk_size = chunk_size
        self.price_noise = price_noise
        self.logger = logging.getLogger(__name__)
        self._initialize_secure_random()
    def _initialize_secure_random(self) -> None:
        """Initialize cryptographically secure random number generator"""
        seed = int.from_bytes(random.randbytes(8), byteorder='big')
        self.random = random.Random(seed)
    def _generate_noise(self) -> float:
        """Generate random price noise"""
        return self.random.uniform(-self.price_noise, self.price_noise)
    def _calculate_chunks(self, 
                         total_size: float,
                         min_chunk: float) -> List[float]:
        """Calculate order chunks with randomization"""
        num_chunks = int(total_size / (total_size * self.chunk_size))
        base_size = total_size / num_chunks
        chunks = []
        remaining = total_size
        while remaining > min_chunk:
            # Add randomness to chunk size
            size = min(
                remaining,
                base_size * (1 + self.random.uniform(-0.1, 0.1))
            )
            size = max(min_chunk, size)
            chunks.append(size)
            remaining -= size
        if remaining > 0:
            chunks[-1] += remaining
        return chunks
    def _get_delay(self) -> int:
        """Get random delay between min and max"""
        return self.random.randint(self.min_delay, self.max_delay)
    def _sign_order(self, 
                    order_params: Dict,
                    secret_key: str) -> str:
        """Create order signature to prevent tampering"""
        message = '&'.join([f"{k}={v}" for k, v in sorted(order_params.items())])
        signature = hmac.new(
            secret_key.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()
        return signature
    def protect_order(self,
                     symbol: str,
                     side: str,
                     quantity: float,
                     price: float,
                     min_quantity: float,
                     secret_key: str) -> List[Dict]:
        """
        Protect order from frontrunning by:
        1. Splitting into random sized chunks
        2. Adding random delays
        3. Adding price noise
        4. Signing orders
        """
        protected_orders = []
        chunks = self._calculate_chunks(quantity, min_quantity)
        base_timestamp = int(time.time() * 1000)
        for i, chunk_size in enumerate(chunks):
            # Add noise to price
            noised_price = price * (1 + self._generate_noise())
            # Create order parameters
            order_params = {
                'symbol': symbol,
                'side': side,
                'quantity': float(Decimal(str(chunk_size)).quantize(Decimal('0.00000001'))),
                'price': float(Decimal(str(noised_price)).quantize(Decimal('0.00000001'))),
                'timestamp': base_timestamp + (i * self._get_delay()),
                'recvWindow': 5000
            }
            # Sign order
            signature = self._sign_order(order_params, secret_key)
            order_params['signature'] = signature
            protected_orders.append(order_params)
        return protected_orders
    def validate_order(self,
                      order_params: Dict,
                      signature: str,
                      secret_key: str) -> bool:
        """Validate order signature"""
        expected_signature = self._sign_order(order_params, secret_key)
        return hmac.compare_digest(signature, expected_signature)
    def analyze_execution(self,
                         orders: List[Dict],
                         trades: List[Dict]) -> Dict[str, float]:
        """
        Analyze execution quality and detect potential frontrunning
        """
        analysis = {
            'price_impact': 0.0,
            'timing_analysis': 0.0,
            'size_analysis': 0.0,
            'frontrun_probability': 0.0
        }
        if not orders or not trades:
            return analysis
        # Calculate price impact
        initial_price = orders[0]['price']
        volume_weighted_price = sum(t['price'] * t['quantity'] for t in trades) / sum(t['quantity'] for t in trades)
        analysis['price_impact'] = (volume_weighted_price - initial_price) / initial_price
        # Analyze timing
        order_times = [o['timestamp'] for o in orders]
        trade_times = [t['timestamp'] for t in trades]
        time_diffs = []
        for trade_time in trade_times:
            closest_order = min(order_times, key=lambda x: abs(x - trade_time))
            time_diffs.append(abs(trade_time - closest_order))
        analysis['timing_analysis'] = np.mean(time_diffs) if time_diffs else 0
        # Analyze size patterns
        order_sizes = [o['quantity'] for o in orders]
        trade_sizes = [t['quantity'] for t in trades]
        size_correlation = np.corrcoef(order_sizes, trade_sizes[:len(order_sizes)])[0, 1]
        analysis['size_analysis'] = size_correlation if not np.isnan(size_correlation) else 0
        # Calculate frontrunning probability
        if analysis['timing_analysis'] < 50 and analysis['price_impact'] > 0.001:
            analysis['frontrun_probability'] = 0.8
        elif analysis['timing_analysis'] < 100 and analysis['price_impact'] > 0.0005:
            analysis['frontrun_probability'] = 0.5
        elif analysis['timing_analysis'] < 200 and analysis['price_impact'] > 0.0002:
            analysis['frontrun_probability'] = 0.3
        return analysis
