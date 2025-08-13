from datetime import datetime
from decimal import Decimal
from typing import Dict
class LiveEvolutionMonitor:
    def __init__(self):
        self.state = {
            'timestamp': datetime.utcnow(),
            'last_price': Decimal('0'),
            'volume_24h': Decimal('0'),
            'hourly_volumes': [],
            'price_velocities': []
        }
        self.max_velocity = Decimal('50.0')  # $/seconde
    def update_state(self, market_data: Dict) -> None:
        self.state['timestamp'] = datetime.utcnow()
        self.state['last_price'] = Decimal(str(market_data['price']))
        self.state['volume_24h'] = Decimal(str(market_data['volume_24h']))
        # Calcul de la vélocité du prix
        if self.price_velocities:
            time_diff = (self.state['timestamp'] - self.price_velocities[-1]['timestamp']).total_seconds()
            if time_diff > 0:
                velocity = abs(self.state['last_price'] - self.price_velocities[-1]['price']) / Decimal(str(time_diff))
                self.price_velocities.append({
                    'timestamp': self.state['timestamp'],
                    'price': self.state['last_price'],
                    'velocity': velocity
                })
    def get_current_velocity(self) -> Decimal:
        if not self.price_velocities:
            return Decimal('0')
        return self.price_velocities[-1]['velocity']
    def is_velocity_alert(self) -> bool:
        return self.get_current_velocity() > self.max_velocity
