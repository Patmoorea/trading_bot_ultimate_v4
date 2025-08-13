from decimal import Decimal
from typing import Dict, List
import pandas as pd
from datetime import datetime, timezone
class PerformanceAnalyzer:
    def __init__(self):
        self.trades: List[Dict] = []
        self.start_balance: Decimal = Decimal('0')
        self.current_balance: Decimal = Decimal('0')
    def add_trade(self, trade: Dict):
        self.trades.append({
            **trade,
        })
    def calculate_metrics(self) -> Dict:
        if not self.trades:
            return {}
        df = pd.DataFrame(self.trades)
        return {
            'total_trades': len(df),
            'win_rate': len(df[df['profit'] > 0]) / len(df) * 100,
            'avg_profit': df['profit'].mean(),
            'max_drawdown': self._calculate_drawdown(df),
            'sharpe_ratio': self._calculate_sharpe(df),
            'profit_factor': abs(df[df['profit'] > 0]['profit'].sum() / 
                               df[df['profit'] < 0]['profit'].sum())
        }
    def _calculate_drawdown(self, df: pd.DataFrame) -> float:
        cumulative = (1 + df['profit']).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        return float(drawdown.min())
    def _calculate_sharpe(self, df: pd.DataFrame, risk_free_rate: float = 0.02) -> float:
        returns = df['profit']
        excess_returns = returns - risk_free_rate/252
        return float(excess_returns.mean() / excess_returns.std() * (252 ** 0.5))
