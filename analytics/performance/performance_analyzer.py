from typing import Dict, List
import pandas as pd
from decimal import Decimal
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
class PerformanceAnalyzer:
    def __init__(self):
        self.trades_df = pd.DataFrame()
    def add_trade(self, trade: Dict):
        trade_df = pd.DataFrame([{
            'symbol': trade['symbol'],
            'side': trade['side'],
            'price': float(trade['price']),
            'amount': float(trade['amount']),
            'cost': float(trade['price']) * float(trade['amount']),
            'exchange': trade['exchange']
        }])
        self.trades_df = pd.concat([self.trades_df, trade_df], ignore_index=True)
    def calculate_metrics(self, timeframe: str = '1d') -> Dict:
        if self.trades_df.empty:
            return {}
        # Regroupement par période
        trades_grouped = self.trades_df.set_index('timestamp').groupby(pd.Grouper(freq=timeframe))
        # Calcul des métriques
        metrics = {
            'total_trades': len(self.trades_df),
            'volume_total': self.trades_df['cost'].sum(),
            'profit_loss': self._calculate_pnl(),
            'win_rate': self._calculate_win_rate(),
            'avg_trade_size': self.trades_df['cost'].mean(),
            'largest_trade': self.trades_df['cost'].max(),
            'most_traded_symbol': self.trades_df['symbol'].mode().iloc[0]
        }
        return metrics
    def generate_report(self, output_file: str = 'performance_report.html'):
        metrics = self.calculate_metrics()
        # Création du rapport HTML
        html_content = f"""
        <html>
            <head>
                <title>Performance Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; }}
                    .metric {{ margin: 20px; padding: 10px; border: 1px solid #ddd; }}
                </style>
            </head>
            <body>
                <h1>Trading Performance Report</h1>
                <div class="metric">
                    <h2>Key Metrics</h2>
                    <p>Total Trades: {metrics['total_trades']}</p>
                    <p>Total Volume: ${metrics['volume_total']:,.2f}</p>
                    <p>Profit/Loss: ${metrics['profit_loss']:,.2f}</p>
                    <p>Win Rate: {metrics['win_rate']:.2%}</p>
                </div>
            </body>
        </html>
        """
        with open(output_file, 'w') as f:
            f.write(html_content)
    def _calculate_pnl(self) -> float:
        buy_trades = self.trades_df[self.trades_df['side'] == 'buy']['cost'].sum()
        sell_trades = self.trades_df[self.trades_df['side'] == 'sell']['cost'].sum()
        return sell_trades - buy_trades
    def _calculate_win_rate(self) -> float:
        if self.trades_df.empty:
            return 0.0
        profitable_trades = len(self.trades_df[self.trades_df['side'] == 'sell'])
        total_trades = len(self.trades_df)
        return profitable_trades / total_trades if total_trades > 0 else 0.0
