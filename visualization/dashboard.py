import streamlit as st
import pandas as pd
import asyncio
import numpy as np
from typing import List, Dict
class Position:
    def __init__(self, symbol: str, amount: float, entry_price: float, current_price: float, leverage: float = 1.0, margin_type: str = "isolated"):
        self.symbol = symbol
        self.amount = amount
        self.entry_price = entry_price
        self.current_price = current_price
        self.leverage = leverage
        self.margin_type = margin_type
    def unrealized_pnl(self) -> float:
        return (self.current_price - self.entry_price) * self.amount
    def value(self) -> float:
        return self.amount * self.current_price
class TradingDashboard:
    def __init__(self, initial_balance: float, websocket_enabled: bool = False, realtime_updates: bool = False):
        self.initial_balance = initial_balance
        self.positions: List[Position] = []
        self.balance = initial_balance
        self.websocket_enabled = websocket_enabled
        self.realtime_updates = realtime_updates
        self.pnl_history = []
        self.drawdown_history = []
        self.metrics: Dict[str, float] = {
            "total_pnl": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "average_return": 0.0,
            "volatility": 0.0
        }
    def add_position(self, position: Position):
        self.positions.append(position)
        self.update_metrics()
    def remove_position(self, symbol: str):
        self.positions = [p for p in self.positions if p.symbol != symbol]
        self.update_metrics()
    def update_balance(self, new_balance: float):
        self.balance = new_balance
        self.update_metrics()
    def update_metrics(self):
        total_pnl = sum([p.unrealized_pnl() for p in self.positions])
        total_value = sum([p.value() for p in self.positions])
        equity = self.balance + total_pnl
        self.pnl_history.append(total_pnl)
        self.drawdown_history.append(self.initial_balance - equity)
        self.metrics["total_pnl"] = total_pnl
        self.metrics["max_drawdown"] = max(self.drawdown_history) if self.drawdown_history else 0
        self._update_metrics()
    def _update_metrics(self):
        returns = np.diff(self.pnl_history) if len(self.pnl_history) > 1 else np.array([0.0])
        negative_returns = returns[returns < 0]
        if len(returns) > 1:
            self.metrics["average_return"] = np.mean(returns)
            self.metrics["volatility"] = np.std(returns)
            self.metrics["sharpe_ratio"] = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0.0
            self.metrics["sortino_ratio"] = (
                np.mean(returns) / np.std(negative_returns) if np.std(negative_returns) > 0 else 0.0
            )
        wins = [p for p in self.positions if p.unrealized_pnl() > 0]
        self.metrics["win_rate"] = len(wins) / len(self.positions) if self.positions else 0
    async def start_data_stream(self):
        while self.websocket_enabled and self.realtime_updates:
            await asyncio.sleep(1)
            self.update_metrics()
    async def stop_data_stream(self):
        self.realtime_updates = False
    def render(self):
        st.set_page_config(layout="wide")
        st.title("📈 Tableau de bord du Trading Bot Ultimate")
        st.subheader("📊 Solde et Performance")
        st.metric("Solde actuel (USDC)", f"{self.balance:,.2f}")
        st.metric("PnL total", f"{self.metrics['total_pnl']:,.2f}")
        st.metric("Drawdown max", f"{self.metrics['max_drawdown']:,.2f}")
        st.subheader("📈 Statistiques de Performance")
        st.write(pd.DataFrame([self.metrics]))
        st.subheader("📄 Positions ouvertes")
        if self.positions:
            positions_data = [
                {
                    "Symbole": p.symbol,
                    "Quantité": p.amount,
                    "Prix d'entrée": p.entry_price,
                    "Prix actuel": p.current_price,
                    "PnL": p.unrealized_pnl(),
                    "Valeur": p.value(),
                    "Levier": p.leverage,
                    "Type de marge": p.margin_type
                }
                for p in self.positions
            ]
            st.dataframe(pd.DataFrame(positions_data))
        else:
            st.info("Aucune position ouverte.")
