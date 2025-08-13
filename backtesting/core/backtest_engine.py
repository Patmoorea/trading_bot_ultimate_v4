import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import json
from src.ai.hybrid_model import HybridAI


class BacktestEngine:
    def __init__(self, initial_capital: float = 10000):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions: Dict = {}
        self.trades: List[Dict] = []
        self.metrics: Dict = {}

        # Chargement automatique des hyperparams Optuna si dispo
        self.best_params = self._load_optuna_params()

    def _load_optuna_params(self) -> Dict:
        """
        Charge automatiquement les meilleurs hyperparamètres Optuna, si le fichier existe.
        """
        try:
            with open("optuna_best_params.json", "r") as f:
                best_params = json.load(f)
            print(f"[BacktestEngine] Hyperparamètres IA chargés : {best_params}")
            return best_params
        except Exception:
            return {}

    def run_backtest(self, data: pd.DataFrame, strategy_func, **params) -> Dict:
        self.reset()
        # Injection automatique des meilleurs hyperparams IA si dispo
        if hasattr(strategy_func, "use_optuna_params") and self.best_params:
            params.update(self.best_params)

        signals = strategy_func(data, **params)
        if isinstance(signals, pd.Series):
            signals = signals.values
        elif isinstance(signals, pd.DataFrame):
            signals = signals.iloc[:, 0].values
        elif isinstance(signals, list):
            signals = np.array(signals)
        else:
            signals = np.array(signals)

        # Debug: print signal distribution
        # print("Signal distribution:", pd.Series(signals).value_counts())

        # Ajout : support long ET short (bi-directionnel)
        for i, (timestamp, row) in enumerate(data.iterrows()):
            signal = signals[i]
            # print(f"[{timestamp}] Signal: {signal}, Position: {self.positions}")

            # Ouvre une position LONG
            if signal == 1 and not self.positions:
                self._open_position("LONG", row["close"], timestamp)
            # Ouvre une position SHORT
            elif signal == -1 and not self.positions:
                self._open_position("SHORT", row["close"], timestamp)
            # Ferme une position (peu importe le sens)
            elif signal == 0 and self.positions:
                self._close_position(row["close"], timestamp)
            # Peut ajouter gestion TP/SL ou autre ici

        # Si position ouverte à la fin du backtest, on la clôture
        if self.positions:
            self._close_position(data.iloc[-1]["close"], data.index[-1])

        return self.calculate_metrics()

    def _open_position(self, direction: str, price: float, timestamp: datetime):
        size = self.current_capital * 0.95  # 95% du capital
        self.positions = {
            "direction": direction,
            "entry_price": price,
            "size": size,
            "entry_time": timestamp,
        }

    def _close_position(self, price: float, timestamp: datetime):
        if not self.positions:
            return
        qty = self.positions["size"] / self.positions["entry_price"]
        pnl = (price - self.positions["entry_price"]) * qty
        if self.positions["direction"] == "SHORT":
            pnl = -pnl
        self.trades.append(
            {
                "entry_time": self.positions["entry_time"],
                "exit_time": timestamp,
                "entry_price": self.positions["entry_price"],
                "exit_price": price,
                "pnl": pnl,
                "direction": self.positions["direction"],
            }
        )
        self.current_capital += pnl
        self.positions = {}

    def calculate_metrics(self) -> dict:
        if not self.trades:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "max_drawdown": 0.0,
                "sharpe_ratio": 0.0,
                "final_capital": self.current_capital,
                "total_return": 0.0,
            }
        pnls = [trade["pnl"] for trade in self.trades]
        winning_trades = [pnl for pnl in pnls if pnl > 0]
        losing_trades = [pnl for pnl in pnls if pnl < 0]
        self.metrics = {
            "total_trades": len(self.trades),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": len(winning_trades) / len(self.trades) if self.trades else 0.0,
            "avg_win": float(np.mean(winning_trades)) if winning_trades else 0.0,
            "avg_loss": float(np.mean(losing_trades)) if losing_trades else 0.0,
            "max_drawdown": float(self._calculate_max_drawdown()),
            "sharpe_ratio": float(self._calculate_sharpe_ratio()),
            "final_capital": float(self.current_capital),
            "total_return": float(
                (self.current_capital - self.initial_capital) / self.initial_capital
            ),
        }
        for k, v in self.metrics.items():
            if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
                self.metrics[k] = 0.0
        return self.metrics

    def _calculate_max_drawdown(self) -> float:
        capital_curve = [self.initial_capital]
        for trade in self.trades:
            capital_curve.append(capital_curve[-1] + trade["pnl"])
        capital_curve = pd.Series(capital_curve)
        rolling_max = capital_curve.expanding().max()
        drawdowns = (capital_curve - rolling_max) / rolling_max
        return drawdowns.min() if not drawdowns.empty else 0.0

    def _calculate_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        if not self.trades:
            return 0.0
        returns = pd.Series([trade["pnl"] for trade in self.trades])
        if returns.std() == 0:
            return 0.0
        excess_returns = returns.mean() - risk_free_rate / 252
        return np.sqrt(252) * excess_returns / returns.std()

    def reset(self):
        self.current_capital = self.initial_capital
        self.positions = {}
        self.trades = []
        self.metrics = {}

    @staticmethod
    def monte_carlo_backtest(equity_curve, n_simulations=1000):
        results = []
        returns = np.diff(equity_curve) / equity_curve[:-1]
        for _ in range(n_simulations):
            simulated = [equity_curve[0]]
            for r in np.random.choice(returns, size=len(returns)):
                simulated.append(simulated[-1] * (1 + r))
            results.append(simulated[-1])
        return np.percentile(results, [5, 50, 95])  # VaR5%, Median, VaR95%
