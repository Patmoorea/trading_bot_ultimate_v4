import pandas as pd
from typing import Callable, Dict
from src.backtesting.core.backtest_engine import BacktestEngine

class StrategyAutoSelector:
    def __init__(self, strategies: Dict[str, Callable], initial_capital: float = 10000):
        """
        strategies: dict nom_strategie -> fonction_strategy (data, **params) -> pd.Series de signaux
        """
        self.strategies = strategies
        self.initial_capital = initial_capital

    def evaluate_strategies(self, data: pd.DataFrame, **params) -> Dict[str, Dict]:
        """
        Teste chaque stratégie et retourne un dict des métriques de performance.
        """
        performances = {}
        for name, strat_func in self.strategies.items():
            engine = BacktestEngine(initial_capital=self.initial_capital)
            try:
                metrics = engine.run_backtest(data, strat_func, **params)
            except Exception as e:
                print(f"[AutoSelector] Erreur backtest {name}: {e}")
                metrics = {}
            performances[name] = metrics
        return performances

    def select_best_strategy(self, data: pd.DataFrame, critere: str = "sharpe_ratio", **params) -> str:
        """
        Retourne le nom de la meilleure stratégie selon un critère (ex: 'sharpe_ratio', 'total_return').
        """
        performances = self.evaluate_strategies(data, **params)
        best_name = None
        best_score = float('-inf')
        for name, metrics in performances.items():
            score = metrics.get(critere, float('-inf'))
            print(f"[AutoSelector] {name}: {critere}={score:.4f}")
            if score > best_score:
                best_score = score
                best_name = name
        print(f"[AutoSelector] Meilleure stratégie: {best_name} ({critere}={best_score:.4f})")
        return best_name

    def run_with_best(self, data: pd.DataFrame, critere: str = "sharpe_ratio", **params):
        """
        Lance le backtest complet avec la meilleure stratégie.
        """
        best_name = self.select_best_strategy(data, critere, **params)
        print(f"[AutoSelector] Lancement du backtest final avec la stratégie: {best_name}")
        engine = BacktestEngine(initial_capital=self.initial_capital)
        metrics = engine.run_backtest(data, self.strategies[best_name], **params)
        return best_name, metrics

