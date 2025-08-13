"""
"""
from typing import Dict, List, Callable
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from itertools import product
class StrategyOptimizer:
    def __init__(self, data: pd.DataFrame, strategy_func: Callable, 
                 param_ranges: Dict[str, List], metric: str = 'sharpe_ratio'):
        self.data = data
        self.strategy_func = strategy_func
        self.param_ranges = param_ranges
        self.metric = metric
        self.results = []
    def optimize(self, max_workers: int = 4) -> Dict:
        param_combinations = [dict(zip(self.param_ranges.keys(), v)) 
                            for v in product(*self.param_ranges.values())]
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(self._evaluate_params, param_combinations))
        self.results = pd.DataFrame(results)
        best_result = self.results.loc[self.results[self.metric].idxmax()]
        return {
            'best_params': {k: best_result[k] for k in self.param_ranges.keys()},
            'best_metric': best_result[self.metric],
            'all_results': self.results
        }
    def _evaluate_params(self, params: Dict) -> Dict:
        result = self.strategy_func(self.data, **params)
        return {**params, self.metric: result[self.metric]}
    def plot_results(self) -> None:
        # Implémentation de la visualisation des résultats
        pass
