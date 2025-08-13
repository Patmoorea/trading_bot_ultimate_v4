class ParallelOptimizer:
    def __init__(self):
        self.study = optuna.create_study(
            pruner=optuna.pruners.HyperbandPruner()
        )
    async def optimize_parallel(self, n_trials=200):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self.objective)
                for _ in range(n_trials)
            ]
            results = [f.result() for f in futures]
