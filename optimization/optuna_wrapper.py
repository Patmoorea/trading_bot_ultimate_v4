import optuna
import json


def tune_hyperparameters(n_trials=10, cache_dir="data_cache", model_type="hybrid"):
    """
    Optimisation des hyperparamètres du modèle sur toutes les paires de la config.
    - n_trials: nombre d'essais Optuna
    - cache_dir: dossier des fichiers de cache OHLCV
    - model_type: "hybrid" (HybridAI) ou "dl" (DeepLearningModel)
    Résultat : meilleures valeurs de lr et logs d'accuracy pour chaque paire.
    """
    from src.bot_runner import load_config

    if model_type == "hybrid":
        from src.ai.hybrid_model import HybridAI

        ModelClass = HybridAI
    elif model_type == "dl":
        from src.ai.deep_learning_model import DeepLearningModel

        ModelClass = DeepLearningModel
    else:
        raise ValueError("model_type doit être 'hybrid' ou 'dl'")

    def objective(trial):
        print(f">>> OBJECTIVE Optuna appelé, trial: {trial.number}")
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
        batch_size = int(batch_size)  # <-- AJOUTE CETTE LIGNE ICI !!!
        pairs = load_config()  # ex: ["BTC/USDC", "ETH/USDC", ...]
        scores = []
        for pair in pairs:
            try:
                print(f"[Optuna] TRAIN sur {pair}…")
                if model_type == "hybrid":
                    model = ModelClass(
                        pair=pair,
                        window=30,
                        interval="1h",
                        start_str="1 Jan, 2023",
                        end_str="now",
                        cache_dir=cache_dir,
                    )
                    model.learning_rate = lr
                    acc = (
                        model.validate(lr=lr, batch_size=batch_size)
                        if hasattr(model, "validate")
                        else 0.0
                    )
                elif model_type == "dl":
                    # DeepLearningModel: adaptation
                    model = ModelClass()
                    acc = model.train_and_validate(
                        pair=pair,
                        window=30,
                        interval="1h",
                        start_str="1 Jan, 2023",
                        end_str="now",
                        cache_dir=cache_dir,
                        lr=lr,
                        batch_size=batch_size,
                    )
                print(f"[Optuna] {pair} | Accuracy={acc:.4f}")
                scores.append(acc)
            except Exception as e:
                print(f"[Optuna] Erreur sur {pair}: {e}")
        if not scores:
            print("[Optuna] Aucune paire dispo pour ce trial !")
        return float(sum(scores)) / len(scores) if scores else 0.0

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    best_params = study.best_params

    # Sauvegarde dans un fichier JSON
    with open("optuna_best_params.json", "w") as f:
        json.dump(best_params, f, indent=2)

    print("=== Optuna terminé, meilleurs hyperparams ===")
    print(best_params)
    return best_params


def optimize_hyperparameters_full(n_trials=200, timeout=3600, cache_dir="data_cache"):
    """
    Optimisation complète du modèle HybridAIEnhanced (CNN-LSTM + Transformer) sur toutes les paires.
    Retourne les meilleurs trials (accuracy et latency).
    """
    from src.ai.hybrid_engine import HybridAIEnhanced
    from src.bot_runner import load_config

    study = optuna.create_study(directions=["maximize", "minimize"])

    def objective(trial):
        print(f">>> OBJECTIVE Optuna appelé, trial: {trial.number}")
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
        batch_size = int(batch_size)  # <-- AJOUTE CETTE LIGNE ICI !!!
        pairs = load_config()  # ex: ["BTC/USDC", "ETH/USDC", ...]
        scores = []
        for pair in pairs:
            try:
                print(f"[Optuna] TRAIN sur {pair}…")
                if model_type == "hybrid":
                    model = ModelClass(
                        pair=pair,
                        window=30,
                        interval="1h",
                        start_str="1 Jan, 2023",
                        end_str="now",
                        cache_dir=cache_dir,
                    )
                    model.learning_rate = lr
                    acc = (
                        model.validate(lr=lr, batch_size=batch_size)
                        if hasattr(model, "validate")
                        else 0.0
                    )
                elif model_type == "dl":
                    # DeepLearningModel: adaptation
                    model = ModelClass()
                    acc = model.train_and_validate(
                        pair=pair,
                        window=30,
                        interval="1h",
                        start_str="1 Jan, 2023",
                        end_str="now",
                        cache_dir=cache_dir,
                        lr=lr,
                        batch_size=batch_size,
                    )
                print(f"[Optuna] {pair} | Accuracy={acc:.4f}")
                scores.append(acc)
            except Exception as e:
                print(f"[Optuna] Erreur sur {pair}: {e}")
        if not scores:
            print("[Optuna] Aucune paire dispo pour ce trial !")
        return float(sum(scores)) / len(scores) if scores else 0.0

    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    print("=== Optuna full terminé, best trials ===")
    print(study.best_trials)
    return study.best_trials
