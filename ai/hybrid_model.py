import tensorflow as tf
import numpy as np
import pandas as pd
import os
import json
from sklearn.model_selection import train_test_split

tf.keras.utils.set_random_seed(42)
tf.config.experimental.enable_op_determinism()


def get_configured_pairs(config_path="config/trading_pairs.json"):
    """Charge la liste des paires configurées par le bot."""
    if not os.path.exists(config_path):
        return ["BTC/USDT", "ETH/USDT"]
    with open(config_path, "r") as f:
        config = json.load(f)
    return config.get("valid_pairs", ["BTC/USDT", "ETH/USDT"])


def get_cache_csv(
    pair, interval="1h", start_str="1 Jan, 2023", end_str="now", cache_dir="data_cache"
):
    """
    Retourne le chemin du CSV cache pour la paire, timeframe et période donnés.
    Ex: BTCUSDC_1h_1 Jan, 2023_now.csv
    """
    symbol = pair.replace("/", "")
    fname = f"{symbol}_{interval}_{start_str}_{end_str}.csv"
    path = os.path.join(cache_dir, fname)
    return path if os.path.exists(path) else None


class HybridModel:
    def __init__(
        self,
        pair=None,
        window=30,
        interval="1h",
        start_str="1 Jan, 2023",
        end_str="now",
        config_path="config/trading_pairs.json",
        cache_dir="data_cache",
    ):
        self.window = window
        pairs = get_configured_pairs(config_path)
        if not pairs:
            raise ValueError("Aucune paire configurée !")
        if pair is None:
            pair = pairs[0]
        self.pair = pair
        cache_path = get_cache_csv(
            pair, interval, start_str, end_str, cache_dir=cache_dir
        )
        if cache_path is None:
            raise FileNotFoundError(
                f"[HybridModel] Fichier cache absent : {cache_dir}/{pair.replace('/', '')}_{interval}_{start_str}_{end_str}.csv"
            )
        print(f"[HybridModel] Chargement des données depuis : {cache_path}")
        df = pd.read_csv(cache_path)
        df.columns = [col.lower() for col in df.columns]
        feats = ["close", "high", "low", "volume"]
        required = set(feats)
        if not required.issubset(df.columns):
            missing = required - set(df.columns)
            raise ValueError(
                f"[HybridModel] Colonnes manquantes dans le CSV: {missing}"
            )
        if "timestamp" in df.columns:
            df = df.sort_values("timestamp").reset_index(drop=True)
        if "rsi" not in df.columns:
            delta = df["close"].diff()
            gain = (delta > 0) * delta
            loss = (delta < 0) * -delta
            avg_gain = gain.rolling(14).mean()
            avg_loss = loss.rolling(14).mean()
            rs = avg_gain / (avg_loss + 1e-9)
            df["rsi"] = 100 - (100 / (1 + rs))
        feats.append("rsi")

        # === DIAGNOSTIC NaN/inf par colonne ===
        print("[DIAG] NaN/inf count per feature before windowing:")
        for col in feats:
            n_nan = df[col].isna().sum()
            n_inf = np.isinf(df[col].values).sum()
            print(
                f"  - {col}: NaN={n_nan}, inf={n_inf}, min={df[col].min()}, max={df[col].max()}"
            )
            if n_nan > 0 or n_inf > 0:
                print(f"  [WARN] Correction NaN/inf dans {col} (ffill, bfill, puis 0)")
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                df[col] = (
                    df[col].fillna(method="ffill").fillna(method="bfill").fillna(0)
                )

        # === NORMALISATION des features ===
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        # On normalise chaque colonne indépendamment (par feature)
        df_norm = df.copy()
        for col in feats:
            df_norm[col] = scaler.fit_transform(df[[col]])

        X, y = [], []
        for i in range(len(df_norm) - window - 5):
            window_df = df_norm.iloc[i : i + window]
            sample = window_df[feats].values
            label = float(
                df["close"].iloc[i + window + 5] > df["close"].iloc[i + window]
            )
            X.append(sample)
            y.append(label)
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        if len(X) == 0 or len(y) == 0:
            raise ValueError(
                "[HybridModel] Pas assez de données pour constituer le dataset."
            )
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.20, random_state=42
        )
        print(
            f"[HybridModel] Nb samples train: {self.X_train.shape[0]} | Nb test: {self.X_test.shape[0]}"
        )
        unique, counts = np.unique(self.y_train, return_counts=True)
        print(f"[HybridModel] Train label distribution: {dict(zip(unique, counts))}")
        self.model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(window, len(feats))),
                tf.keras.layers.Conv1D(64, kernel_size=3, activation="relu"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPooling1D(2),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Conv1D(32, kernel_size=3, activation="relu"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(32, activation="relu"),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ]
        )
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

    def train_and_validate(self, lr=0.001, batch_size=64, n_epochs=5):
        batch_size = int(batch_size)
        print(f"[DEBUG] type(batch_size): {type(batch_size)}, batch_size: {batch_size}")
        assert isinstance(batch_size, int), f"batch_size is not int: {type(batch_size)}"
        print(f"[DEBUG] optimizer type: {type(self.model.optimizer)}, lr: {lr}")
        tf.keras.backend.set_value(self.model.optimizer.learning_rate, lr)

        # === DIAGNOSTICS avant fit ===
        print(
            f"[DIAG] X_train shape: {self.X_train.shape}, dtype: {self.X_train.dtype}"
        )
        print(
            f"[DIAG] y_train shape: {self.y_train.shape}, dtype: {self.y_train.dtype}"
        )
        print(
            f"[DIAG] X_train min: {np.nanmin(self.X_train)}, max: {np.nanmax(self.X_train)}, contains NaN: {np.isnan(self.X_train).any()}, contains inf: {np.isinf(self.X_train).any()}"
        )
        print(
            f"[DIAG] y_train min: {np.nanmin(self.y_train)}, max: {np.nanmax(self.y_train)}, contains NaN: {np.isnan(self.y_train).any()}, contains inf: {np.isinf(self.y_train).any()}"
        )
        print(f"[DIAG] y_train unique: {np.unique(self.y_train)}")

        # Correction NaN/inf éventuels
        if np.isnan(self.X_train).any() or np.isinf(self.X_train).any():
            print("[WARN] NaN/inf detected in X_train, replacing with 0")
            self.X_train = np.nan_to_num(self.X_train, nan=0.0, posinf=0.0, neginf=0.0)
        if np.isnan(self.y_train).any() or np.isinf(self.y_train).any():
            print("[WARN] NaN/inf detected in y_train, replacing with 0")
            self.y_train = np.nan_to_num(self.y_train, nan=0.0, posinf=0.0, neginf=0.0)

        # Callback pour détecter NaN dans la loss
        class NaNLossCallback(tf.keras.callbacks.Callback):
            def on_batch_end(self, batch, logs=None):
                loss = logs.get("loss")
                if loss is not None and (np.isnan(loss) or np.isinf(loss)):
                    print(
                        f"[ERROR] NaN/inf loss detected at batch {batch}! Stopping training."
                    )
                    self.model.stop_training = True

        self.model.fit(
            self.X_train,
            self.y_train,
            batch_size=batch_size,
            epochs=n_epochs,
            verbose=0,
            callbacks=[NaNLossCallback()],
        )
        loss, acc = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        print(f"[HybridAI] Accuracy test: {acc:.3f} | Loss: {loss:.4f}")
        return acc

    def validate(self, lr=0.001, batch_size=64, n_epochs=5):
        batch_size = int(batch_size)
        return self.train_and_validate(lr=lr, batch_size=batch_size, n_epochs=n_epochs)


class HybridAI(HybridModel):
    def __init__(
        self,
        pair=None,
        window=30,
        interval="1h",
        start_str="1 Jan, 2023",
        end_str="now",
        config_path="config/trading_pairs.json",
        cache_dir="data_cache",
    ):
        super().__init__(
            pair=pair,
            window=window,
            interval=interval,
            start_str=start_str,
            end_str=end_str,
            config_path=config_path,
            cache_dir=cache_dir,
        )
        print("🔐 Environnement TensorFlow pro initialisé")
