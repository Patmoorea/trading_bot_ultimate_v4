import os
import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from src.ai.deep_learning_model import CNNLSTMModel
import json
import shutil
import time

SEQ_LEN = 20
FUTURE_SHIFT = 10
THRESHOLD = 0.002
MAX_TOTAL_EPOCHS = 500  # <-- Change ici si tu veux moins/plus d'epochs cumulés


def load_best_params(path="config/best_hyperparams.json"):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_checkpoint(model, optimizer, epoch, path="src/models/checkpoint.pth"):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
        },
        path,
    )


def load_checkpoint(model, optimizer, path="src/models/checkpoint.pth"):
    print(f"[DEBUG] torch.load: path={path}")  # Ajout log debug
    if os.path.exists(path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint["epoch"]
    else:
        print(f"[WARN] Checkpoint {path} absent, pas de reprise d'entraînement.")
    return 0


def load_data_from_df(df, seq_len=20, future_shift=10, threshold=0.002):
    feature_cols = ["close", "high", "low", "volume", "rsi", "macd", "volatility"]
    for col in feature_cols:
        if col not in df:
            print(
                f"Manque la colonne {col} dans le DataFrame, impossible d’entraîner !"
            )
            return None, None
        df[col] = (df[col] - df[col].mean()) / (df[col].std() + 1e-8)
        if df[col].isnull().any() or np.isinf(df[col]).any():
            print(f"Colonne {col} contient nan ou inf après normalisation !")
    df = df.dropna(subset=feature_cols)
    X, y = [], []
    for i in range(len(df) - seq_len - future_shift):
        features = [df[col].iloc[i : i + seq_len].values for col in feature_cols]
        if any(np.isnan(f).any() or np.isinf(f).any() for f in features):
            continue
        feat_arr = np.stack(features, axis=1)
        X.append(feat_arr)
        future_close = df["close"].iloc[i + seq_len + future_shift - 1]
        now_close = df["close"].iloc[i + seq_len - 1]
        try:
            label = (
                1.0 if (future_close - now_close) / abs(now_close) > threshold else 0.0
            )
        except Exception:
            label = 0.0
        y.append(label)
    if not X or not y:
        print("Aucune donnée d'entraînement disponible")
        return None, None
    X = np.stack(X)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.array(y, dtype=np.float32).reshape(-1, 1)
    return X, y


def add_dl_features(df):
    import pandas_ta as pta

    if "rsi" not in df:
        df["rsi"] = pta.rsi(df["close"], length=14)
    if "macd" not in df:
        macd = pta.macd(df["close"])
        df["macd"] = macd["MACD_12_26_9"] if "MACD_12_26_9" in macd else np.nan
    if "volatility" not in df:
        returns = np.log(df["close"]).diff()
        df["volatility"] = returns.rolling(14).std()
    return df


def promote_trained_model():
    src = "src/models/cnn_lstm_model_training.pth"
    dst = "src/models/cnn_lstm_model.pth"
    if os.path.exists(src):
        shutil.copy(src, dst)
        print(f"🚀 Nouveau modèle promu en production ({dst})")
    else:
        print("Aucun modèle entraîné à promouvoir.")


def train_with_live_data(
    df_live,
    model_save_path="src/models/cnn_lstm_model_training.pth",
    reset_on_n_epochs=True,
):
    """
    Entraîne le modèle CNN-LSTM sur des données live.
    Limite automatique : reset après MAX_TOTAL_EPOCHS cumulés pour éviter le surapprentissage.
    """

    # 1. Chargement des hyperparams Optuna/AutoML
    best_params = load_best_params()
    lr = best_params.get("lr", 0.001)
    n_epochs = best_params.get("n_epochs", 100)
    batch_size = best_params.get("batch_size", 64)

    # 2. Préparation des données
    X, y = load_data_from_df(df_live)
    if X is None or y is None:
        print("Pas assez de données pour entraîner le modèle.")
        return
    print("Features shape:", X.shape, "Targets shape:", y.shape)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.1, shuffle=True, random_state=42
    )

    # 3. Initialisation du modèle et de l'optimizer
    model = CNNLSTMModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()
    checkpoint_path = "src/models/checkpoint.pth"

    # 4. Chargement du checkpoint si présent
    start_epoch = load_checkpoint(model, optimizer, checkpoint_path)
    if start_epoch > 0:
        print(f"✅ Reprise de l'entraînement à l'epoch {start_epoch+1}")
    else:
        print(
            "⏩ Entraînement à partir de zéro (pas de reset forcé, checkpoint conservé si existant)."
        )

    # 4.1. Reset si trop d'epochs cumulés
    if start_epoch >= MAX_TOTAL_EPOCHS:
        print(
            f"⏹️ Reset automatique : {start_epoch} epochs cumulés atteints. Nouveau training sur données récentes."
        )
        # Supprimer l'ancien checkpoint
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
        # Réinitialiser modèle et optimizer
        model = CNNLSTMModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        start_epoch = 0

    # 5. Boucle d'entraînement (n_epochs à chaque run)
    for epoch in range(start_epoch, start_epoch + n_epochs):
        t0 = time.time()
        model.train()
        idxs = np.random.permutation(len(X_train))
        X_train, y_train = X_train[idxs], y_train[idxs]
        batch_losses = []
        for i in range(0, len(X_train), batch_size):
            xb = torch.FloatTensor(X_train[i : i + batch_size]).transpose(1, 2)
            yb = torch.FloatTensor(y_train[i : i + batch_size])
            optimizer.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
            # print(f"  Epoch {epoch+1} - Batch {i//batch_size+1}/{(len(X_train)-1)//batch_size+1} - Batch loss: {loss.item():.6f}")

        epoch_duration = time.time() - t0
        print(
            f"Epoch {epoch+1}/{start_epoch+n_epochs} - Mean batch loss: {np.mean(batch_losses):.6f} - Durée: {epoch_duration:.2f}s"
        )
        print("-" * 60)

        # Sauvegarde du checkpoint à chaque epoch
        save_checkpoint(model, optimizer, epoch + 1, checkpoint_path)

        # Évaluation
        model.eval()
        with torch.no_grad():
            xb = torch.FloatTensor(X_val).transpose(1, 2)
            yb = torch.FloatTensor(y_val)
            y_pred = model(xb)
            val_loss = loss_fn(y_pred, yb).item()
            acc = ((y_pred > 0.5).float() == yb).float().mean().item()
            print(f"    Validation loss: {val_loss:.6f}  |  Val acc: {acc:.3f}")

    # 6. Sauvegarde finale du modèle entraîné
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"✅ Modèle entraîné et sauvegardé à {model_save_path}")

    # 7. Promotion du modèle en production
    promote_trained_model()
