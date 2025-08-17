import pandas as pd
import ast
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# === 1. Charger le dataset ===
df = pd.read_csv("training_data.csv")

print(f"Dataset chargé: {len(df)} lignes")

# === 2. Préparer les features ===
def extract_features(features_str):
    """Convertir string dict en flat features"""
    try:
        features = ast.literal_eval(features_str)  # convertir str -> dict
        flat = {}
        for group, vals in features.items():
            if isinstance(vals, dict):
                for k, v in vals.items():
                    flat[f"{group}_{k}"] = v
            else:
                flat[group] = vals
        return flat
    except Exception:
        return {}

# Appliquer à toutes les lignes
features_expanded = df["features"].apply(extract_features).apply(pd.Series)
df = pd.concat([df, features_expanded], axis=1)

# === 3. Label = décision (action BUY=1, SELL=0, HOLD=-1) ===
def map_decision(dec):
    if isinstance(dec, str):
        dec = dec.upper()
        if "BUY" in dec:
            return 1
        elif "SELL" in dec:
            return 0
        elif "HOLD" in dec:
            return -1
    return -1

df["label"] = df["decision"].apply(map_decision)

# === 4. Features finales ===
X = df.drop(columns=["timestamp", "pair", "equity", "decision", "price", "features", "label"])
y = df["label"]

# === 5. Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 6. Standardisation ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.fillna(0))
X_test_scaled = scaler.transform(X_test.fillna(0))

# === 7. Modèle simple (Logistic Regression) ===
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# === 8. Évaluation ===
y_pred = model.predict(X_test_scaled)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# === 9. Sauvegarde modèle + scaler ===
joblib.dump(model, "ml_model.pkl")
joblib.dump(scaler, "ml_scaler.pkl")

print("✅ Modèle et scaler sauvegardés.")
