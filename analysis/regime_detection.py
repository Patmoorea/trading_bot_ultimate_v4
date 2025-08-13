from hmmlearn import hmm
from sklearn.cluster import KMeans

def detect_market_regime(X):
    # X: np.array shape (n_samples, n_features)
    model = hmm.GaussianHMM(n_components=5, covariance_type="diag", n_iter=100)
    model.fit(X)
    hidden_states = model.predict(X)
    kmeans = KMeans(n_clusters=5, random_state=42).fit(X)
    regime_labels = kmeans.labels_
    # Fusion des deux
    final_regimes = [ (h, k) for h, k in zip(hidden_states, regime_labels) ]
    return final_regimes