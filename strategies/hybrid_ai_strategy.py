import pandas as pd
# from src.ai.hybrid_model import HybridAI

def hybrid_ai_strategy(data: pd.DataFrame, **kwargs) -> pd.Series:
    """
    Stratégie IA (exemple) : à remplacer par ton vrai modèle IA.
    """
    # model = HybridAI()  # Décommente si tu as un modèle IA
    # return model.predict_signals(data)
    # Dummy : random signal
    import numpy as np
    np.random.seed(42)
    signal = pd.Series(np.random.choice([0,1,-1], size=len(data)), index=data.index)
    return signal
