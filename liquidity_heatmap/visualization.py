import cv2
import numpy as np
def generate_heatmap(orderbook: dict, levels: int = 20) -> np.ndarray:
    bids = np.array(orderbook['bids'][:levels])
    asks = np.array(orderbook['asks'][:levels])
    # Normalisation
    bid_heat = (bids[:, 1] / bids[:, 1].max() * 255).astype(np.uint8)
    ask_heat = (asks[:, 1] / asks[:, 1].max() * 255).astype(np.uint8)
    # Création de la heatmap
    heatmap = np.zeros((levels, 2))
    heatmap[:, 0] = bid_heat
    heatmap[:, 1] = ask_heat
    return cv2.applyColorMap(heatmap.astype(np.uint8), cv2.COLORMAP_JET)
