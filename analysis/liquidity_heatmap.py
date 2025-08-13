import cv2
import numpy as np

def generate_heatmap(orderbook_matrix):
    heatmap = cv2.applyColorMap(orderbook_matrix, cv2.COLORMAP_JET)
    return heatmap