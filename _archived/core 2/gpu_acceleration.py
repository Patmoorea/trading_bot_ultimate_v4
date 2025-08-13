# Nouveau module d'accélération matérielle
import tensorflow as tf
def configure_gpu():
    """Configuration automatique pour M1/M2/M4"""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            return True
        except RuntimeError as e:
            print(f"GPU Error: {e}")
    return False
