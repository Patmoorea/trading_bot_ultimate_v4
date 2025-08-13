import tensorflow as tf
import os
def init_gpu():
    os.environ['TF_METAL_ENABLED'] = '1'
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Configuration optimale pour Apple M1/M2/M4
            tf.config.optimizer.set_jit(True)
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            return True
        except RuntimeError as e:
            print(f"GPU Config Error: {e}")
    return False
GPU_AVAILABLE = init_gpu()
