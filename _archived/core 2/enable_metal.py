import os
import tensorflow as tf
# Configuration spécifique M4
os.environ['TF_METAL_ENABLED'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
def verify_metal():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("GPU Apple M4 détecté - Optimisations Metal activées")
        tf.config.optimizer.set_experimental_options(
            {'disable_meta_optimizer': False})
        return True
    return False
METAL_ACTIVE = verify_metal()
