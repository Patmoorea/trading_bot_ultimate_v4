import tensorflow as tf
from .gpu_config import GPU_ACTIVATED
def verify_gpu():
    """Vérification complète des capacités GPU"""
    print("\n=== Diagnostic Apple Silicon ===")
    print(f"TensorFlow Version: {tf.__version__}")
    print(f"GPU Activé: {GPU_ACTIVATED}")
    print(f"Devices Physiques: {tf.config.list_physical_devices()}")
    print(f"is_built_with_cuda: {tf.test.is_built_with_cuda()}")
    # Alternative pour vérifier Metal
    try:
        from tensorflow.python.compiler.mlcompute import mlcompute
        mlcompute.set_mlc_device(device_name='gpu')
        print("Metal Support: True (via mlcompute)")
    except Exception as e:
        print(f"Metal Support: False - {str(e)}")
if __name__ == "__main__":
    verify_gpu()
