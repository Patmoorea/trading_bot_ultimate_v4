"""Configuration automatique pour Apple Silicon"""
import tensorflow as tf
def configure_acceleration():
    gpus = tf.config.list_physical_devices('GPU')
    cpus = tf.config.list_physical_devices('CPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"[ACCELERATION] GPU Metal configurée: {gpus}")
        except RuntimeError as e:
            print(f"[WARNING] Erreur configuration GPU: {e}")
    else:
        print(f"[INFO] Mode CPU activé: {cpus}")
configure_acceleration()
def configure_gpu():
    import tensorflow as tf
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        tf.config.set_memory_growth(physical_devices[0], True)
    print(f"[ACCELERATION] GPU Metal configurée: {physical_devices}")
def configure_gpu():
    import tensorflow as tf
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        tf.config.set_memory_growth(physical_devices[0], True)
    print(f"[ACCELERATION] GPU Metal configurée: {physical_devices}")
