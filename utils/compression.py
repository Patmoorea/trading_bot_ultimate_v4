import lz4.frame

def save_lz4(obj, filename):
    import pickle
    with lz4.frame.open(filename, 'wb') as f:
        pickle.dump(obj, f)

def load_lz4(filename):
    import pickle
    with lz4.frame.open(filename, 'rb') as f:
        return pickle.load(f)