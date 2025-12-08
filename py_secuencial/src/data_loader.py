import os
import struct
import numpy as np

# Carga MNIST desde archivos IDX originales
# Referencia formato: http://yann.lecun.com/exdb/mnist/

def _read_idx_images(path: str):
    with open(path, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        assert magic == 2051, f"Formato IDX imágenes inválido: {magic}"
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape((num, rows * cols))
        return data.astype(np.float32) / 255.0


def _read_idx_labels(path: str, num_classes: int = 10):
    with open(path, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        assert magic == 2049, f"Formato IDX labels inválido: {magic}"
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        one_hot = np.zeros((num, num_classes), dtype=np.float32)
        one_hot[np.arange(num), labels] = 1.0
        return one_hot

# Carga desde binarios generados por scripts/preprocess_for_c.py
# Formato esperado:
#   train_images.bin: float32 [N, 784]
#   train_labels.bin: float32 one-hot [N, 10]

def _read_bin_matrix(path: str, cols: int):
    data = np.fromfile(path, dtype=np.float32)
    assert data.size % cols == 0, f"Dimensiones inesperadas en {path}"
    return data.reshape((-1, cols))


def load_mnist(data_root: str, use_bin: bool = False):
    if use_bin:
        X_train = _read_bin_matrix(os.path.join(data_root, 'train_images.bin'), 784)
        Y_train = _read_bin_matrix(os.path.join(data_root, 'train_labels.bin'), 10)
        X_test = _read_bin_matrix(os.path.join(data_root, 't10k_images.bin'), 784)
        Y_test = _read_bin_matrix(os.path.join(data_root, 't10k_labels.bin'), 10)
    else:
        X_train = _read_idx_images(os.path.join(data_root, 'train-images-idx3-ubyte'))
        Y_train = _read_idx_labels(os.path.join(data_root, 'train-labels-idx1-ubyte'))
        X_test = _read_idx_images(os.path.join(data_root, 't10k-images-idx3-ubyte'))
        Y_test = _read_idx_labels(os.path.join(data_root, 't10k-labels-idx1-ubyte'))
    return X_train, Y_train, X_test, Y_test
