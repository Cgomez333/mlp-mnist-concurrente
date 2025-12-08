import numpy as np

class MLP:
    def __init__(self, input_dim: int = 784, hidden_dim: int = 512, output_dim: int = 10, seed: int = 42):
        rng = np.random.default_rng(seed)
        # Usamos sqrt(2 / input_dim). 
        # El "2" es crucial para compensar que ReLU apaga la mitad de las neuronas.
        self.W1 = rng.normal(0, np.sqrt(2 / input_dim), size=(input_dim, hidden_dim)).astype(np.float32)
        self.b1 = np.zeros((1, hidden_dim), dtype=np.float32)
        # Usamos sqrt(1 / hidden_dim). 
        # Aquí usamos "1" porque Softmax no "mata" neuronas como ReLU.
        # (El Código 1 también usaba sqrt(1. / ...) aquí, lo cual es correcto).
        self.W2 = rng.normal(0, np.sqrt(1 / hidden_dim), size=(hidden_dim, output_dim)).astype(np.float32)
        self.b2 = np.zeros((1, output_dim), dtype=np.float32)

    # Fase 1: Forward Propagation
    def forward(self, X: np.ndarray):
        # X: (batch, input_dim)
        Z1 = X @ self.W1 + self.b1  # (batch, hidden)
        A1 = np.maximum(Z1, 0)      # ReLU
        Z2 = A1 @ self.W2 + self.b2 # (batch, output)
        # Softmax
        Z2_shift = Z2 - Z2.max(axis=1, keepdims=True)
        exp_scores = np.exp(Z2_shift)
        probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
        cache = {"X": X, "Z1": Z1, "A1": A1, "Z2": Z2, "probs": probs}
        return probs, cache

    # Fase 2: Cálculo de la Pérdida (Cross-Entropy)
    def compute_loss(self, probs: np.ndarray, Y: np.ndarray):
        # Y is one-hot (batch, output_dim)
        # Avoid log(0)
        eps = 1e-8
        log_probs = np.log(probs + eps)
        loss = -np.sum(Y * log_probs) / Y.shape[0]
        return float(loss)

    # Fase 3: Backward Propagation
    def backward(self, cache: dict, probs: np.ndarray, Y: np.ndarray):
        # grads for W2, b2, W1, b1
        batch_size = Y.shape[0]
        A1 = cache["A1"]
        X = cache["X"]
        # dL/dZ2 = probs - Y
        dZ2 = (probs - Y) / batch_size
        dW2 = A1.T @ dZ2
        db2 = dZ2.sum(axis=0, keepdims=True)
        # ReLU backprop
        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * (cache["Z1"] > 0)
        dW1 = X.T @ dZ1
        db1 = dZ1.sum(axis=0, keepdims=True)
        return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

    # Fase 4: Actualización de Pesos (SGD)
    def step(self, grads: dict, lr: float = 1e-2):
        self.W1 -= lr * grads["dW1"].astype(np.float32)
        self.b1 -= lr * grads["db1"].astype(np.float32)
        self.W2 -= lr * grads["dW2"].astype(np.float32)
        self.b2 -= lr * grads["db2"].astype(np.float32)

    def predict(self, X: np.ndarray):
        probs, _ = self.forward(X)
        return np.argmax(probs, axis=1)

    def accuracy(self, X: np.ndarray, Y: np.ndarray):
        # Y one-hot
        preds = self.predict(X)
        labels = np.argmax(Y, axis=1)
        return float((preds == labels).mean())
