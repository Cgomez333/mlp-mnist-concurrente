import time
import argparse
import numpy as np
from mlp import MLP
from data_loader import load_mnist


def iterate_minibatches(X, Y, batch_size, shuffle=True, rng=None):
    n = X.shape[0]
    indices = np.arange(n)
    if shuffle:
        rng = rng or np.random.default_rng()
        rng.shuffle(indices)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_idx = indices[start:end]
        yield X[batch_idx], Y[batch_idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, default='../data/mnist')
    parser.add_argument('--use-bin', action='store_true')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--hidden-dim', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # =============================
    # Cargar datos MNIST
    # =============================
    X_train, Y_train, X_test, Y_test = load_mnist(args.data_root, use_bin=args.use_bin)

    # ======= INFO DEL DATASET =======
    print("==== Dataset cargado ====")
    print("Train X:", X_train.shape)
    print("Train Y:", Y_train.shape)
    print("Test  X:", X_test.shape)
    print("Test  Y:", Y_test.shape)

    num_batches = (X_train.shape[0] + args.batch_size - 1) // args.batch_size
    print(f"Batch size = {args.batch_size}")
    print(f"Batches por epoch = {num_batches}")
    print(f"Muestras procesadas aprox. por epoch = {num_batches * args.batch_size}")
    print("==========================\n")

    # Crear modelo
    mlp = MLP(input_dim=784, hidden_dim=args.hidden_dim, output_dim=10, seed=args.seed)

    start_time = time.time()
    rng = np.random.default_rng(args.seed)

    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        batches = 0

        # ===== CONTADOR DE VERDADEROS BATCHES =====
        batch_counter = 0

        for Xb, Yb in iterate_minibatches(X_train, Y_train, args.batch_size, shuffle=True, rng=rng):

            batch_counter += 1  # cuántos batches reales se procesaron

            probs, cache = mlp.forward(Xb)
            loss = mlp.compute_loss(probs, Yb)
            grads = mlp.backward(cache, probs, Yb)
            mlp.step(grads, lr=args.lr)

            epoch_loss += loss
            batches += 1

        epoch_loss /= max(batches, 1)

        # Evaluación
        train_acc = mlp.accuracy(X_train[:5000], Y_train[:5000])
        test_acc = mlp.accuracy(X_test, Y_test)

        print(f"Epoch {epoch}: loss={epoch_loss:.4f} | train_acc={train_acc:.4f} | test_acc={test_acc:.4f}")
        print(f"   Batches procesados este epoch: {batch_counter}\n")

    total_time = time.time() - start_time
    print(f"Entrenamiento completado en {total_time:.2f} segundos")


if __name__ == '__main__':
    main()
