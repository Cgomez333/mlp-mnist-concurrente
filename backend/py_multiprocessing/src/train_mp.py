import time
import argparse
import numpy as np
import multiprocessing as mp
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


def worker_compute_grads(args_tuple):
    # Unpack inputs for the worker
    weights, Xb, Yb = args_tuple
    # Reconstruct temporary model from shared weights
    tmp = MLP(input_dim=784, hidden_dim=weights['W1'].shape[1], output_dim=10)
    tmp.set_weights(weights)
    probs, cache = tmp.forward(Xb)
    grads = tmp.backward(cache, probs, Yb)
    # Return summed grads (not averaged) and batch size for proper aggregation
    return {
        'dW1': grads['dW1'],
        'db1': grads['db1'],
        'dW2': grads['dW2'],
        'db2': grads['db2'],
        'bs': Yb.shape[0]
    }


def aggregate_grads(grad_list):
    # Sum grads weighted by batch sizes, then divide by total batch size
    total_bs = sum(g['bs'] for g in grad_list)
    dW1 = sum(g['dW1'] * (g['bs'] / total_bs) for g in grad_list)
    db1 = sum(g['db1'] * (g['bs'] / total_bs) for g in grad_list)
    dW2 = sum(g['dW2'] * (g['bs'] / total_bs) for g in grad_list)
    db2 = sum(g['db2'] * (g['bs'] / total_bs) for g in grad_list)
    return {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}


def split_batch(Xb, Yb, workers):
    # Split into roughly equal sub-batches
    n = Xb.shape[0]
    sizes = [(n * i) // workers for i in range(workers + 1)]
    slices = [(sizes[i], sizes[i+1]) for i in range(workers)]
    return [ (Xb[s:e], Yb[s:e]) for s, e in slices if e > s ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, default='../data/mnist')
    parser.add_argument('--use-bin', action='store_true')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--hidden-dim', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()

    # Datos
    X_train, Y_train, X_test, Y_test = load_mnist(args.data_root, use_bin=args.use_bin)

    # Modelo maestro
    mlp = MLP(input_dim=784, hidden_dim=args.hidden_dim, output_dim=10, seed=args.seed)

    start_time = time.time()
    rng = np.random.default_rng(args.seed)

    # Multiprocessing Pool
    with mp.get_context('spawn').Pool(processes=args.workers) as pool:
        for epoch in range(1, args.epochs + 1):
            epoch_loss = 0.0
            batches = 0
            for Xb, Yb in iterate_minibatches(X_train, Y_train, args.batch_size, shuffle=True, rng=rng):
                # Broadcast current weights to workers
                weights = mlp.copy_weights()
                # Split batch
                splits = split_batch(Xb, Yb, args.workers)
                # Map to workers
                grad_results = pool.map(worker_compute_grads, [ (weights, xs, ys) for xs, ys in splits ])

                # Aggregate grads (average by sub-batch sizes)
                grads = aggregate_grads(grad_results)

                # Compute loss (on full batch using master)
                probs, cache = mlp.forward(Xb)
                loss = mlp.compute_loss(probs, Yb)
                epoch_loss += loss
                batches += 1

                # Update master weights
                mlp.step(grads, lr=args.lr)

            epoch_loss /= max(batches, 1)
            # Evaluación
            train_acc = mlp.accuracy(X_train[:5000], Y_train[:5000])
            test_acc = mlp.accuracy(X_test, Y_test)
            print(f"Epoch {epoch}: loss={epoch_loss:.4f} train_acc={train_acc:.4f} test_acc={test_acc:.4f}")

    total_time = time.time() - start_time
    print(f"Entrenamiento completado en {total_time:.2f} segundos")


if __name__ == '__main__':
    main()
