import time
import numpy as np
import minitensor as mt
import mt_utils

# Settings
HIDDEN = 128
BATCH_SIZE = 64
LR = 0.075
EPOCHS = 20
SEED = 0


# Loading MNIST: IDX files are raw bytes after a header (16 bytes for images, 8 for labels)
def load_mnist():
    images = lambda f: np.fromfile(f"MNIST/{f}", np.uint8, offset=16).reshape(-1, 784).astype(np.float32) / 255.0
    labels = lambda f: np.fromfile(f"MNIST/{f}", np.uint8, offset=8)
    return (images("train-images.idx3-ubyte"), labels("train-labels.idx1-ubyte"),
            images("t10k-images.idx3-ubyte"), labels("t10k-labels.idx1-ubyte"))


# Helpers
def param(values):
    t = mt.Tensor.from_numpy(np.asarray(values, dtype=np.float32))
    t.requires_grad = True
    return t


# Model
rng = np.random.default_rng(SEED)
# He initialization: random weights scaled by sqrt(2 / fan_in); biases start at 0
W1 = param(rng.standard_normal((784, HIDDEN)) * np.sqrt(2 / 784))
b1 = param(np.zeros(HIDDEN))
W2 = param(rng.standard_normal((HIDDEN, 10)) * np.sqrt(2 / HIDDEN))
b2 = param(np.zeros(10))
params = [W1, b1, W2, b2]


def forward(x):
    hidden = mt.relu(x @ W1 + b1)
    return hidden @ W2 + b2               # raw scores (logits), one row per image


def evaluate(X, y, chunk=1000):
    """Fraction of images classified correctly. No graph is built."""
    correct = 0
    with mt_utils.no_grad():
        for i in range(0, len(X), chunk):
            logits = forward(mt.Tensor.from_numpy(X[i:i + chunk]))
            correct += int((logits.to_numpy().argmax(axis=1) == y[i:i + chunk]).sum())
    return correct / len(X)


# Training
def main():
    X_train, y_train, X_test, y_test = load_mnist()
    print(f"train: {X_train.shape}, test: {X_test.shape}")
    print(f"before training: test accuracy {evaluate(X_test, y_test) * 100:.2f}% (about 10% is chance)")

    n = len(X_train)
    for epoch in range(1, EPOCHS + 1):
        order = rng.permutation(n)          # shuffle the training set every epoch
        total_loss, num_batches = 0.0, 0
        start = time.perf_counter()

        for i in range(0, n, BATCH_SIZE):
            idx = order[i:i + BATCH_SIZE]
            xb = mt.Tensor.from_numpy(X_train[idx])
            yb = mt.Tensor.from_numpy(y_train[idx].astype(np.float32))

            for p in params:
                p.zero_grad()
            loss = mt.softmax_cross_entropy(forward(xb), yb)
            loss.backward()
            for p in params:
                mt.sgd_step(p, LR)

            total_loss += loss.item()
            num_batches += 1

        mt.synchronize()
        elapsed = time.perf_counter() - start
        test_acc = evaluate(X_test, y_test)
        print(f"epoch {epoch}: avg loss {total_loss / num_batches:.4f}, "
              f"test accuracy {test_acc * 100:.2f}%, {elapsed:.1f}s")


if __name__ == "__main__":
    main()