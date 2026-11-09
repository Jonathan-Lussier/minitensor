import time
import numpy as np
import torch
import minitensor as mt
import mt_utils

# Settings
HIDDEN = 128
BATCH_SIZE = 64
LR = 0.075
EPOCHS = 10
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
def init_weights():
    rng = np.random.default_rng(SEED)
    W1 = (rng.standard_normal((784, HIDDEN)) * np.sqrt(2 / 784)).astype(np.float32)
    W2 = (rng.standard_normal((HIDDEN, 10)) * np.sqrt(2 / HIDDEN)).astype(np.float32)
    return rng, W1, np.zeros(HIDDEN, np.float32), W2, np.zeros(10, np.float32)


def train(name, rng, step, predict, X_train, y_train, X_test, y_test):
    """Shared loop. step(xb, yb) trains one batch and returns its loss; predict(x) returns logits as numpy."""
    for epoch in range(1, EPOCHS + 1):
        order, total_loss, num_batches = rng.permutation(len(X_train)), 0.0, 0
        start = time.perf_counter()
        for i in range(0, len(X_train), BATCH_SIZE):
            idx = order[i:i + BATCH_SIZE]
            total_loss += step(X_train[idx], y_train[idx])
            num_batches += 1
        elapsed = time.perf_counter() - start
        correct = sum(int((predict(X_test[i:i + 1000]).argmax(1) == y_test[i:i + 1000]).sum())
                      for i in range(0, len(X_test), 1000))
        print(f"{name} epoch {epoch}: avg loss {total_loss / num_batches:.4f}, "
              f"test accuracy {correct / len(X_test) * 100:.2f}%, {elapsed:.1f}s")


def train_numpy(*data):
    rng, W1, b1, W2, b2 = init_weights()

    def predict(x):
        return np.maximum(x @ W1 + b1, 0) @ W2 + b2

    def step(xb, yb):
        nonlocal W1, b1, W2, b2
        h = np.maximum(xb @ W1 + b1, 0)                         # forward
        logits = h @ W2 + b2
        probs = np.exp(logits - logits.max(1, keepdims=True))   # shifted to avoid overflow
        probs /= probs.sum(1, keepdims=True)
        rows = np.arange(len(xb))
        loss = -np.log(probs[rows, yb]).mean()

        g = probs.copy()                                        # backward: dL/dlogits
        g[rows, yb] -= 1
        g /= len(xb)
        g_h = (g @ W2.T) * (h > 0)                              # dL/d(pre-relu), ReLU gradient at 0 is 0

        W2 -= LR * (h.T @ g)
        b2 -= LR * g.sum(0)
        W1 -= LR * (xb.T @ g_h)
        b1 -= LR * g_h.sum(0)
        return float(loss)

    train("numpy", rng, step, predict, *data)


def train_pytorch(*data):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("\npyTorch device: ", device)
    rng, *weights = init_weights()
    W1, b1, W2, b2 = params = [torch.tensor(w, device=device, requires_grad=True) for w in weights]
    optimizer = torch.optim.SGD(params, LR)

    def forward(x):
        return torch.relu(x @ W1 + b1) @ W2 + b2

    @torch.no_grad()
    def predict(x):
        return forward(torch.from_numpy(x).to(device)).cpu().numpy()

    def step(xb, yb):
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(
            forward(torch.from_numpy(xb).to(device)), torch.from_numpy(yb).long().to(device))
        loss.backward()
        optimizer.step()
        return loss.item()

    train("pytorch", rng, step, predict, *data)


def main():
    #test mini tenosr
    X_train, y_train, X_test, y_test = load_mnist()
    print(f"train: {X_train.shape}, test: {X_test.shape}")
    print(f"before training: test accuracy {evaluate(X_test, y_test) * 100:.2f}% (about 10% is chance)")

    print("\nmini_tensor:")
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

    print("\nnumPy:")
    train_numpy(X_train, y_train, X_test, y_test)

    print("\npyTorch:")
    train_pytorch(X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    main()