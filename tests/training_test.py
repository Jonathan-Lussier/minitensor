"""
Training tests for minitensor: SGD on small problems with known answers.

Run with:
    PYTHONPATH=build pytest -v test_training.py

Each test is a bigger step toward MNIST:
  1. sgd_step / zero_grad mechanics
  2. minimize a simple function (the full training loop, no matmul)
  3. linear regression (matmul + bias broadcasting gradients)
  4. a 2-layer MLP on XOR (relu + cross_entropy: a miniature MNIST)
"""
import numpy as np
import pytest

import minitensor as mt


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def param(values):
    """A trainable tensor (requires_grad = True)."""
    t = mt.Tensor.from_numpy(np.asarray(values, dtype=np.float32))
    t.requires_grad = True
    return t


def const(values):
    """A fixed tensor (no gradient): inputs, targets, constants."""
    return mt.Tensor.from_numpy(np.asarray(values, dtype=np.float32))


def train_step(params, compute_loss, lr):
    """One step: clear grads, forward, backward, update. Returns the loss value."""
    for p in params:
        p.zero_grad()
    loss = compute_loss()
    loss.backward()
    for p in params:
        mt.sgd_step(p, lr)
    return loss.item()


# ---------------------------------------------------------------------
# 1. Mechanics
# ---------------------------------------------------------------------
def test_sgd_step_applies_update_in_place():
    p_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    w_np = np.array([0.5, -1.0, 2.0], dtype=np.float32)
    p = param(p_np)

    mt.sum(p * const(w_np)).backward()     # gradient of sum(p * w) is w
    mt.sgd_step(p, 0.1)

    np.testing.assert_allclose(p.to_numpy(), p_np - 0.1 * w_np, rtol=1e-6)
    assert p.requires_grad                 # still a trainable parameter


def test_sgd_step_without_grad_does_nothing():
    p = param([1.0, 2.0, 3.0])
    mt.sgd_step(p, 0.1)                    # no backward() yet, so no gradient
    np.testing.assert_array_equal(p.to_numpy(), [1.0, 2.0, 3.0])


def test_zero_grad_clears_gradient():
    p = param([1.0, 2.0])
    mt.sum(p * p).backward()
    assert p.grad is not None
    p.zero_grad()
    assert p.grad is None


# ---------------------------------------------------------------------
# 2. Minimize a simple function: sum((p - target)^2)
#    The minimum is p = target. Gradient is 2(p - target), so each step with
#    lr = 0.1 shrinks the error by a factor of 0.8.
# ---------------------------------------------------------------------
def test_sgd_minimizes_quadratic():
    target_np = np.array([3.0, -2.0, 0.5], dtype=np.float32)
    target = const(target_np)
    p = param(np.zeros(3))

    def loss_fn():
        diff = p - target
        return mt.sum(diff * diff)

    losses = [train_step([p], loss_fn, lr=0.1) for _ in range(200)]

    assert losses[-1] < losses[0] * 1e-6, "loss should shrink to almost nothing"
    np.testing.assert_allclose(p.to_numpy(), target_np, atol=1e-4)


# ---------------------------------------------------------------------
# 3. Linear regression: learn W and b from y = X @ W_true + b_true
#    Exercises matmul's backward and the bias broadcast's unbroadcast.
# ---------------------------------------------------------------------
def test_sgd_learns_linear_regression():
    rng = np.random.default_rng(0)
    n = 64
    X_np = rng.standard_normal((n, 3)).astype(np.float32)
    W_true = np.array([[2.0], [-1.0], [0.5]], dtype=np.float32)
    b_true = np.array([0.3], dtype=np.float32)
    y_np = X_np @ W_true + b_true                       # (64, 1)

    X, y, inv_n = const(X_np), const(y_np), const(1.0 / n)
    W = param(np.zeros((3, 1)))
    b = param(np.zeros(1))                              # broadcasts against (64, 1)

    def loss_fn():                                      # mean squared error
        diff = (X @ W + b) - y
        return mt.sum(diff * diff) * inv_n

    losses = [train_step([W, b], loss_fn, lr=0.1) for _ in range(500)]

    assert losses[-1] < 1e-6
    np.testing.assert_allclose(W.to_numpy(), W_true, atol=1e-3)
    np.testing.assert_allclose(b.to_numpy(), b_true, atol=1e-3)


# ---------------------------------------------------------------------
# 4. A 2-layer MLP on XOR: a miniature version of the MNIST classifier
#    XOR can't be solved by a single linear layer, so this only passes if
#    the hidden layer (matmul + bias + relu) and cross_entropy all work.
# ---------------------------------------------------------------------
def test_mlp_learns_xor():
    rng = np.random.default_rng(0)
    X_np = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    y_np = np.array([0, 1, 1, 0])
    X, labels = const(X_np), const(y_np.astype(np.float32))

    hidden = 32
    # He initialization: random weights scaled by sqrt(2 / fan_in); biases start at 0
    W1 = param(rng.standard_normal((2, hidden)) * np.sqrt(2 / 2))
    b1 = param(np.zeros(hidden))
    W2 = param(rng.standard_normal((hidden, 2)) * np.sqrt(2 / hidden))
    b2 = param(np.zeros(2))
    params = [W1, b1, W2, b2]

    def forward():
        h = mt.relu(X @ W1 + b1)
        return h @ W2 + b2

    def loss_fn():
        return mt.softmax_cross_entropy(forward(), labels)

    losses = [train_step(params, loss_fn, lr=0.5) for _ in range(2000)]

    predictions = forward().to_numpy().argmax(axis=1)
    assert losses[-1] < 0.01, f"loss only reached {losses[-1]:.4f}"
    np.testing.assert_array_equal(predictions, y_np)