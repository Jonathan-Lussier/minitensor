import operator
import numpy as np
import pytest
import minitensor as mt

rng = np.random.default_rng(0)

# Helpers
def scalar(value, requires_grad=True):
    """Make a 0-dimensional (shape []) tensor holding one value."""
    t = mt.Tensor.from_numpy(np.array(value, dtype=np.float32))
    t.requires_grad = requires_grad
    return t

def value_of(t):
    """Read a single-element tensor back as a Python float."""
    return float(t.to_numpy())

def numeric_grad(f, x, eps=1e-2):
    """Estimate d f(x) / dx by nudging each element of x up and down."""
    x = np.asarray(x)                     
    g = np.zeros(x.shape, dtype=np.float64)
    for i in range(x.size):
        xp = x.copy(); xp.flat[i] += eps
        xm = x.copy(); xm.flat[i] -= eps
        g.flat[i] = (f(xp) - f(xm)) / (2 * eps)
    return g

#Hand-checked scalar examples
#    Each case: (name, inputs, function, expected value, expected gradients)
#    Gradients worked out by hand with the chain rule.
SCALAR_CASES = [
    # a*b + c:  d/da = b,  d/db = a,  d/dc = 1
    ("a*b+c", {"a": 2.0, "b": 5.5, "c": 3.0},
     lambda v: v["a"] * v["b"] + v["c"],
     14.0, {"a": 5.5, "b": 2.0, "c": 1.0}),

    # (a - b) * c:  d/da = c,  d/db = -c,  d/dc = a - b
    ("(a-b)*c", {"a": 4.0, "b": 1.5, "c": 2.0},
     lambda v: (v["a"] - v["b"]) * v["c"],
     5.0, {"a": 2.0, "b": -2.0, "c": 2.5}),

    # a / b:  d/da = 1/b,  d/db = -a/b^2
    ("a/b", {"a": 3.0, "b": 4.0},
     lambda v: v["a"] / v["b"],
     0.75, {"a": 0.25, "b": -0.1875}),

    # -a * a = -a^2:  d/da = -2a   (a is used twice: tests accumulation and negation)
    ("-a*a", {"a": 3.0},
     lambda v: -v["a"] * v["a"],
     -9.0, {"a": -6.0}),

    # a*a*a = a^3:  d/da = 3a^2   (three uses of a across two nodes)
    ("a*a*a", {"a": 2.0},
     lambda v: v["a"] * v["a"] * v["a"],
     8.0, {"a": 12.0}),

    # (a + b) * (a - b) = a^2 - b^2:  d/da = 2a,  d/db = -2b
    ("(a+b)*(a-b)", {"a": 3.0, "b": 2.0},
     lambda v: (v["a"] + v["b"]) * (v["a"] - v["b"]),
     5.0, {"a": 6.0, "b": -4.0}),
]


@pytest.mark.parametrize("name,inputs,fn,expected_value,expected_grads",
                         SCALAR_CASES, ids=[c[0] for c in SCALAR_CASES])
def test_scalar_examples(name, inputs, fn, expected_value, expected_grads):
    tensors = {k: scalar(v) for k, v in inputs.items()}

    out = fn(tensors)
    assert value_of(out) == pytest.approx(expected_value)

    out.backward()
    for k, expected in expected_grads.items():
        assert tensors[k].grad is not None, f"{k}.grad was never set"
        assert value_of(tensors[k].grad) == pytest.approx(expected), f"wrong gradient for {k}"


#Small graph-behavior checks
def test_no_grad_for_inputs_that_dont_require_it():
    a = scalar(2.0)
    c = scalar(3.0, requires_grad=False)
    (a * c).backward()
    assert value_of(a.grad) == pytest.approx(3.0)
    assert c.grad is None


def test_grads_accumulate_across_backward_calls():
    # Building and backpropagating the same expression twice should double the
    # gradient, until you clear it ... have I made a zero grad??? might need to do that.
    a = scalar(2.0)
    b = scalar(5.0)
    (a * b).backward()
    (a * b).backward()
    assert value_of(a.grad) == pytest.approx(10.0)
    assert value_of(b.grad) == pytest.approx(4.0)


def test_backward_requires_single_element():
    a = mt.Tensor.from_numpy(np.ones(3, dtype=np.float32))
    a.requires_grad = True
    with pytest.raises(RuntimeError, match="single-element"):
        (a * a).backward()


#Vectors: hand-checkable formula
def test_grad_mul_add_vector():
    a_np = rng.standard_normal(5).astype(np.float32)
    b_np = rng.standard_normal(5).astype(np.float32)
    a = mt.Tensor.from_numpy(a_np) 
    a.requires_grad = True
    b = mt.Tensor.from_numpy(b_np) 
    b.requires_grad = True

    loss = mt.sum(a * b + a)
    loss.backward()

    np.testing.assert_allclose(a.grad.to_numpy(), b_np + 1, rtol=1e-6)
    np.testing.assert_allclose(b.grad.to_numpy(), a_np, rtol=1e-6)


#Vectors: numerical gradient check for every binary op, both inputs
OPS = [operator.add, operator.sub, operator.mul, operator.truediv]

@pytest.mark.parametrize("op", OPS, ids=lambda op: op.__name__)
def test_grad_numeric(op):
    a_np = rng.standard_normal(4).astype(np.float32)
    b_np = (rng.standard_normal(4) + 3.0).astype(np.float32)   # keep away from zero for division

    def loss(a_vals, b_vals):   # forward pass only, returns a float
        a = mt.Tensor.from_numpy(a_vals.astype(np.float32))
        b = mt.Tensor.from_numpy(b_vals.astype(np.float32))
        return value_of(mt.sum(op(a, b)))

    a = mt.Tensor.from_numpy(a_np); a.requires_grad = True
    b = mt.Tensor.from_numpy(b_np); b.requires_grad = True
    mt.sum(op(a, b)).backward()

    expected_a = numeric_grad(lambda x: loss(x, b_np), a_np)
    expected_b = numeric_grad(lambda x: loss(a_np, x), b_np)
    np.testing.assert_allclose(a.grad.to_numpy(), expected_a, rtol=1e-2, atol=1e-3)
    np.testing.assert_allclose(b.grad.to_numpy(), expected_b, rtol=1e-2, atol=1e-3)


BROADCAST_SHAPES = [((2, 3), (3,)), ((2, 3), (2, 1)), ((2, 1), (1, 3)), ((3,), ()), ((4, 5), (1, 1))]
@pytest.mark.parametrize("shape_a,shape_b", BROADCAST_SHAPES)
@pytest.mark.parametrize("op", OPS, ids=lambda op: op.__name__)
def test_grad_numeric_broadcast(op, shape_a, shape_b):
    a_np = rng.standard_normal(shape_a).astype(np.float32)
    b_np = (rng.standard_normal(shape_b) + 3.0).astype(np.float32)
    def loss(a_vals, b_vals):   # forward pass only, returns a float
        a = mt.Tensor.from_numpy(a_vals.astype(np.float32))
        b = mt.Tensor.from_numpy(b_vals.astype(np.float32))
        return value_of(mt.sum(op(a, b)))

    a = mt.Tensor.from_numpy(a_np); a.requires_grad = True
    b = mt.Tensor.from_numpy(b_np); b.requires_grad = True
    mt.sum(op(a, b)).backward()

    expected_a = numeric_grad(lambda x: loss(x, b_np), a_np)
    expected_b = numeric_grad(lambda x: loss(a_np, x), b_np)
    np.testing.assert_allclose(a.grad.to_numpy(), expected_a, rtol=1e-2, atol=1e-3)
    np.testing.assert_allclose(b.grad.to_numpy(), expected_b, rtol=1e-2, atol=1e-3)