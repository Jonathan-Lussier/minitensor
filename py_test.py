import operator
import numpy as np
import pytest
import minitensor as mt

rng = np.random.default_rng(0)

SHAPES = [((2, 3), (2, 3)), ((2, 3), (3,)), ((2, 3), (2, 1)),
          ((2, 1), (1, 3)), ((3,), ()), ((4, 5), (1, 1))]
OPS = [operator.add, operator.sub, operator.mul, operator.truediv]

@pytest.mark.parametrize("shape_a,shape_b", SHAPES)
@pytest.mark.parametrize("op", OPS)
def test_binary_ops(shape_a, shape_b, op):
    a = np.asarray(rng.standard_normal(shape_a, dtype=np.float32))
    b = np.asarray(rng.standard_normal(shape_b, dtype=np.float32)) + 3.0  # stay away from 0 for division
    expected = op(a, b)
    result = op(mt.Tensor.from_numpy(a), mt.Tensor.from_numpy(b.astype(np.float32))).to_numpy()
    assert result.shape == expected.shape
    np.testing.assert_allclose(result, expected, rtol=1e-6)

def test_incompatible_shapes_raise():
    a = mt.Tensor.from_numpy(np.ones((2, 3), dtype=np.float32))
    b = mt.Tensor.from_numpy(np.ones((2,), dtype=np.float32))
    with pytest.raises(RuntimeError, match="incompatible shapes"): a + b

@pytest.mark.parametrize("n", [1, 7, 255, 256, 257, 1000, 1 << 20, 3_000_001])
def test_sum(n):
    a = rng.standard_normal(n).astype(np.float32)
    result = mt.Tensor.from_numpy(a).sum().to_numpy()
    expected = a.astype(np.float64).sum()       # float64 reference: closer to the true sum
    np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-3)

@pytest.mark.parametrize("n", [0, 1, 7, 255, 256, 257, 1000, 1 << 20, 3_000_001])
def test_mean(n):
    a = rng.standard_normal(n).astype(np.float32)
    result = mt.Tensor.from_numpy(a).mean().to_numpy()
    expected = a.astype(np.float64).mean()       # float64 reference: closer to the true sum
    np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-3)


@pytest.mark.parametrize('n', [0,1,2,255,256,1000, 1<<20, 4_000_001])
def test_max(n):
    a = rng.standard_normal(n).astype(np.float32)
    result = mt.Tensor.from_numpy(a).max().to_numpy()
    expected = a.astype(np.float64).max()
    np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-3)

@pytest.mark.parametrize('n', [0,1,2,255,256,1000, 1<<20, 4_000_001])
def test_min(n):
    a = rng.standard_normal(n).astype(np.float32)
    result = mt.Tensor.from_numpy(a).min().to_numpy()
    expected = a.astype(np.float64).min()
    np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-3)