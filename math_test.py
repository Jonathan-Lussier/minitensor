import operator
import numpy as np
import pytest
import minitensor as mt

rng = np.random.default_rng(0)

SHAPES = [((2, 3), (2, 3)), ((2, 3), (3,)), ((2, 3), (2, 1)),
          ((2, 1), (1, 3)), ((3,), ()), ((4, 5), (1, 1))]
OPS = [operator.add, operator.sub, operator.mul, operator.truediv]

shape = [(3,3)]
@pytest.mark.parametrize("shape", shape)
def test_overloads(shape):
    anp = np.asarray(rng.standard_normal(shape, dtype=np.float32))
    bnp = np.asarray(rng.standard_normal(shape, dtype=np.float32)) 
    cnp = np.asarray(rng.standard_normal(shape, dtype=np.float32)) 
    a = mt.Tensor.from_numpy(anp)
    b = mt.Tensor.from_numpy(bnp)
    c = mt.Tensor.from_numpy(cnp)
    out = a + b * c
    result = out.to_numpy()
    expected = anp + bnp * cnp
    assert result.shape == expected.shape
    np.testing.assert_allclose(result, expected, rtol=1e-6)

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

@pytest.mark.parametrize("n", [1, 7, 255, 256, 257, 1000, 1 << 20, 3_000_001])
def test_mean(n):
    a = rng.standard_normal(n).astype(np.float32)
    result = mt.Tensor.from_numpy(a).mean().to_numpy()
    expected = a.astype(np.float64).mean()       # float64 reference: closer to the true sum
    np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-3)


@pytest.mark.parametrize('n', [1,2,255,256,1000, 1<<20, 4_000_001])
def test_max(n):
    a = rng.standard_normal(n).astype(np.float32)
    result = mt.Tensor.from_numpy(a).max().to_numpy()
    expected = a.astype(np.float64).max()
    np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-3)

@pytest.mark.parametrize('n', [1,2,255,256,1000, 1<<20, 4_000_001])
def test_min(n):
    a = rng.standard_normal(n).astype(np.float32)
    result = mt.Tensor.from_numpy(a).min().to_numpy()
    expected = a.astype(np.float64).min()
    np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-3)

MATMUL_SHAPES = [((2, 2), (2, 3)), ((10, 10), (10, 5)), ((2, 1), (1, 3)),
                 ((1, 1), (1, 1)), ((3, 300), (300, 2)), ((64, 33), (33, 129))]

@pytest.mark.parametrize("shape_a,shape_b", MATMUL_SHAPES)
def test_naiveMatMul(shape_a, shape_b):
    a = rng.standard_normal(shape_a, dtype=np.float32)
    b = rng.standard_normal(shape_b, dtype=np.float32)
    result = mt.naiveMatMul(mt.Tensor.from_numpy(a), mt.Tensor.from_numpy(b)).to_numpy()
    expected = a @ b
    assert result.shape == expected.shape
    np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-3)

@pytest.mark.parametrize("shape_a,shape_b", [((2, 3), (2, 1)), ((4, 5), (1, 1)),
                                             ((2, 3), (3,)), ((3,), ())])
def test_naiveMatMul_bad_shapes_raise(shape_a, shape_b):
    a = mt.Tensor.from_numpy(np.ones(shape_a, dtype=np.float32))
    b = mt.Tensor.from_numpy(np.ones(shape_b, dtype=np.float32))
    with pytest.raises(RuntimeError):
        mt.naiveMatMul(a, b)

DOT_SHAPES = [(1,1), (2,2), (10,10), (255,255), (256,256), (4_000_001, 4_000_001)]
@pytest.mark.parametrize("shape_a,shape_b", DOT_SHAPES)
def test_dot(shape_a, shape_b):
    a = rng.standard_normal(shape_a, dtype=np.float32)
    b = rng.standard_normal(shape_b, dtype=np.float32)
    result = mt.dot(mt.Tensor.from_numpy(a), mt.Tensor.from_numpy(b)).to_numpy()
    expected = np.dot(a,b)
    assert result.shape == expected.shape
    np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-3)

@pytest.mark.parametrize("shape_a,shape_b", [(2,3), (3,2), (0,0)])
def test_dot_bad_shapes(shape_a, shape_b):
    a = mt.Tensor.from_numpy(np.ones(shape_a, dtype=np.float32))
    b = mt.Tensor.from_numpy(np.ones(shape_b, dtype=np.float32))
    with pytest.raises(RuntimeError):
        mt.dot(a, b)

@pytest.mark.parametrize("shape_a,shape_b", MATMUL_SHAPES)
def test_tiledMatMul(shape_a, shape_b):
    a = rng.standard_normal(shape_a, dtype=np.float32)
    b = rng.standard_normal(shape_b, dtype=np.float32)
    result = mt.tiledMatMul(mt.Tensor.from_numpy(a), mt.Tensor.from_numpy(b)).to_numpy()
    expected = a @ b
    assert result.shape == expected.shape
    np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-3)

@pytest.mark.parametrize("shape", [(1, 1), (3, 5), (64, 10), (7, 1000), (1000, 7)])
@pytest.mark.parametrize("axis", [0, 1])
def test_sum_axis(shape, axis):
    x = rng.standard_normal(shape).astype(np.float32)
    result = mt.sumAxis(mt.Tensor.from_numpy(x), axis).to_numpy()
    expected = x.sum(axis=axis, keepdims=True)
    assert result.shape == expected.shape
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)