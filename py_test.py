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