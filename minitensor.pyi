"""Minimal float32 tensor library built on CUDA. GPU only."""

from collections.abc import Sequence
from typing import Annotated, overload

import numpy
from numpy.typing import NDArray


class Tensor:
    """A float32 tensor stored on the GPU."""

    @staticmethod
    def from_list(shape: Sequence[int], values: Sequence[float]) -> Tensor:
        """Create a GPU tensor from a shape and a flat list of row-major values."""

    @staticmethod
    def from_numpy(array: Annotated[NDArray[numpy.float32], dict(order='C', device='cpu', writable=False)]) -> Tensor:
        """Create a GPU tensor by copying a C-contiguous float32 NumPy array."""

    def to_list(self) -> list[float]:
        """Copy the tensor to the CPU as a flat list of row-major values."""

    def to_numpy(self) -> NDArray[numpy.float32]:
        """
        Copy the tensor to the CPU as a float32 NumPy array with the same shape.
        """

    @property
    def shape(self) -> list[int]:
        """The tensor's dimensions as a list of ints."""

    @overload
    def __add__(self, other: Tensor) -> Tensor:
        """Elementwise add (2D broadcasting supported)."""

    @overload
    def __add__(self, scalar: float) -> Tensor:
        """Add a scalar to every element."""

    def __sub__(self, other: Tensor) -> Tensor:
        """Elementwise subtract (2D broadcasting supported)."""

    @overload
    def __mul__(self, other: Tensor) -> Tensor:
        """Elementwise multiply (2D broadcasting supported)."""

    @overload
    def __mul__(self, scalar: float) -> Tensor:
        """Multiply every element by a scalar."""

    def __truediv__(self, other: Tensor) -> Tensor:
        """Elementwise divide (2D broadcasting supported)."""

    def __neg__(self) -> Tensor:
        """Elementwise negation."""

    def __rmul__(self, scalar: float) -> Tensor:
        """Multiply every element by a scalar."""

    def __radd__(self, scalar: float) -> Tensor:
        """Add a scalar to every element."""

    def __rsub__(self, scalar: float) -> Tensor:
        """Subtract every element from a scalar."""

    def __rtruediv__(self, scalar: float) -> Tensor:
        """Divide a scalar by every element."""

    def sum(self) -> Tensor:
        """Sum of all elements, returned as a tensor."""

    def sum_axis(self, axis: int) -> Tensor:
        """
        Sum a 2D tensor along one axis, keeping it as size 1. axis=0: sum down each column, (rows, cols) -> (1, cols). axis=1: sum across each row, (rows, cols) -> (rows, 1).
        """

    def reshape(self, shape: Sequence[int]) -> Tensor:
        """
        Return a tensor with a new shape and the same number of elements. The result shares memory with the original (no copy).
        """

    def dot(self, arg: Tensor, /) -> Tensor:
        """dot product of two vectors."""

    def matmul(self, arg: Tensor, /) -> Tensor:
        """matmul of two matrices."""

    def naive_matMul(self, arg: Tensor, /) -> Tensor:
        """Naive matmul, inefficient for long and narrow matrices."""

    def tiled_matMul(self, arg: Tensor, /) -> Tensor:
        """tiled MatMul matmul, more efficient."""

    def max(self) -> Tensor:
        """Maximum of all elements, returned as a tensor."""

    def min(self) -> Tensor:
        """Minimum of all elements, returned as a tensor."""

    def mean(self) -> Tensor:
        """
        Sum of all elements, deivde by number of elements, returned as a tensor.
        """

    def transpose(self) -> Tensor:
        """Transpose tensor t"""

    def exp(self) -> Tensor:
        """Elementwise e^x."""

    def log(self) -> Tensor:
        """Elementwise natural log."""

    def __repr__(self) -> str: ...

    def __str__(self) -> str: ...

    def __array__(self, *args, **kwargs) -> object: ...

    def item(self) -> float: ...

    @property
    def requires_grad(self) -> bool: ...

    @requires_grad.setter
    def requires_grad(self, arg: bool, /) -> None: ...

    @property
    def grad(self) -> Tensor | None: ...

    def backward(self) -> None: ...

    def zero_grad(self) -> None:
        """Clear this tensor's gradient (sets grad to None)."""

    def relu(self) -> Tensor:
        """Elementwise max(x, 0)."""

    def softmax(self) -> Tensor:
        """Row-wise softmax of a 2D tensor shaped (batch, classes)"""

    def softmax_cross_entropy(self, arg: Tensor, /) -> Tensor:
        """Mean softmax cross-entropy loss over a batch."""

    def __matmul__(self, other: Tensor) -> Tensor:
        """Matrix multiplication with gradient support: a @ b."""

def sum(a: Tensor) -> Tensor:
    """Sum of all elements of a tensor."""

def sum_axis(a: Tensor, axis: int) -> Tensor:
    """
    Sum a 2D tensor along one axis, keeping it as size 1 (see Tensor.sum_axis).
    """

def reshape(a: Tensor, shape: Sequence[int]) -> Tensor:
    """Reshape a tensor without copying its data (see Tensor.reshape)."""

def max(a: Tensor) -> Tensor:
    """Maximum of all elements of a tensor."""

def min(a: Tensor) -> Tensor:
    """Minimum of all elements of a tensor."""

def mean(a: Tensor) -> Tensor:
    """mean of all elements of a tensor."""

def dot(a: Tensor, b: Tensor) -> Tensor:
    """dot product of two vectors."""

@overload
def matmul(a: Tensor, b: Tensor) -> Tensor:
    """default matrix multiplication of 2D tensors a and b."""

@overload
def matmul(a: Tensor, b: Tensor) -> Tensor:
    """Matrix multiplication with gradient support."""

def naive_matMul(a: Tensor, b: Tensor) -> Tensor:
    """naive matrix multiplication of 2D tensors a and b."""

def tiled_matMul(a: Tensor, b: Tensor) -> Tensor:
    """tiled matrix multiplication of 2D tensors a and b."""

def transpose(a: Tensor) -> Tensor:
    """simple transpose of tensor a."""

def exp(a: Tensor) -> Tensor:
    """Elementwise e^x."""

def log(a: Tensor) -> Tensor:
    """Elementwise natural log."""

def synchronize() -> None: ...

def relu(a: Tensor) -> Tensor:
    """Elementwise max(x, 0)."""

def softmax(x: Tensor) -> Tensor:
    """Row-wise softmax of a 2D tensor shaped (batch, classes)"""

def softmax_cross_entropy(logits: Tensor, labels: Tensor) -> Tensor:
    """
    Mean softmax cross-entropy loss over a batch.
    A seperate softmax call is NOT needed.
    """

def sgd_step(param: Tensor, lr: float) -> None:
    """
    In-place SGD update: param -= lr * param.grad. Does nothing if param has no gradient.
    """

def set_grad_enabled(enabled: bool) -> None:
    """Turn graph recording on or off globally."""

def is_grad_enabled() -> bool:
    """Return whether graph recording is currently on."""
