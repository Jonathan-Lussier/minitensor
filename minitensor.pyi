"""Minimal float32 tensor library backed by CUDA."""

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

    def __repr__(self) -> str: ...
