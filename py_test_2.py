import numpy as np
import minitensor as mt

a = mt.Tensor.from_numpy(np.ones((2, 3), dtype=np.float32))
npb = np.array([[1, 2, 3], [4, 5, 6]])
b = mt.Tensor.from_numpy(npb)
c = mt.Tensor.to_numpy(b + a)
print(c)