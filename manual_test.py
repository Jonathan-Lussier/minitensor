#A human readable test block, for sanity checks.
import numpy as np
import minitensor as mt

a = mt.Tensor.from_list([3], [5.0,12.0,3.14])
b = mt.Tensor.from_list([3], [1.0,4.0,2.72])
c = mt.Tensor.from_list([3], [2.0,10.0,-1.0])
c.requires_grad=True

d = a * b + c

print(d)