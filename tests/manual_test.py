#A human readable test block, for sanity checks.
import numpy as np
import minitensor as mt
import torch as pt
#import torchvision


a = mt.Tensor.from_list([3], [5.0,12.0,3.14])
b = mt.Tensor.from_list([3], [1.0,4.0,2.72])
c = mt.Tensor.from_list([3], [2.0,10.0,-1.0])
c.requires_grad=True

d = a * b + c

print(d)

t_tensor = d.to_numpy()
print(t_tensor, type(t_tensor))


torch_t = pt.torch.from_numpy(t_tensor)
print(torch_t, type(torch_t))

print(pt.cuda.is_available())

torch_g = torch_t.to("cuda")
print(torch_g)
