import torch
import torch.nn as nn
import numpy as np


def get_bcast_data(shape1, shape2):
    """Return random tensors a, b
    and empty tensor c to store broadcast results between a and b
    shape1, shape2: shapes of input tensors
    """
    np.random.seed(0)
    a = np.random.normal(size=shape1).astype("float32")
    b = np.random.normal(size=shape2).astype("float32")
    out_shape = (shape1[0] if shape2[0] == 1 else shape2[0],
                 shape1[1] if shape2[1] == 1 else shape2[1])
    c = np.empty(out_shape, dtype='float32')
    a = torch.from_numpy(a)
    b = torch.from_numpy(b)
    c = torch.from_numpy(c)
    return a, b, c

def broadcastAdd_torch():
    def inner_add(a,b):
        c = torch.add(a,b)
        return c
    return inner_add

def broadcastAdd_compiled():
    f = broadcastAdd_torch()
    f_compiled = torch.compile(f)
    return f_compiled



def main():
    m = 3
    n = 4
    shape1 = (m, 1)
    shape2 = (1, n)
    a, b, c = get_bcast_data(shape1, shape2)
    f = broadcastAdd_compiled()
    c = f(a, b)
    print(c)
    print(c.shape)



if __name__ == "__main__":
  main()








