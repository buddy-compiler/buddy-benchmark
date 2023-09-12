import torch
import torch.nn as nn
import numpy as np


def conv_out_size(n, k, p, s):
    """Compute the output size by given input size n (width or height),
    kernel size k, padding p, and stride s
    Return output size (width or height)
    """
    return (n - k + 2 * p)//s + 1



def get_conv_data(oc, ic, n, k, p=0, s=1, constructor=None):
    """Return random 3-D data tensor, 3-D kernel tenor and empty 3-D output
    tensor with the shapes specified by input arguments.

    oc, ic : output and input channels
    n : input width and height
    k : kernel width and height
    p : padding size, default 0
    s : stride, default 1
    constructor : user-defined tensor constructor
    """
    np.random.seed(0)
    data = np.random.normal(size=(ic, n, n)).astype('float32')
    weight = np.random.normal(size=(oc, ic, k, k)).astype('float32')
    on = conv_out_size(n, k, p, s)
    out = np.empty((oc, on, on), dtype='float32')
    if constructor:
        data, weight, out = (constructor(x) for x in [data, weight, out])
    return data, weight, out



def get_pool_data_torch(c, n, k, p, s):
    data, _, out = get_conv_data(c, c, n, k, p, s,lambda x: torch.from_numpy(x))
    # data, out = data.expand_dims(axis=0), out.expand_dims(axis=0)
    data = data.unsqueeze(0)  
    out = out.unsqueeze(0)
    return data, out


def pool_torch(k, p, s):
    f = nn.MaxPool2d(k, s, p)
    return f


def pool_compiled(k,p,s):
    f = nn.MaxPool2d(k, s, p)
    f_compiled = torch.compile(f)
    return f_compiled


# def main():
#     size = (64, 64, 3)
#     c, n, k, p, s = size[0], size[0], size[1], size[2], 1
#     oc, ic, n, k, p, s = size[0], size[0], size[1], size[2], 1, 1
#     data, out_max = get_pool_data_torch(c, n, k, p, s)
#     f = pool_compiled(k,p,s)
#     f(data)


# if __name__ == "__main__":
#   main()








