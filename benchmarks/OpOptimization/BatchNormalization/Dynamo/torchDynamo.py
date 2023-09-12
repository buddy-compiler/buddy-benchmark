import torch
import torch.nn as nn
import numpy as np


def get_bn_data(c, n):
    """Return the batch norm data, mean, variance, gamma and beta tensors.
       Also return the empty tensor for output.

    c : channels
    n : input width and height
    """
    np.random.seed(0)
    data = np.random.normal(size=(c, n, n)).astype('float32')
    mean = np.random.normal(size=(c, 1, 1)).astype('float32')
    var = np.random.normal(loc=1.0, size=(c, 1, 1)).astype('float32')
    var = np.absolute(var)
    gamma = np.random.normal(size=(c, 1, 1)).astype('float32')
    beta = np.random.normal(size=(c, 1, 1)).astype('float32')
    out = np.empty((c, n, n), dtype='float32')

    data = torch.from_numpy(data)
    mean = torch.from_numpy(mean)
    var = torch.from_numpy(var)
    gamma = torch.from_numpy(gamma)
    beta = torch.from_numpy(beta)
    out = torch.from_numpy(out)
    

    return data, mean, var, gamma, beta, out


def batch_norm_torch(X, mean, var, gamma, beta, out, eps=1e-5):
    C1 = X - Mean
    C2 = torch.sqrt(Var + eps)
    Y = C1 / C2 * Gamma + Beta
    return Y


def batch_norm_default():
    def batch_norm(X, mean, var, gamma, beta, out, eps=1e-5):
        C1 = X - mean
        C2 = torch.sqrt(var + eps)
        Y = C1 / C2 * gamma + beta
        return Y
    return batch_norm

    

def batch_norm_compiled():
    f = batch_norm_default()
    f_compiled = torch.compile(f)
    return f_compiled



# def main():
#     size = (1024, 28)
#     data, mean, var, gamma, beta, out = get_bn_data(size[0], size[1])
#     Y = batch_norm_compiled()
#     Y(data, mean, var, gamma, beta, out)




# if __name__ == "__main__":
#   main()








