import torch
import torch.nn as nn
import torch_mlir
import iree_torch
import io
import numpy as np

class CustomBatchNorm(nn.Module):
    def __init__(self, mean, var, gamma, beta, eps=1e-5):
        super(CustomBatchNorm, self).__init__()
        self.mean = nn.Parameter(mean)
        self.var = nn.Parameter(var)
        self.gamma = nn.Parameter(gamma)
        self.beta = nn.Parameter(beta)
        self.eps = nn.Parameter(torch.tensor(eps, dtype=torch.float32), requires_grad=True)
    def forward(self, x):
        C1 = x - self.mean
        C2 = torch.sqrt(self.var + self.eps)
        y = C1 / C2 * self.gamma + self.beta
        return y

def torch_CustomBatchNorm():
    model = CustomBatchNorm
    return model,weight

def iree_BatchNorm(model, example_input):
    linalg_on_tensors_mlir = torch_mlir.compile(model,example_input,output_type="linalg-on-tensors",use_tracing=False)
    iree_backend = "cuda"  #"llvm-cpu"
    iree_vmfb = iree_torch.compile_to_vmfb(linalg_on_tensors_mlir, iree_backend)
    invoker = iree_torch.load_vmfb(iree_vmfb, iree_backend)
    return invoker

def get_bn_data(c, n):
    """
    Return the batch norm data, mean, variance, gamma and beta tensors.
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