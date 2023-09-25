import torch
import torch.nn as nn
import torch_mlir
import iree_torch
import io
import numpy as np

class BroadcastAdd(nn.Module):
    def __init__(self, weight):
        super(BroadcastAdd, self).__init__()
        self.weight = nn.Parameter(weight)  
    def forward(self, x):
        result = torch.add(x, self.weight)
        return result

def torch_BroadcastAdd(b_dim1, b_dim2):
    weight = torch.randn(b_dim1, b_dim2)
    model = BroadcastAdd(weight)
    return model,weight

def iree_BroadcastAdd(model, example_input):
    linalg_on_tensors_mlir = torch_mlir.compile(model,example_input,output_type="linalg-on-tensors",use_tracing=False)
    iree_backend = "llvm-cpu"
    iree_vmfb = iree_torch.compile_to_vmfb(linalg_on_tensors_mlir, iree_backend)
    invoker = iree_torch.load_vmfb(iree_vmfb, iree_backend)
    return invoker