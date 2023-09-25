import torch
import torch.nn as nn
import torch_mlir
import iree_torch
import io
import numpy as np

class MatrixMultiplication(nn.Module):
    def __init__(self, weight):
        super(MatrixMultiplication, self).__init__()
        self.weight = nn.Parameter(weight)  
    def forward(self, x):
        result = torch.bmm(x, self.weight)
        return result

def torch_matrix_multiply(batch_num, b_dim1, b_dim2):
    weight = torch.randn(batch_num, b_dim1, b_dim2)
    model = MatrixMultiplication(weight)
    return model,weight

def iree_matrix_multiply(model, example_input):
    linalg_on_tensors_mlir = torch_mlir.compile(model,example_input,output_type="linalg-on-tensors",use_tracing=False)
    iree_backend = "llvm-cpu"
    iree_vmfb = iree_torch.compile_to_vmfb(linalg_on_tensors_mlir, iree_backend)
    invoker = iree_torch.load_vmfb(iree_vmfb, iree_backend)
    return invoker







