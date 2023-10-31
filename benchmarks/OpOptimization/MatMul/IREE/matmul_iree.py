# ===- matmul_iree.py ----------------------------------------------------------
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ===---------------------------------------------------------------------------
#
# This file implements the IREE optimization for matmul.
# you can choose run on CPU/GPU by change iree_backend = "cuda" or "llvm-cpu" in pooling_iree.py.
# See the IREE license at: https://github.com/openxla/iree/blob/main/LICENSE
#
# ===---------------------------------------------------------------------------

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
        result = torch.mm(x, self.weight)
        return result


def torch_matrix_multiply(b_dim1, b_dim2):
    weight = torch.randn(b_dim1, b_dim2)
    model = MatrixMultiplication(weight)
    return model, weight


def iree_matrix_multiply(model, example_input):
    linalg_on_tensors_mlir = torch_mlir.compile(
        model, example_input, output_type="linalg-on-tensors", use_tracing=False
    )
    iree_backend = "llvm-cpu"
    iree_vmfb = iree_torch.compile_to_vmfb(linalg_on_tensors_mlir, iree_backend)
    invoker = iree_torch.load_vmfb(iree_vmfb, iree_backend)
    return invoker
