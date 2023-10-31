# ===- torchDynamo.py ----------------------------------------------------------
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
# This file implements the dynamo optimization for benchmark Matmul on GPU.
# torchdynamo is an internal API that uses a CPython feature called the Frame Evaluation
# API to safely capture PyTorch graphs. Methods that are available externally for PyTorch
# users are surfaced through the torch.compiler namespace.
# which can automatically generate search spaces for optimizing tensor expressions.
# See the pytorch license at: https://github.com/pytorch/pytorch/blob/main/LICENSE
#
# ===---------------------------------------------------------------------------
import torch


def matrix_multiply(matrix1, matrix2):
    m, n = matrix1.size()
    n, p = matrix2.size()
    result = torch.zeros(m, p)
    for i in range(m):
        for j in range(p):
            for k in range(n):
                result[i][j] += matrix1[i][k] * matrix2[k][j]
    return result


def default_matrix_multiply():
    def inner_matrix_multiply(matrix1, matrix2):
        m, n = matrix1.size()
        n, p = matrix2.size()
        result = torch.zeros(m, p)
        for i in range(m):
            for j in range(p):
                for k in range(n):
                    result[i][j] += matrix1[i][k] * matrix2[k][j]
        return result

    return inner_matrix_multiply


def dynamo_matrix_multiply():
    compiled_mm = torch.compile(matrix_multiply, mode="max-autotune")
    return compiled_mm
