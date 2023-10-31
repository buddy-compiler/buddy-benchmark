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
# This file implements the dynamo optimization for benchmark BatchNormalization on GPU.
# torchdynamo is an internal API that uses a CPython feature called the Frame Evaluation
# API to safely capture PyTorch graphs. Methods that are available externally for PyTorch
# users are surfaced through the torch.compiler namespace.
# which can automatically generate search spaces for optimizing tensor expressions.
# See the pytorch license at: https://github.com/pytorch/pytorch/blob/main/LICENSE
#
# ===---------------------------------------------------------------------------
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
    data = np.random.normal(size=(c, n, n)).astype("float32")
    mean = np.random.normal(size=(c, 1, 1)).astype("float32")
    var = np.random.normal(loc=1.0, size=(c, 1, 1)).astype("float32")
    var = np.absolute(var)
    gamma = np.random.normal(size=(c, 1, 1)).astype("float32")
    beta = np.random.normal(size=(c, 1, 1)).astype("float32")
    out = np.empty((c, n, n), dtype="float32")
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
