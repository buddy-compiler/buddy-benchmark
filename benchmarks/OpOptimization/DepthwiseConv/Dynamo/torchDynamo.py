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
# This file implements the dynamo optimization for benchmark DepthwiseConv on GPU.
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


def conv_out_size(n, k, p, s):
    """Compute the output size by given input size n (width or height),
    kernel size k, padding p, and stride s
    Return output size (width or height)
    """
    return (n - k + 2 * p) // s + 1


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
    data = np.random.normal(size=(ic, n, n)).astype("float32")
    weight = np.random.normal(size=(oc, ic, k, k)).astype("float32")
    on = conv_out_size(n, k, p, s)
    out = np.empty((oc, on, on), dtype="float32")
    if constructor:
        data, weight, out = (constructor(x) for x in [data, weight, out])
    return data, weight, out


def depthwise_conv_mxnet(data, weight, bias, out, k, p, s):
    mx.nd.Convolution(
        data,
        weight,
        bias,
        kernel=(k, k),
        stride=(s, s),
        pad=(p, p),
        num_filter=out.shape[1],
        out=out,
        num_group=weight.shape[0],
    )


def depthwise_conv_torch(data, out, weight, k, p, s):
    f = nn.Conv2d(
        data.shape[1],
        out.shape[1],
        kernel_size=k,
        stride=s,
        padding=p,
        groups=weight.shape[0],
    )
    return f


def depthwise_conv_compiled(data, out, weight, k, p, s):
    f = nn.Conv2d(
        data.shape[1],
        out.shape[1],
        kernel_size=k,
        stride=s,
        padding=p,
        groups=weight.shape[0],
    )
    f_s = torch.compile(f)
    return f_s


def get_conv_data_torch(c, n, k, p, s):
    data, weight, out = get_conv_data(c, c, n, k, p, s, lambda x: torch.from_numpy(x))
    data = data.unsqueeze(0)
    out = out.unsqueeze(0)
    return data, weight, out
