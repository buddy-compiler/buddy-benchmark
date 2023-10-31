# ===- dep_conv_mxnet.py --------------------------------------------------------
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
# This file implements the mxnet depthwise Conv.
# This file is based on the TVM tutorial:
# https://tvm.apache.org/docs/tutorial/tensor_expr_get_started.html
# TVM is an Apache-2.0 licensed project.
# See the TVM license at: https://github.com/apache/tvm/blob/main/LICENSE
#
# ===---------------------------------------------------------------------------

import mxnet as mx
from gpu_dep_conv_manual import *
import timeit


def bench_workload(workload):
    """Benchmark a workload
    workload: a method that accept a num_repeat argument
    and return its total execution time
    """
    workload(1)  # warmup
    time = workload(1)  # the time to run once
    if time > 1:
        return time
    # The number of repeats to measure at least 1 second
    num_repeats = max(int(1.0 / time), 5)
    return workload(num_repeats) / num_repeats


def get_conv_data_mxnet(oc, ic, n, k, p, s, ctx="cpu", conv_type="direct"):
    ctx = getattr(mx, ctx)()
    data, weight, out = get_conv_data(
        oc,
        ic,
        n,
        k,
        p,
        s,
        constructor=lambda x: mx.nd.array(x, ctx=ctx),
        conv_type=conv_type,
    )
    data, out = data.expand_dims(axis=0), out.expand_dims(axis=0)
    bias = mx.nd.zeros(out.shape[1], ctx=ctx)
    return data, weight, bias, out


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


def depthwise_conv_timer_mxnet(c, n, k, ctx):
    """Benchmark convolution in MXNet
    c : input, output channels
    n : input width and height
    k : kernel width and height
    """
    timer = timeit.Timer(
        setup="import mxnet as mx\n"
        "from dep_conv_mxnet import get_conv_data_mxnet,depthwise_conv_mxnet \n"
        "c, n, k, p, s = %d, %d, %d, %d, 1\n"
        "data, weight, bias, out = get_conv_data_mxnet(\n"
        '    c, c, n, k, p, s, "%s", "%s")' % (c, n, k, (k - 1) // 2, ctx, "depthwise"),
        stmt="depthwise_conv_mxnet(data, weight, bias, out, k, p, s);"
        "out.wait_to_read()",
    )
    return timer.timeit


def bench_depthwise_conv_mxnet(size, ctx="cpu"):
    """Return the GFLOPS of MXNet convolution"""
    return bench_workload(depthwise_conv_timer_mxnet(size[0], size[1], size[2], ctx))
