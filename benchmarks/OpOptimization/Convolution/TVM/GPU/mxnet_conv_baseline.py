import mxnet as mx
import timeit
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


def get_conv_data_mxnet(oc, ic, n, k, p, s, ctx="cpu"):
    ctx = getattr(mx, ctx)()
    data, weight, out = get_conv_data(
        oc, ic, n, k, p, s, lambda x: mx.nd.array(x, ctx=ctx)
    )
    data, out = data.expand_dims(axis=0), out.expand_dims(axis=0)
    bias = mx.nd.zeros(out.shape[1], ctx=ctx)
    return data, weight, bias, out


def conv_mxnet(data, weight, bias, out, k, p, s):
    mx.nd.Convolution(
        data,
        weight,
        bias,
        kernel=(k, k),
        stride=(s, s),
        pad=(p, p),
        num_filter=out.shape[1],
        out=out,
    )


def conv_timer_mxnet(c, n, k, ctx):
    """Benchmark convolution in MXNet

    c : input, output channels
    n : input width and height
    k : kernel width and height
    """
    timer = timeit.Timer(
        setup="import mxnet as mx\n"
        "from mxnet_conv_baseline import get_conv_data_mxnet,conv_mxnet,get_conv_data,conv_out_size\n"
        "c, n, k, p, s = %d, %d, %d, %d, 1\n"
        "data, weight, bias, out = get_conv_data_mxnet(\n"
        '    c, c, n, k, p, s, "%s")' % (c, n, k, (k - 1) // 2, ctx),
        stmt="conv_mxnet(data, weight, bias, out, k, p, s);" "out.wait_to_read()",
    )
    return timer.timeit


def bench_conv_mxnet(size, ctx="cpu"):
    """Return the GFLOPS of MXNet convolution"""
    return bench_workload(conv_timer_mxnet(size[0], size[1], size[2], ctx))
