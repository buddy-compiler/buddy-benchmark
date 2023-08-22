import tvm
from tvm import te
import numpy as np
from tvm import topi
import timeit
import mxnet as mx


# ------------------------------------------------------------------------------
# Default Scheduling
# ------------------------------------------------------------------------------
# Save to the d2ltvm package.
def get_bn_data_mxnet(c, n, ctx='cpu'):
    ctx = getattr(mx, ctx)()
    data, mean, var, gamma, beta, out = get_bn_data(c, n,
                                      lambda x: mx.nd.array(x, ctx=ctx))
    data, out = data.expand_dims(axis=0), out.expand_dims(axis=0)
    return data, mean, var, gamma, beta, out


def batch_norm_mxnet(data, mean, var, gamma, beta, out, eps=1e-5):
    # use_global_stats=True to use the input mean and var instead of computing
    # the mean and var of the input data.
    # fix_gamma=False so that gamma won't be set to 1.
    mx.nd.BatchNorm(data, gamma, beta, mean, var, eps,
                    use_global_stats=True, fix_gamma=False, out=out)

def batch_norm(c, n, eps=1e-5):
    """batch normalization

    c : channels
    N : input width and height
    eps : small positive value to prevent divide 0
    """

    X = te.placeholder((c, n, n), name='X')
    Mean = te.placeholder((c, 1, 1), name='Mean')
    Var = te.placeholder((c, 1, 1), name='Var')
    Gamma = te.placeholder((c, 1, 1), name='Gamma')
    Beta = te.placeholder((c, 1, 1), name='Beta')
    C1 = topi.subtract(X,Mean)
    C2 = topi.sqrt(Var + eps)
    Y = C1 / C2 * Gamma + Beta
    return X, Mean, Var, Gamma, Beta, Y
   

def get_bn_data(c, n, constructor=None):
    """Return the batch norm data, mean, variance, gamma and beta tensors.
       Also return the empty tensor for output.

    c : channels
    n : input width and height
    constructor : user-defined tensor constructor
    """
    np.random.seed(0)
    data = np.random.normal(size=(c, n, n)).astype('float32')
    mean = np.random.normal(size=(c, 1, 1)).astype('float32')
    var = np.random.normal(loc=1.0, size=(c, 1, 1)).astype('float32')
    var = np.absolute(var)
    gamma = np.random.normal(size=(c, 1, 1)).astype('float32')
    beta = np.random.normal(size=(c, 1, 1)).astype('float32')
    out = np.empty((c, n, n), dtype='float32')
    if constructor:
        data, mean, var, gamma, beta, out = \
        (constructor(x) for x in [data, mean, var, gamma, beta, out])
    return data, mean, var, gamma, beta, out


def default_bn(size):
    c, n = size[:]
    X, Mean, Var, Gamma, Beta, Y = batch_norm(c, n)
    sch = te.create_schedule(Y.op)
    return sch, (X, Mean, Var, Gamma, Beta, Y)


def optimized_bn(size):
    sch, (X, Mean, Var, Gamma, Beta, Y) = default_bn(size)
    te.schedule.AutoInlineInjective(sch)
    c, h, w = Y.op.axis[0:3]
    fused = sch[Y].fuse(c, h)
    sch[Y].parallel(fused)
    sch[Y].vectorize(w)
    return sch, (X, Mean, Var, Gamma, Beta, Y)


def bench_workload(workload):
    """Benchmark a workload

    workload: a method that accept a num_repeat argument
    and return its total execution time
    """
    workload(1)  # warmup
    time = workload(1)  # the time to run once
    if time > 1: return time
    # The number of repeats to measure at least 1 second
    num_repeats = max(int(1.0 / time), 5)
    return workload(num_repeats) / num_repeats


def bn_timer_mxnet(c, n, ctx):
    """Benchmark batch normalization in MXNet

    c : channels
    n : input width and height
    ctx : compute ctx, e.g., cpu or gpu
    """
    timer = timeit.Timer(
        setup='import mxnet as mx\n'
        'from batchNormalization_manual import get_bn_data_mxnet\n'
        'from batchNormalization_manual import batch_norm_mxnet\n'
        'c, n = %d, %d\n'
        'data, mean, var, gamma, beta, out = get_bn_data_mxnet(\n'
        '    c, n, "%s")'%(c, n, ctx),
        stmt='batch_norm_mxnet(data, mean, var, gamma, beta, out);'
        'out.wait_to_read()')
    return timer.timeit

# Save to the d2ltvm package.
def bench_bn_mxnet(size, ctx='cpu'):
    """Return the execution times of MXNet batch norm"""
    return bench_workload(bn_timer_mxnet(size[0], size[1], ctx))




def main():
  size = (32, 28)
  mxnet_times = bench_bn_mxnet(size)
  print(mxnet_times)
#   sch, args = default_bn(size)
#   mod = tvm.build(sch, args)


#   data, mean, var, gamma, beta, out = get_bn_data(size[0], size[1], tvm.nd.array)
#   mod(data, mean, var, gamma, beta, out)


if __name__ == "__main__":
    main()





  