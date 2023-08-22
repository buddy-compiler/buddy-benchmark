import numpy as np
import tvm
from tvm import te
from tvm import te, auto_scheduler
from dep_conv_manual import *


# def padding(X, ph, pw, val=0):
#     """Pad X with the given value in 2-D

#     ph, pw : height and width padding
#     val : padding value, default 0
#     """
#     assert len(X.shape) >= 2
#     nh, nw = X.shape[-2], X.shape[-1]
#     return te.compute(
#             (*X.shape[0:-2], nh+ph*2, nw+pw*2),
#             lambda *i: te.if_then_else(
#                 te.any(i[-2]<ph, i[-2]>=nh+ph, i[-1]<pw, i[-1]>=nw+pw),
#                 val, X[i[:-2]+(i[-2]-ph, i[-1]-pw)]),
#             name='PaddedX')


# def conv_out_size(n, k, p, s):
#     """Compute the output size by given input size n (width or height),
#     kernel size k, padding p, and stride s
#     Return output size (width or height)
#     """
#     return (n - k + 2 * p)//s + 1



# def get_conv_data(oc, ic, n, k, p=0, s=1, constructor=None, conv_type='direct'):
#     """Return random 3-D data tensor, 3-D kernel tenor and empty 3-D output
#     tensor with the shapes specified by input arguments.

#     oc, ic : output and input channels
#     n : input width and height
#     k : kernel width and height
#     p : padding size, default 0
#     s : stride, default 1
#     conv_type: either direct 2D or depthwise, default direct
#     constructor : user-defined tensor constructor
#     """
#     np.random.seed(0)
#     data = np.random.normal(size=(ic, n, n)).astype('float32')
#     ic_weight = ic
#     if conv_type == 'depthwise':
#         ic_weight = 1
#     weight = np.random.normal(size=(oc, ic_weight, k, k)).astype('float32')
#     on = conv_out_size(n, k, p, s)
#     out = np.empty((oc, on, on), dtype='float32')
#     if constructor:
#         data, weight, out = (constructor(x) for x in [data, weight, out])
#     return data, weight, out



@auto_scheduler.register_workload
def depthwise_conv_autoTem(ic, nh, nw, kh, kw, ph=0, pw=0, sh=1, sw=1):
    """Convolution

    ic : number of channels for both input and output
    nh, nw : input width and height
    kh, kw : kernel width and height
    ph, pw : height and width padding sizes, default 0
    sh, sw : height and width strides, default 1
    """
    # reduction axes
    rkh = te.reduce_axis((0, kh), name='rkh')
    rkw = te.reduce_axis((0, kw), name='rkw')
    # output height and weights
    oh = conv_out_size(nh, kh, ph, sh)
    ow = conv_out_size(nw, kw, pw, sw)
    # pad X and then compute Y
    X = te.placeholder((ic, nh, nw), name='X')
    K = te.placeholder((ic, 1, kh, kw), name='K')
    PaddedX = padding(X, ph, pw) if ph * pw != 0 else X
    Y = te.compute(
        (ic, oh, ow),
        lambda c, i, j: te.sum(
            (PaddedX[c, i*sh+rkh, j*sw+rkw] * K[c, 0, rkh, rkw]),
            axis=[rkh, rkw]), name='Y')
    # sch = te.create_schedule(Y.op)
    return X, K, Y, PaddedX
    # return sch,(X, K, Y)

def depthwise_conv_auto(c, n, k, p, s):
    # X, K, Y, PaddedX = conv_default(oc, ic, n, n, k, k, p, p, s, s)
    # s = te.create_schedule(Y.op)
    target = tvm.target.Target(target="llvm", host="llvm")

    task = tvm.auto_scheduler.SearchTask(func=depthwise_conv_autoTem, args= (c, n, n, k, k, p, p, s, s), target=target)

    print("==========depthwise_conv_auto=========")

    log_file = "depthwise_conv_auto.log"
    measure_ctx = None
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=120,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        verbose=1,
        early_stopping = 20,
        )
        # vervose to determine whether output or not
    task.tune(tune_option)
    sch, args = task.apply_best(log_file)
    X, K, Y, PaddedX = args
    return sch, (X, K, Y)

# c, n, k, p, s, tc = 256, 64, 3, 1, 1, 16
# data, weight, out = get_conv_data(c, c, n, k, p, s,tvm.nd.array, conv_type='depthwise')
# sch ,args= depthwise_conv_auto(c, n, k, p, s)
# mod = tvm.build(sch, args)
# mod(data, weight, out)



