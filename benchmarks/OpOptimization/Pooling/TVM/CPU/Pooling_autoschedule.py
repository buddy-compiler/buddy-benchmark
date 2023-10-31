# ===- Pooling_autoschedule.py -------------------------------------------------
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
# This file implements the TVM auto optimization entry for Pooling on CPU.
# This file is based on the TVM tutorial:
# https://tvm.apache.org/docs/tutorial/tensor_expr_get_started.html
# TVM is an Apache-2.0 licensed project.
# See the TVM license at: https://github.com/apache/tvm/blob/main/LICENSE
#
# ===---------------------------------------------------------------------------

import tvm
from tvm import autotvm
from tvm import te, auto_scheduler
import numpy as np
from Pooling_manual import *


# ------------------------------------------------------------------------------
# Template Function
# ------------------------------------------------------------------------------
@auto_scheduler.register_workload
def pool(pool_type, c, nh, nw, kh, kw, ph=0, pw=0, sh=1, sw=1):
    """2D pooling
    pool_type: pooling type, 'max' or 'avg'
    c : channels
    nh, nw : input width and height
    kh, kw : kernel width and height
    ph, pw : height and width padding sizes, default 0
    sh, sw : height and width strides, default 1
    """
    # reduction axes
    rkh = te.reduce_axis((0, kh), name="rkh")
    rkw = te.reduce_axis((0, kw), name="rkw")
    # output height and weights
    oh = conv_out_size(nh, kh, ph, sh)
    ow = conv_out_size(nw, kw, pw, sw)
    # pad X and then compute Y
    X = te.placeholder((c, nh, nw), name="X")

    if pool_type == "max":
        PaddedX = padding(X, ph, pw, val=te.min_value(X.dtype)) if ph * pw != 0 else X
        Y = te.compute(
            (c, oh, ow),
            lambda c, h, w: te.max(
                PaddedX[c, h * sh + rkh, w * sw + rkw], axis=[rkh, rkw]
            ),
            tag="pool_max",
            name="PoolMax",
        )
    elif pool_type == "avg":
        PaddedX = padding(X, ph, pw) if ph * pw != 0 else X
        tsum = te.compute(
            (c, oh, ow),
            lambda c, h, w: te.sum(
                PaddedX[c, h * sh + rkh, w * sw + rkw], axis=[rkh, rkw]
            ),
            tag="pool_avg1",
            name="PoolSum",
        )
        Y = te.compute(
            (c, oh, ow),
            lambda c, h, w: tsum[c, h, w] / (kh * kw),
            tag="pool_avg2",
            name="PoolAvg",
        )
    else:
        raise ValueError("Pool type should be 'avg' or 'max'.")
    return X, Y, PaddedX


def Pooling_autoschedule(size, target):
    target = tvm.target.Target(target)
    c, n, k = size[:]
    theArgs = "max", c, n, n, k, k, 1, 1, 1, 1
    task = tvm.auto_scheduler.SearchTask(func=pool, args=theArgs, target=target)
    print("==========Pooling_autoschedule=========")
    log_file = "Pooling_autoschedule.log"
    measure_ctx = None
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=120,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        verbose=1,
        early_stopping=20,
    )
    # vervose to determine whether output or not
    task.tune(tune_option)
    sch, args = task.apply_best(log_file)
    return sch, args


def default_max(size):
    c, n, k = size[:]
    X, Y, PaddedX = pool("max", c, n, n, k, k, 1, 1, 1, 1)
    sch = te.create_schedule(Y.op)
    return sch, (X, Y)


def main():
    target = tvm.target.Target(target="cuda", host="llvm")
    size = (64, 64, 3)
    c, n, k, p, s = size[0], size[0], size[1], size[2], 1
    data, _, out_max = get_conv_data(
        size[0], size[0], size[1], size[2], 1, 1, tvm.nd.array
    )
    sch, arg_bufs = Pooling_autoschedule(size, target)
    X, Y, PaddedX = arg_bufs
    arg_bufs = X, Y
    func = tvm.build(sch, arg_bufs, "cuda", target_host="llvm")
    dev = tvm.cuda(0)
    data = tvm.nd.array(data, dev)
    out_max = tvm.nd.array(out_max, dev)
    func(data, out_max)


if __name__ == "__main__":
    main()
