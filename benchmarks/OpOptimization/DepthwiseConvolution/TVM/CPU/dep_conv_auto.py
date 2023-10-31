# ===- dep_conv_auto.py --------------------------------------------------------
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
# This file implements the TVM auto optimization for depthwise Conv on CPU.
# This file is based on the TVM tutorial:
# https://tvm.apache.org/docs/tutorial/tensor_expr_get_started.html
# TVM is an Apache-2.0 licensed project.
# See the TVM license at: https://github.com/apache/tvm/blob/main/LICENSE
#
# ===---------------------------------------------------------------------------

import numpy as np
import tvm
from tvm import te
from tvm import te, auto_scheduler
from dep_conv_manual import *


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
    rkh = te.reduce_axis((0, kh), name="rkh")
    rkw = te.reduce_axis((0, kw), name="rkw")
    # output height and weights
    oh = conv_out_size(nh, kh, ph, sh)
    ow = conv_out_size(nw, kw, pw, sw)
    # pad X and then compute Y
    X = te.placeholder((ic, nh, nw), name="X")
    K = te.placeholder((ic, 1, kh, kw), name="K")
    PaddedX = padding(X, ph, pw) if ph * pw != 0 else X
    Y = te.compute(
        (ic, oh, ow),
        lambda c, i, j: te.sum(
            (PaddedX[c, i * sh + rkh, j * sw + rkw] * K[c, 0, rkh, rkw]),
            axis=[rkh, rkw],
        ),
        name="Y",
    )
    return X, K, Y, PaddedX


def depthwise_conv_auto(c, n, k, p, s):
    target = tvm.target.Target(target="llvm", host="llvm")
    task = tvm.auto_scheduler.SearchTask(
        func=depthwise_conv_autoTem, args=(c, n, n, k, k, p, p, s, s), target=target
    )
    print("==========depthwise_conv_auto=========")
    log_file = "depthwise_conv_auto.log"
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
    X, K, Y, PaddedX = args
    return sch, (X, K, Y)
