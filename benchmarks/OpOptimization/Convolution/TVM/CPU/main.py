# ===- main.py --------------------------------------------------------
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
# This file implements the TVM optimization entry for Conv on CPU.
# This file is based on the TVM tutorial:
# https://tvm.apache.org/docs/tutorial/tensor_expr_get_started.html
# TVM is an Apache-2.0 licensed project.
# See the TVM license at: https://github.com/apache/tvm/blob/main/LICENSE
#
# ===---------------------------------------------------------------------------

from mxnet_conv_baseline import *
from convolution_manual import *
from convolution_auto import *
import numpy
import timeit
import tvm.testing
import mxnet as mx
import os

os.environ["KMP_AFFINITY"] = "granularity=fine,noduplicates,compact,1,0"


# ------------------------------------------------------------------------------
# Helper Function
# ------------------------------------------------------------------------------
def evaluate_operation(s, vars, target, inputs, optimization, log):
    """Evaluate operation correctness and print the performance information.
    Args:
      s: The schedule to be built.
      vars: The argument lists to the function.
      target: The target and option of the compilation.
      inputs: The input tensors.
      standard: The standard result for correctness evaluation.
      optimization: The name of the optimization.
      log: The log list.
    """
    func = tvm.build(s, vars, target=target)
    dev = tvm.device(target.kind.name, 0)
    data_x, data_k, data_y = inputs
    evaluator = func.time_evaluator(func.entry_name, dev, number=10)
    mean_time = evaluator(data_x, data_k, data_y).mean * 1000  # Convert to milliseconds
    log.append((optimization, mean_time))


def report_performance(log):
    """Convert the log into a performance table.
    Args:
      log: The log list.
    """
    baseline = log[0][1]
    header = (
        "Benchmark".ljust(30) + "\t" + "Time".rjust(10) + "\t" + "SpeedUp".rjust(10)
    )
    split_line = "-" * 70
    print(split_line)
    print(header)
    print(split_line)
    for result in log:
        formatted_time = "{:.4f}".format(result[1])
        formatted_performance = "{:.4f}".format(baseline / result[1])
        print(
            "\033[32m%s\033[0m\t\033[33m%s\033[0m\t\033[34m%s\033[0m"
            % (
                result[0].ljust(30),
                str(formatted_time + " ms").rjust(10),
                str(formatted_performance).rjust(10),
            )
        )


def main():
    # ----------------------------------------------------------------------------
    # Initialization and Baseline
    # ----------------------------------------------------------------------------
    # Initialize the log list.
    log = []
    size = 64, 64, 3
    target = tvm.target.Target(target="llvm", host="llvm")
    dtype = "float32"
    c, n, k = size
    oc = c
    ic = c
    p = (k - 1) // 2
    s = 1
    data_x, data_k, data_y = get_conv_data(c, c, n, k, p, s, tvm.nd.array)
    ctx = getattr(mx, "cpu")()
    mxnet_times = bench_conv_mxnet(size)
    # ----------------------------------------------------------------------------
    # Register Benchmarks and Dump Report
    # ----------------------------------------------------------------------------
    # Register default schedule.
    sch, arg_bufs = default_conv(oc, ic, n, k, p, s)
    evaluate_operation(
        sch,
        arg_bufs,
        target=target,
        inputs=(data_x, data_k, data_y),
        optimization="default_conv",
        log=log,
    )

    sch, arg_bufs = blocked_cached_conv(oc, ic, n, k, p, s)
    evaluate_operation(
        sch,
        arg_bufs,
        target=target,
        inputs=(data_x, data_k, data_y),
        optimization="blocked_cached_conv",
        log=log,
    )

    sch, arg_bufs = packed_cached_conv(oc, ic, n, k, p, s)
    # print(arg_bufs)
    evaluate_operation(
        sch,
        arg_bufs,
        target=target,
        inputs=(data_x, data_k, data_y),
        optimization="packed_cached_conv",
        log=log,
    )

    sch, arg_bufs = conv_auto(oc, ic, n, k, p, s)
    evaluate_operation(
        sch,
        arg_bufs,
        target=target,
        inputs=(data_x, data_k, data_y),
        optimization="auto_conv",
        log=log,
    )

    # Register numpy case.
    log.append(("mxnet_baseline", mxnet_times * 1000))  # Milliseconds
    # Dump the performance table.
    report_performance(log)


if __name__ == "__main__":
    main()
