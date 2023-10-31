# ===- main.py -----------------------------------------------------------------
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
# This file implements the TVM optimization entry for broadcastAdd on GPU.
# This file is based on the TVM tutorial:
# https://tvm.apache.org/docs/tutorial/tensor_expr_get_started.html
# TVM is an Apache-2.0 licensed project.
# See the TVM license at: https://github.com/apache/tvm/blob/main/LICENSE
#
# ===---------------------------------------------------------------------------

import tvm
import tvm.testing
from tvm import te
import numpy as np
import timeit
from tvm.script import tir as T
from tvm import meta_schedule as ms
from tvm.script.parser.tir import evaluate
from gpu_broadcast_add_autoscheduler import *
from gpu_broadcast_add_manual import *

nt = 64  # number of threads in a block
nb = 256  # number of blocks
N = 1024
target = "cuda"
dev = tvm.cuda(0)

with tvm.target.Target("cuda"):
    assert (
        nt <= tvm.target.Target.current(allow_none=False).max_num_threads
    ), "the number of threads in a block exceed the hardware limit"


def get_bcast_data(shape1, shape2, constructor=None):
    """Return random tensors a, b
    and empty tensor c to store broadcast results between a and b
    shape1, shape2: shapes of input tensors
    constructor : user-defined tensor constructor
    """
    np.random.seed(0)
    a = np.random.normal(size=shape1).astype("float32")
    b = np.random.normal(size=shape2).astype("float32")
    out_shape = (
        shape1[0] if shape2[0] == 1 else shape2[0],
        shape1[1] if shape2[1] == 1 else shape2[1],
    )
    c = np.empty(out_shape, dtype="float32")
    if constructor:
        a, b, c = [constructor(x) for x in (a, b, c)]
    return a, b, c


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
    func = tvm.build(s, vars, "cuda")
    a, b, c = inputs
    func(a, b, c)
    # Evaluate performance.
    evaluator = func.time_evaluator(func.entry_name, dev, number=100)
    mean_time = evaluator(a, b, c).mean * 1000  # Convert to milliseconds
    log.append((optimization, mean_time))


def report_performance(log):
    """Convert the log into a performance table.
    Args:
      log: The log list.
    """
    baseline = log[-1][1]
    header = (
        "Benchmark".ljust(20) + "\t" + "Time".rjust(10) + "\t" + "SpeedUp".rjust(10)
    )
    split_line = "-" * 50
    print(split_line)
    print(header)
    print(split_line)
    for result in log:
        formatted_time = "{:.8f}".format(result[1])
        formatted_performance = "{:.2f}".format(baseline / result[1])
        print(
            "\033[32m%s\033[0m\t\033[33m%s\033[0m\t\033[34m%s\033[0m"
            % (
                result[0].ljust(20),
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
    np_repeat = 100
    np_running_time = timeit.timeit(
        setup="import numpy\n"
        "N = " + str(N) + "\n"
        'dtype = "float32"\n'
        "a = numpy.random.rand(N, 1).astype(dtype)\n"
        "b = numpy.random.rand(N, N).astype(dtype)\n",
        stmt="answer = a+b",
        number=np_repeat,
    )
    shape1 = (N, 1)
    shape2 = (N, N)
    # Generate random tensor for testing.
    A_np = np.random.uniform(size=shape1).astype("float32")
    B_np = np.random.uniform(size=shape2).astype("float32")
    A_nd = tvm.nd.array(A_np, dev)
    B_nd = tvm.nd.array(B_np, dev)
    C_nd = tvm.nd.array(np.zeros((N, N), dtype="float32"), dev)
    s, arg_bufs = continuous_parallel(N)
    evaluate_operation(
        s,
        arg_bufs,
        target=target,
        inputs=(A_nd, B_nd, C_nd),
        optimization="continuous_paralle",
        log=log,
    )
    s, arg_bufs = alternate_parallel(N)
    evaluate_operation(
        s,
        arg_bufs,
        target=target,
        inputs=(A_nd, B_nd, C_nd),
        optimization="alternate_parallel",
        log=log,
    )
    s = broadcastAdd_autoschedule(shape1, shape2)
    evaluate_operation(
        s.mod,
        arg_bufs,
        target=target,
        inputs=(A_nd, B_nd, C_nd),
        optimization="autoschedule",
        log=log,
    )
    # Register numpy case.
    log.append(
        ("NUMPY_BroadcastAdd", np_running_time / np_repeat * 1000)
    )  # Milliseconds
    # Dump the performance table.
    report_performance(log)


if __name__ == "__main__":
    main()
