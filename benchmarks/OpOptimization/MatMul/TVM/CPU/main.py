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
# This is the entry of the TVM MatMul benchmark.
#
# ===---------------------------------------------------------------------------

import numpy
import timeit
import tvm.testing
from matmul_manual import *
from matmul_autotvm import *
from matmul_autoschedule import *

# ------------------------------------------------------------------------------
# User Configurable Variables
# ------------------------------------------------------------------------------

# Define the size of the matrix.
# (M, K) x (K, N)
M = 64
N = 3136
K = 576

# Choose the target for your target hardware plateform.
target = tvm.target.Target(target="llvm", host="llvm")
# target = tvm.target.Target("llvm -mcpu=core-avx2")
# target = tvm.target.Target("llvm -mcpu=skylake-avx512")
# target = tvm.target.arm_cpu("llvm -mattr=+neon")

# Define the tensor data type.
dtype = "float32"


# ------------------------------------------------------------------------------
# Helper Function
# ------------------------------------------------------------------------------
def evaluate_operation(s, vars, target, inputs, standard, optimization, log):
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
    a, b = inputs
    c = tvm.nd.array(numpy.zeros((M, N), dtype=dtype), dev)
    func(a, b, c)
    # Evaluate correctness.
    tvm.testing.assert_allclose(c.numpy(), standard, rtol=1e-5)
    # Evaluate performance.
    evaluator = func.time_evaluator(func.entry_name, dev, number=10)
    mean_time = evaluator(a, b, c).mean * 1000  # Convert to milliseconds
    log.append((optimization, mean_time))


def report_performance(log):
    """Convert the log into a performance table.
    Args:
      log: The log list.
    """
    baseline = log[0][1]
    header = (
        "Benchmark".ljust(20) + "\t" + "Time".rjust(10) + "\t" + "SpeedUp".rjust(10)
    )
    split_line = "-" * 50
    print(split_line)
    print(header)
    print(split_line)
    for result in log:
        formatted_time = "{:.2f}".format(result[1])
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

    dev = tvm.device(target.kind.name, 0)

    # Generate random tensor for testing.
    a = tvm.nd.array(numpy.random.rand(M, K).astype(dtype), dev)
    b = tvm.nd.array(numpy.random.rand(K, N).astype(dtype), dev)

    # Repeatedly perform a matrix multiplication to get a performance baseline
    # for the default numpy implementation.
    np_repeat = 100
    np_running_time = timeit.timeit(
        setup="import numpy\n"
        "M = " + str(M) + "\n"
        "K = " + str(K) + "\n"
        "N = " + str(N) + "\n"
        'dtype = "float32"\n'
        "a = numpy.random.rand(M, K).astype(dtype)\n"
        "b = numpy.random.rand(K, N).astype(dtype)\n",
        stmt="answer = numpy.dot(a, b)",
        number=np_repeat,
    )
    standard_res = numpy.dot(a.numpy(), b.numpy())

    # ----------------------------------------------------------------------------
    # Register Benchmarks and Dump Report
    # ----------------------------------------------------------------------------
    # Register default schedule.
    s, arg_bufs = matmul_default((M, K, N))
    evaluate_operation(
        s,
        arg_bufs,
        target=target,
        inputs=(a, b),
        standard=standard_res,
        optimization="TVM_MATMUL_DEFAULT",
        log=log,
    )
    # Register blocking schedule.
    s, arg_bufs = matmul_blocking((M, K, N))
    evaluate_operation(
        s,
        arg_bufs,
        target=target,
        inputs=(a, b),
        standard=standard_res,
        optimization="TVM_MATMUL_BLK",
        log=log,
    )
    # Register blocking and vectorization schedule.
    s, arg_bufs = matmul_blocking_vectorization((M, K, N))
    evaluate_operation(
        s,
        arg_bufs,
        target=target,
        inputs=(a, b),
        standard=standard_res,
        optimization="TVM_MATMUL_BLK_VEC",
        log=log,
    )
    # Register loop permutation.
    s, arg_bufs = matmul_loop_permutation((M, K, N))
    evaluate_operation(
        s,
        arg_bufs,
        target=target,
        inputs=(a, b),
        standard=standard_res,
        optimization="TVM_MATMUL_LOOP_PER",
        log=log,
    )
    # Register array packing.
    s, arg_bufs = matmul_array_packing((M, K, N))
    evaluate_operation(
        s,
        arg_bufs,
        target=target,
        inputs=(a, b),
        standard=standard_res,
        optimization="TVM_MATMUL_ARRAY_PACK",
        log=log,
    )
    # Register block caching.
    s, arg_bufs = matmul_block_caching((M, K, N))
    evaluate_operation(
        s,
        arg_bufs,
        target=target,
        inputs=(a, b),
        standard=standard_res,
        optimization="TVM_MATMUL_BLK_CACHE",
        log=log,
    )
    # Register block caching + parallelization.
    s, arg_bufs = matmul_block_caching_parallel((M, K, N))
    evaluate_operation(
        s,
        arg_bufs,
        target=target,
        inputs=(a, b),
        standard=standard_res,
        optimization="TVM_MATMUL_CACHE_PAR",
        log=log,
    )
    # # Register Auto-TVM case.
    s, arg_bufs = matmul_auto_tuning((M, K, N, "float32"), target)
    evaluate_operation(
        s,
        arg_bufs,
        target=target,
        inputs=(a, b),
        standard=standard_res,
        optimization="TVM_MATMUL_AUTO",
        log=log,
    )

    s, arg_bufs = matmul_auto_tuning_plus((M, K, N, "float32"), target)
    evaluate_operation(
        s,
        arg_bufs,
        target=target,
        inputs=(a, b),
        standard=standard_res,
        optimization="TVM_MATMUL_AUTO_PLUS",
        log=log,
    )

    # Register numpy case.
    log.append(("NUMPY_DOT", np_running_time / np_repeat * 1000))  # Milliseconds
    # Dump the performance table.
    report_performance(log)


if __name__ == "__main__":
    main()
