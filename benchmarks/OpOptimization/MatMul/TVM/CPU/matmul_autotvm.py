# ===- matmul_autotvm.py -------------------------------------------------------
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
# This file implements the benchmark for Auto-TVM MatMul.
# This file is based on the TVM tutorial:
# https://tvm.apache.org/docs/tutorial/autotvm_matmul_x86.html
# TVM is an Apache-2.0 licensed project.
# See the TVM license at: https://github.com/apache/tvm/blob/main/LICENSE
#
# ===---------------------------------------------------------------------------

import tvm
from tvm import te
from tvm import autotvm


# ------------------------------------------------------------------------------
# Template Function
# ------------------------------------------------------------------------------
@autotvm.template("benchmark/matmul")
def matmul(M, K, N, dtype):
    """AutoTVM MatMul Template
    Compute matrix multiplication of two matrices.
    Define search space.
    Perform scheduling according to configurations.

    Args:
      M (int): The number of rows in matrix A.
      K (int): The number of columns in matrix A and rows in matrix B.
      N (int): The number of columns in matrix B.
      dtype (str): The data type of the elements in matrices A, B, and C.

    Returns:
      tvm.te.schedule.Schedule: The schedule for the MatMul computation.
      List[tvm.te.tensor.Tensor]: The list of input and output tensors.
    """
    A = te.placeholder((M, K), name="A", dtype=dtype)
    B = te.placeholder((K, N), name="B", dtype=dtype)

    k = te.reduce_axis((0, K), name="k")
    C = te.compute((M, N), lambda x, y: te.sum(A[x, k] * B[k, y], axis=k), name="C")
    s = te.create_schedule(C.op)

    # schedule
    y, x = s[C].op.axis
    k = s[C].op.reduce_axis[0]

    ##### define space begin #####
    cfg = autotvm.get_config()
    cfg.define_split("tile_y", y, num_outputs=2)
    cfg.define_split("tile_x", x, num_outputs=2)
    ##### define space end #####

    # schedule according to config
    yo, yi = cfg["tile_y"].apply(s, C, y)
    xo, xi = cfg["tile_x"].apply(s, C, x)

    s[C].reorder(yo, xo, k, yi, xi)

    return s, [A, B, C]


# ------------------------------------------------------------------------------
# Auto-tuning
# ------------------------------------------------------------------------------
def matmul_auto_tuning(args, target):
    """AutoTVM MatMul Benchmark
    Perform auto-tuning for the MatMul template.

    Args:
      args (Tuple[int, int, int, str]):
      The arguments specifying the shape of the matrices and the data type.
        - M (int): The number of rows in matrix A.
        - K (int): The number of columns in matrix A and rows in matrix B.
        - N (int): The number of columns in matrix B.
        - dtype (str): The data type of the elements in matrices A, B, and C.
      target (str): The target device for optimization.

    Returns:
        tvm.te.schedule.Schedule: The schedule for the MatMul computation.
        List[tvm.te.tensor.Tensor]: The list of input and output tensors.
    """
    task = autotvm.task.create("benchmark/matmul", args=args, target=target)
    M, K, N, dtype = args
    measure_option = autotvm.measure_option(
        builder=autotvm.LocalBuilder(n_parallel=4, timeout=100, do_fork=True),
        runner=autotvm.LocalRunner(number=5, repeat=3, timeout=100),
    )

    tuner = autotvm.tuner.XGBTuner(task)
    log_file = open("autotvm-matmul.log", "w")
    tuner.tune(
        n_trial=10,
        measure_option=measure_option,
        callbacks=[autotvm.callback.log_to_file("autotvm-matmul.log")],
    )

    with autotvm.apply_history_best("autotvm-matmul.log"):
        with target:
            s, arg_bufs = matmul(M, K, N, "float32")

    return s, arg_bufs
