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
# This file implements the benchmark for AutoScheduler MatMul.
# Autoscheduler is TVM's next-generation performance tuning tool,
# which can automatically generate search spaces for optimizing tensor expressions.
# TVM is an Apache-2.0 licensed project.
# See the TVM license at: https://github.com/apache/tvm/blob/main/LICENSE
#
# ===---------------------------------------------------------------------------

import tvm
from tvm import autotvm
from tvm import te, auto_scheduler


# ------------------------------------------------------------------------------
# Template Function
# ------------------------------------------------------------------------------
@auto_scheduler.register_workload
def matmul(M, K, N, dtype):
    A = te.placeholder((M, K), name="A", dtype=dtype)
    B = te.placeholder((K, N), name="B", dtype=dtype)

    k = te.reduce_axis((0, K), name="k")
    C = te.compute(
        (M, N),
        lambda i, j: te.sum(A[i, k] * B[k, j], axis=k),
        name="C",
        attrs={
            "layout_free_placeholders": [B]
        },  # enable automatic layout transform for tensor B
    )

    return [A, B, C]


def matmul_auto_tuning_plus(args, target):
    target = tvm.target.Target(target)
    M, K, N, dtype = args
    task = tvm.auto_scheduler.SearchTask(
        func=matmul, args=(M, K, N, dtype), target=target
    )

    print("==========matmul_auto_tuning_plus=========")

    log_file = "autotvm-plus-matmul.log"
    tune_option = None
    measure_ctx = None
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=60,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        verbose=1,
    )
    # vervose to determine whether output or not
    task.tune(tune_option)
    sch, args = task.apply_best(log_file)

    return sch, args
