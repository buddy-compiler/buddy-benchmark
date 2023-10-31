# ===- batch_matmul_gpu.py --------------------------------------------------------
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
# This file implements the TVM optimization for batch MatMul on GPU.
# This file is based on the TVM tutorial:
# https://tvm.apache.org/docs/tutorial/tensor_expr_get_started.html
# TVM is an Apache-2.0 licensed project.
# See the TVM license at: https://github.com/apache/tvm/blob/main/LICENSE
#
# ===---------------------------------------------------------------------------
import sys
import tvm
from tvm import topi
import numpy as np
from tvm import te, auto_scheduler


@auto_scheduler.register_workload
def batchMatmul_default(batch, M, K, N):
    A = tvm.te.placeholder((batch, M, K), name="A")
    B = tvm.te.placeholder((batch, K, N), name="B")
    k = tvm.te.reduce_axis((0, K), "k")
    C = tvm.te.compute(
        (batch, M, N),
        lambda b, y, x: tvm.te.sum(A[b, y, k] * B[b, k, x], axis=k),
        name="C",
    )
    # schedule optimization
    s = tvm.te.create_schedule(C.op)
    return [A, B, C]


def batchMatmul_auto_tuning(shape, target):
    target = tvm.target.Target(target)
    batch, M, K, N = shape
    task = tvm.auto_scheduler.SearchTask(
        func=batchMatmul_default, args=(batch, M, K, N), target=target
    )
    print("==========batchMatmul_auto_tuning=========")
    log_file = "batchMatmul_auto_tuning.log"
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


def batchMatmul_numpy(shape, a_np, b_np):
    batch_size, M, K, N = shape
    c_np = np.zeros((batch_size, M, N), dtype=np.float32)
    for bs in range(batch_size):
        c_np[bs, :, :] = np.dot(a_np[bs, :, :], b_np[bs, :, :])


def batchMatmul_manual(batch, M, K, N):
    num_thread_y = 8
    num_thread_x = 32
    vthread_y = 1
    vthread_x = 1
    A = tvm.te.placeholder((batch, M, K), name="A")
    B = tvm.te.placeholder((batch, K, N), name="B")
    k = tvm.te.reduce_axis((0, K), "k")
    C = tvm.te.compute(
        (batch, M, N),
        lambda b, y, x: tvm.te.sum(A[b, y, k] * B[b, k, x], axis=k),
        name="C",
    )
    # schedule optimization
    s = tvm.te.create_schedule(C.op)
    # thread indices
    block_y = tvm.te.thread_axis("blockIdx.y")
    block_x = tvm.te.thread_axis("blockIdx.x")
    thread_y = tvm.te.thread_axis((0, num_thread_y), "threadIdx.y")
    thread_x = tvm.te.thread_axis((0, num_thread_x), "threadIdx.x")
    thread_yz = tvm.te.thread_axis((0, vthread_y), "vthread", name="vy")
    thread_xz = tvm.te.thread_axis((0, vthread_x), "vthread", name="vx")
    # block partitioning
    BB, FF, MM = s[C].op.axis
    BBFF = s[C].fuse(BB, FF)
    by, ty_block = s[C].split(BBFF, factor=num_thread_y * vthread_y)
    bx, tx_block = s[C].split(MM, factor=num_thread_x * vthread_x)
    s[C].bind(by, block_y)
    s[C].bind(bx, block_x)
    vty, ty = s[C].split(ty_block, nparts=vthread_y)
    vtx, tx = s[C].split(tx_block, nparts=vthread_x)
    s[C].reorder(by, bx, vty, vtx, ty, tx)
    s[C].reorder(by, bx, ty, tx)
    s[C].bind(ty, thread_y)
    s[C].bind(tx, thread_x)
    s[C].bind(vty, thread_yz)
    s[C].bind(vtx, thread_xz)
    return s, (A, B, C)
