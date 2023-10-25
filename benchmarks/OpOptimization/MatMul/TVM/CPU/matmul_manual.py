# ===- matmul_manual.py --------------------------------------------------------
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
# This file implements the benchmark for TVM MatMul manual scheduling.
# This file is based on the TVM tutorial:
# https://tvm.apache.org/docs/tutorial/tensor_expr_get_started.html
# TVM is an Apache-2.0 licensed project.
# See the TVM license at: https://github.com/apache/tvm/blob/main/LICENSE
#
# ===---------------------------------------------------------------------------

import tvm
from tvm import te


# ------------------------------------------------------------------------------
# Default Scheduling
# ------------------------------------------------------------------------------
def matmul_default(sizes):
    """Matmul benchmark with default schedule.

    Args:
      sizes: The target workload sizes.

    Returns:
      tvm.te.schedule.Schedule: The schedule for the MatMul computation.
      List[tvm.te.tensor.Tensor]: The list of input and output tensors.
    """
    M, K, N = sizes
    # TVM Matrix Multiplication using TE
    k = te.reduce_axis((0, K), "k")
    A = te.placeholder((M, K), name="A")
    B = te.placeholder((K, N), name="B")
    C = te.compute((M, N), lambda x, y: te.sum(A[x, k] * B[k, y], axis=k), name="C")
    s = te.create_schedule(C.op)
    arg_bufs = [A, B, C]
    return s, arg_bufs


# ------------------------------------------------------------------------------
# Blocking
# ------------------------------------------------------------------------------
def matmul_blocking(sizes):
    """Matmul benchmark with blocking schedule.

    Args:
      sizes: The target workload sizes.

    Returns:
      tvm.te.schedule.Schedule: The schedule for the MatMul computation.
      List[tvm.te.tensor.Tensor]: The list of input and output tensors.
    """
    M, K, N = sizes
    # TVM Matrix Multiplication using TE
    k = te.reduce_axis((0, K), "k")
    A = te.placeholder((M, K), name="A")
    B = te.placeholder((K, N), name="B")
    C = te.compute((M, N), lambda x, y: te.sum(A[x, k] * B[k, y], axis=k), name="C")
    s = te.create_schedule(C.op)
    bn = 32
    # Blocking by loop tiling.
    xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)
    (k,) = s[C].op.reduce_axis
    ko, ki = s[C].split(k, factor=4)
    # Hoist reduction domain outside the blocking loop.
    s[C].reorder(xo, yo, ko, ki, xi, yi)
    arg_bufs = [A, B, C]
    return s, arg_bufs


# ------------------------------------------------------------------------------
# Blocking + Vectorization
# ------------------------------------------------------------------------------
def matmul_blocking_vectorization(sizes):
    """Matmul benchmark with blocking and vectorization schedule.

    Args:
      sizes: The target workload sizes.

    Returns:
      tvm.te.schedule.Schedule: The schedule for the MatMul computation.
      List[tvm.te.tensor.Tensor]: The list of input and output tensors.
    """
    M, K, N = sizes
    # TVM Matrix Multiplication using TE
    k = te.reduce_axis((0, K), "k")
    A = te.placeholder((M, K), name="A")
    B = te.placeholder((K, N), name="B")
    C = te.compute((M, N), lambda x, y: te.sum(A[x, k] * B[k, y], axis=k), name="C")
    s = te.create_schedule(C.op)
    bn = 32
    # Blocking by loop tiling.
    xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)
    (k,) = s[C].op.reduce_axis
    ko, ki = s[C].split(k, factor=4)
    # Hoist reduction domain outside the blocking loop.
    s[C].reorder(xo, yo, ko, ki, xi, yi)
    # Apply the vectorization optimization
    s[C].vectorize(yi)
    arg_bufs = [A, B, C]
    return s, arg_bufs


# ------------------------------------------------------------------------------
# Loop Permutation
# ------------------------------------------------------------------------------
def matmul_loop_permutation(sizes):
    """Matmul benchmark with loop permutation schedule.

    Args:
      sizes: The target workload sizes.

    Returns:
      tvm.te.schedule.Schedule: The schedule for the MatMul computation.
      List[tvm.te.tensor.Tensor]: The list of input and output tensors.
    """
    M, K, N = sizes
    # TVM Matrix Multiplication using TE
    k = te.reduce_axis((0, K), "k")
    A = te.placeholder((M, K), name="A")
    B = te.placeholder((K, N), name="B")
    C = te.compute((M, N), lambda x, y: te.sum(A[x, k] * B[k, y], axis=k), name="C")
    s = te.create_schedule(C.op)
    bn = 32
    xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)
    (k,) = s[C].op.reduce_axis
    ko, ki = s[C].split(k, factor=4)
    # re-ordering
    s[C].reorder(xo, yo, ko, xi, ki, yi)
    s[C].vectorize(yi)
    arg_bufs = [A, B, C]
    return s, arg_bufs


# ------------------------------------------------------------------------------
# Array Packing
# ------------------------------------------------------------------------------
def matmul_array_packing(sizes):
    """Matmul benchmark with array packing schedule.

    Args:
      sizes: The target workload sizes.

    Returns:
      tvm.te.schedule.Schedule: The schedule for the MatMul computation.
      List[tvm.te.tensor.Tensor]: The list of input and output tensors.
    """
    M, K, N = sizes
    bn = 32
    # TVM Matrix Multiplication using TE
    k = te.reduce_axis((0, K), "k")
    A = te.placeholder((M, K), name="A")
    B = te.placeholder((K, N), name="B")
    packedB = te.compute(
        (N / bn, K, bn), lambda x, y, z: B[y, x * bn + z], name="packedB"
    )
    C = te.compute(
        (M, N),
        lambda x, y: te.sum(
            A[x, k] * packedB[y // bn, k, tvm.tir.indexmod(y, bn)], axis=k
        ),
        name="C",
    )
    s = te.create_schedule(C.op)

    xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)
    (k,) = s[C].op.reduce_axis
    ko, ki = s[C].split(k, factor=4)

    s[C].reorder(xo, yo, ko, xi, ki, yi)
    s[C].vectorize(yi)

    x, y, z = s[packedB].op.axis
    s[packedB].vectorize(z)
    s[packedB].parallel(x)
    arg_bufs = [A, B, C]
    return s, arg_bufs


# ------------------------------------------------------------------------------
# Optimizing Block Writing Through Caching
# ------------------------------------------------------------------------------
def matmul_block_caching(sizes):
    """Matmul benchmark with block caching schedule.

    Args:
      sizes: The target workload sizes.

    Returns:
      tvm.te.schedule.Schedule: The schedule for the MatMul computation.
      List[tvm.te.tensor.Tensor]: The list of input and output tensors.
    """
    M, K, N = sizes
    bn = 32
    # TVM Matrix Multiplication using TE
    k = te.reduce_axis((0, K), "k")
    A = te.placeholder((M, K), name="A")
    B = te.placeholder((K, N), name="B")
    packedB = te.compute(
        (N / bn, K, bn), lambda x, y, z: B[y, x * bn + z], name="packedB"
    )
    C = te.compute(
        (M, N),
        lambda x, y: te.sum(
            A[x, k] * packedB[y // bn, k, tvm.tir.indexmod(y, bn)], axis=k
        ),
        name="C",
    )
    s = te.create_schedule(C.op)

    # Allocate write cache
    CC = s.cache_write(C, "global")

    xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)

    # Write cache is computed at yo
    s[CC].compute_at(s[C], yo)

    # New inner axes
    xc, yc = s[CC].op.axis

    (k,) = s[CC].op.reduce_axis
    ko, ki = s[CC].split(k, factor=4)
    s[CC].reorder(ko, xc, ki, yc)
    s[CC].unroll(ki)
    s[CC].vectorize(yc)

    x, y, z = s[packedB].op.axis
    s[packedB].vectorize(z)
    s[packedB].parallel(x)
    arg_bufs = [A, B, C]
    return s, arg_bufs


# ------------------------------------------------------------------------------
# Parallelization
# ------------------------------------------------------------------------------
def matmul_block_caching_parallel(sizes):
    """Matmul benchmark with block caching and parallelization schedule.

    Args:
      sizes: The target workload sizes.

    Returns:
      tvm.te.schedule.Schedule: The schedule for the MatMul computation.
      List[tvm.te.tensor.Tensor]: The list of input and output tensors.
    """
    M, K, N = sizes
    bn = 32
    # TVM Matrix Multiplication using TE
    k = te.reduce_axis((0, K), "k")
    A = te.placeholder((M, K), name="A")
    B = te.placeholder((K, N), name="B")
    packedB = te.compute(
        (N / bn, K, bn), lambda x, y, z: B[y, x * bn + z], name="packedB"
    )
    C = te.compute(
        (M, N),
        lambda x, y: te.sum(
            A[x, k] * packedB[y // bn, k, tvm.tir.indexmod(y, bn)], axis=k
        ),
        name="C",
    )
    s = te.create_schedule(C.op)

    # Allocate write cache
    CC = s.cache_write(C, "global")

    xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)

    # Write cache is computed at yo
    s[CC].compute_at(s[C], yo)

    # New inner axes
    xc, yc = s[CC].op.axis

    (k,) = s[CC].op.reduce_axis
    ko, ki = s[CC].split(k, factor=4)
    s[CC].reorder(ko, xc, ki, yc)
    s[CC].unroll(ki)
    s[CC].vectorize(yc)

    s[C].parallel(xo)

    x, y, z = s[packedB].op.axis
    s[packedB].vectorize(z)
    s[packedB].parallel(x)
    arg_bufs = [A, B, C]
    return s, arg_bufs
