import tvm
import tvm.testing
from tvm import te  # tensor expression
import numpy as np

from tvm.script import tir as T
from tvm import meta_schedule as ms
from tvm.script.parser.tir import evaluate


# define computation using tvm script
def matmul_module(M, K, N):
    @tvm.script.ir_module
    class MyMatMulModule:
        @T.prim_func
        def main(
            A: T.Buffer((M, K), "float32"),
            B: T.Buffer((K, N), "float32"),
            C: T.Buffer((M, N), "float32"),
        ):
            T.func_attr({"global_symbol": "main", "tir.noalias": True})
            for i, j, k in T.grid(M, N, K):
                with T.block("C"):
                    vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                    with T.init():
                        C[vi, vj] = 0.0
                    C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

    return MyMatMulModule


def matmul_default(sizes):
    M, K, N = sizes
    my_module = matmul_module(M, K, N)
    sch = tvm.tir.Schedule(my_module)
    block_C = sch.get_block("C")
    i, j, k = sch.get_loops(block=block_C)
    i0, i1 = sch.split(i, [None, 128])
    sch.bind(i0, "blockIdx.x")
    sch.bind(i1, "threadIdx.x")
    return sch


def matmul_blocking(sizes):
    M, K, N = sizes
    tile_local_y = 8
    tile_local_x = 8
    tile_block_y = 8
    tile_block_x = 8
    tile_k = 4
    my_module = matmul_module(M, K, N)
    sch = tvm.tir.Schedule(my_module)
    block_C = sch.get_block("C")
    C_local = sch.cache_write(block_C, 0, "local")

    i, j, k = sch.get_loops(block=block_C)

    i0, i1, i2 = sch.split(loop=i, factors=[None, tile_block_y, tile_local_y])
    j0, j1, j2 = sch.split(loop=j, factors=[None, tile_block_x, tile_local_x])
    k0, k1 = sch.split(loop=k, factors=[None, tile_k])
    sch.unroll(k1)
    sch.reorder(i0, j0, i1, j1, k0, k1, i2, j2)
    sch.reverse_compute_at(C_local, j1)

    sch.bind(i0, "blockIdx.y")
    sch.bind(j0, "blockIdx.x")

    sch.bind(i1, "threadIdx.y")
    sch.bind(j1, "threadIdx.x")
    sch.decompose_reduction(block_C, k0)

    return sch


def cache_read_and_coop_fetch(sch, block, nthread, read_idx, read_loc):
    read_cache = sch.cache_read(
        block=block, read_buffer_index=read_idx, storage_scope="shared"
    )
    sch.compute_at(block=read_cache, loop=read_loc)
    # vectorized cooperative fetch
    inner0, inner1 = sch.get_loops(block=read_cache)[-2:]
    inner = sch.fuse(inner0, inner1)
    _, tx, vec = sch.split(loop=inner, factors=[None, nthread, 4])
    sch.vectorize(vec)
    sch.bind(tx, "threadIdx.x")


def matmul_blocking_with_shared(sizes):
    M, K, N = sizes
    tile_local_y = 8
    tile_local_x = 8
    tile_block_y = 8
    tile_block_x = 8
    tile_k = 8
    my_module = matmul_module(M, K, N)
    sch = tvm.tir.Schedule(my_module)

    block_C = sch.get_block("C")
    C_local = sch.cache_write(block_C, 0, "local")

    i, j, k = sch.get_loops(block=block_C)

    i0, i1, i2 = sch.split(loop=i, factors=[None, tile_block_y, tile_local_y])
    j0, j1, j2 = sch.split(loop=j, factors=[None, tile_block_x, tile_local_x])
    k0, k1 = sch.split(loop=k, factors=[None, tile_k])

    sch.reorder(i0, j0, i1, j1, k0, k1, i2, j2)
    sch.reverse_compute_at(C_local, j1)

    sch.bind(i0, "blockIdx.y")
    sch.bind(j0, "blockIdx.x")

    tx = sch.fuse(i1, j1)
    sch.bind(tx, "threadIdx.x")
    nthread = tile_block_y * tile_block_x
    cache_read_and_coop_fetch(sch, block_C, nthread, 0, k0)
    cache_read_and_coop_fetch(sch, block_C, nthread, 1, k0)
    sch.decompose_reduction(block_C, k0)

    return sch


def matmul_autoschedule(sizes):
    M, K, N = sizes

    my_module = matmul_module(M, K, N)
    database = ms.tune_tir(
        mod=my_module,
        target="nvidia/nvidia-a100",
        work_dir="./tune_tmp",
        max_trials_global=64,
        num_trials_per_iter=64,
    )

    sch_tuned = ms.tir_integration.compile_tir(database, my_module, "cuda")

    return sch_tuned
