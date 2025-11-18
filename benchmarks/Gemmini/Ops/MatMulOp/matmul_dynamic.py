from __future__ import annotations

from exo.platforms.gemmini import *
from exo.stdlib.scheduling import *


@proc
def matmul_on_cpu(
    N: size,
    M: size,
    K: size,
    scale: f32,
    act: bool,
    A: i8[N, K] @ DRAM,
    B: i8[K, M] @ DRAM,
    C: i8[N, M] @ DRAM,
):
    for i in seq(0, N):
        for j in seq(0, M):
            res: i32 @ DRAM
            res = 0.0
            for k in seq(0, K):
                a: i8 @ DRAM
                a = A[i, k]

                b: i8 @ DRAM
                b = B[k, j]

                a2: i32
                b2: i32
                a2 = a
                b2 = b
                res += a2 * b2

            src_tmp: i32
            src_tmp = res
            tmp_res1: f32
            # 缩放 用于将定点计算结果转换为浮点数
            acc_scale(src_tmp, tmp_res1, scale)
            tmp_res2: i8
            # 将浮点数的结果限制为i8的范围
            clamp(tmp_res1, tmp_res2)
            if act == True:
                # 如果需要激活函数，则应用ReLU
                tmp_res2 = relu(tmp_res2)
            C[i, j] = tmp_res2


def sched_matmul(
    name,
    NN,
    MM,
    KK,
):
    cpu = rename(matmul_on_cpu, f"cpu_{name}")
    cpu = cpu.partial_eval(NN, MM, KK)

    gemmini = rename(cpu, name)

    gemmini = set_memory(gemmini, "res", GEMM_ACCUM)
    gemmini = set_memory(gemmini, "a", GEMM_SCRATCH)
    gemmini = set_memory(gemmini, "b", GEMM_SCRATCH)
    """
    modify the res, a, b to GEMM's inside Mem
    """

    # Tile outer loops
    # gemmini = tile_outer_loops(gemmini)
    # print("\n=== Stage 3: After tiling outer loops ===")
    # print(gemmini)
    gemmini = divide_loop(gemmini, "i", 16, ["i", "i_in"], perfect=True)
    gemmini = old_reorder(gemmini, "i_in j")
    gemmini = divide_loop(gemmini, "j", 64, ["j", "j_in"], perfect=True)
    gemmini = divide_loop(gemmini, "j_in", 16, ["j_in_o", "j_in_i"], perfect=True)
    gemmini = old_reorder(gemmini, "j_in_o j_in_i")

    # Lift res allocations
    gemmini = old_lift_alloc(gemmini, "res : _ #0", n_lifts=2)
    gemmini = old_lift_alloc(gemmini, "res : _ #0", n_lifts=1, mode="col", size=16)
    """
    1. just declare a tensor res once instead of 
    declaring a scalar res in very loop
    2. make the res be a continuous memory
    """

    # Fission outer blocks
    # gemmini = fission_outer_blocks(gemmini)
    # print("\n=== Stage 5: After fission outer blocks ===")
    # print(gemmini)
    gemmini = old_fission_after(gemmini, "res[_] = 0.0 #0", n_lifts=3)
    gemmini = old_fission_after(gemmini, "for k in _:_ #0", n_lifts=3)
    gemmini = old_reorder(gemmini, "j_in_i j_in_o")
    gemmini = old_reorder(gemmini, "i_in k")
    gemmini = old_reorder(gemmini, "j_in_i k")
    gemmini = old_reorder(gemmini, "j_in_o k")
    # 这里将k提到最外层，Gemmini的计算是并行的，可以一次性加载多个数据并行计算

    # Fission inner blocks
    # gemmini = fission_inner_blocks(gemmini)
    # print("\n=== Stage 6: After fission inner blocks ===")
    # print(gemmini)
    gemmini = divide_loop(gemmini, "k", 64, ["ko", "k"], perfect=True)
    gemmini = divide_loop(gemmini, "k", 16, ["k", "ki"], perfect=True)
    gemmini = old_lift_alloc(gemmini, "a : i8", n_lifts=3)
    gemmini = old_lift_alloc(gemmini, "a : _ #0", n_lifts=1, mode="col")
    gemmini = old_lift_alloc(gemmini, "a : _", n_lifts=2)
    gemmini = old_reorder(gemmini, "ki j_in_o")
    gemmini = old_reorder(gemmini, "ki j_in_i")
    gemmini = old_lift_alloc(gemmini, "b : i8", n_lifts=2)
    gemmini = old_lift_alloc(gemmini, "b : i8", n_lifts=1, mode="col")
    gemmini = old_lift_alloc(gemmini, "b : _", n_lifts=3)
    gemmini = old_fission_after(gemmini, "a[_] = _", n_lifts=5)
    gemmini = old_fission_after(gemmini, "b[_] = _", n_lifts=5)
    gemmini = old_reorder(gemmini, "j_in_i i_in")
    gemmini = old_reorder(gemmini, "ki i_in")
    gemmini = old_reorder(gemmini, "k i_in")
    gemmini = old_reorder(gemmini, "j_in_i ki")
    gemmini = old_reorder(gemmini, "j_in_o ki")
    gemmini = old_reorder(gemmini, "j_in_i i_in")

    # Replace with gemmini calls
    gemmini = replace_gemmini_calls(gemmini)

    # Inline and lift config
    gemmini = inline_lift_config(gemmini)

    return cpu, gemmini


def schedule_matmul_4(NN=256, MM=256, KK=256):
        """
        Parameterized matmul_4 scheduling function
        Args:
            NN: Number of rows in the first matrix, default 256
            MM: Number of columns in the second matrix, default 256
            KK: Number of columns in the first matrix/number of rows in the second matrix, default 256
        """
        # Verify if dimensions are divisible by 16 (gemmini requirement)
        tile_size = 16
        assert NN % tile_size == 0, f"NN ({NN}) must be divisible by {tile_size}"
        assert MM % tile_size == 0, f"MM ({MM}) must be divisible by {tile_size}"
        assert KK % tile_size == 0, f"KK ({KK}) must be divisible by {tile_size}"
        
        # Calculate block parameters
        n_tiles = NN // tile_size
        m_tiles = MM // tile_size
        k_tiles = KK // tile_size

        cpu, gemmini = sched_matmul("matmul_4", NN, MM, KK)

        # Real optimization
        gemmini = old_unroll(gemmini, "ko")
        gemmini = old_lift_alloc(gemmini, "res:_")
        gemmini = simplify(gemmini)

        # Adaptive block logic - calculate divide_factor based on actual dimensions
        if n_tiles >= 4:
            divide_factor = n_tiles // 4
            gemmini = divide_loop(gemmini, "i", divide_factor, ["io", "i"], perfect=True)
            gemmini = old_lift_alloc(gemmini, "a : _", n_lifts=2)
            gemmini = old_lift_alloc(gemmini, "b : _", n_lifts=2)
        else:
            # For smaller matrices, do not perform outer block division
            gemmini = old_lift_alloc(gemmini, "a : _", n_lifts=1)
            gemmini = old_lift_alloc(gemmini, "b : _", n_lifts=1)

        # tile
        gemmini = old_lift_alloc(gemmini, "a : i8", n_lifts=1, keep_dims=False)
        gemmini = old_lift_alloc(gemmini, "b : i8", n_lifts=1, keep_dims=False)
        gemmini = old_lift_alloc(gemmini, "res : _", n_lifts=2, keep_dims=False)

        # Previously add_guard
        gemmini = simplify(gemmini)

        def do_fission(pattern, n):
            nonlocal gemmini
            gemmini = autofission(gemmini, gemmini.find(pattern).after(), n_lifts=n)

        do_fission("for j_in_o in _:_", 5)
        do_fission("do_ld_i8_block_id1(_)", 6)
        do_fission("for k in _:_", 6)
        
        # Dynamically add loops based on actual tile count
        gemmini = add_loop(gemmini, "do_ld_i8_block_id1(_)", "j", 4, guard=True)
        # Determine inner loop size based on whether there is a divide_loop
        inner_loop_size = n_tiles // 4 if n_tiles >= 4 else n_tiles
        gemmini = add_loop(gemmini, "do_ld_i8_block_id2(_)", "i", inner_loop_size, guard=True)
        
        if n_tiles >= 4:
            outer_loop_size = n_tiles // 4
            gemmini = add_loop(gemmini, "if i == 0: _", "io", 4, guard=True)
            
            # Fuse_loop cleanup - only execute when there is an outer loop
            gemmini = old_reorder(gemmini, "i io")
            gemmini = old_reorder(gemmini, "k io")
            gemmini = old_reorder(gemmini, "j io")
            gemmini = add_loop(gemmini, "for j in _:_ #0", "io", 4)
            gemmini = fuse(gemmini, "for io in _:_ #0", "for io in _:_ #1")
            gemmini = fuse(gemmini, "for io in _:_ #0", "for io in _:_ #1")
            gemmini = fuse(
                gemmini, "for io in _:_ #0", "for io in _:_ #1", unsafe_disable_check=True
            )
            gemmini = add_loop(gemmini, "for j in _:_ #0", "i", divide_factor)
            gemmini = old_reorder(gemmini, "k i")
            gemmini = old_reorder(gemmini, "j i")
            gemmini = fuse(gemmini, "for i in _:_ #0", "for i in _:_ #1")
            gemmini = fuse(gemmini, "for i in _:_ #0", "for i in _:_ #1")
            gemmini = fuse(
                gemmini, "for i in _:_ #0", "for i in _:_ #1", unsafe_disable_check=True
            )
            gemmini = fuse(gemmini, "for j in _:_ #0", "for j in _:_ #1")
            gemmini = fuse(gemmini, "for j in _:_ #0", "for j in _:_ #1")
            gemmini = fuse(gemmini, "for j in _:_ #0", "for j in _:_ #1")

        gemmini = old_unroll(gemmini, "j_in_o")
        gemmini = old_unroll(gemmini, "k")

        return cpu, gemmini

# -----------------------------------------------------------------------------
# Generic parameterized template function
# -----------------------------------------------------------------------------

def schedule_matmul_flexible(name, NN, MM, KK):
    """
    Practical parameterized matmul scheduling function
    Args:
        name: Generated kernel name
        NN: Number of rows in the first matrix
        MM: Number of columns in the second matrix
        KK: Number of columns in the first matrix/number of rows in the second matrix
    """
    # Verify if dimensions are divisible by 16 (gemmini requirement)
    tile_size = 16
    assert NN % tile_size == 0, f"NN ({NN}) must be divisible by {tile_size}"
    assert MM % tile_size == 0, f"MM ({MM}) must be divisible by {tile_size}"
    assert KK % tile_size == 0, f"KK ({KK}) must be divisible by {tile_size}"
    
    # For dimensions divisible by 64, use full gemmini optimization
    if MM % 64 == 0 and KK % 64 == 0:
        try:
            # Use original sched_matmul function (with full gemmini optimization)
            cpu, gemmini = sched_matmul(name, NN, MM, KK)
            return cpu, gemmini
        except Exception as e:
            print(f"Warning: Failed to use gemmini optimization for {name}: {e}")
            # If gemmini optimization fails, revert to CPU version
    
    # For matrices not divisible by 64, return basic CPU version
    # In these cases, avoid complex gemmini optimization to avoid complex loop structure matching issues
    print(f"Info: Using CPU-only version for {name} (MM={MM}, KK={KK} - not divisible by 64)")
    
    cpu = rename(matmul_on_cpu, f"cpu_{name}")
    cpu = cpu.partial_eval(NN, MM, KK)

    # For gemmini version, use simple basic tiling
    gemmini = rename(cpu, name)
    gemmini = set_memory(gemmini, "res", GEMM_ACCUM)
    gemmini = set_memory(gemmini, "a", GEMM_SCRATCH)
    gemmini = set_memory(gemmini, "b", GEMM_SCRATCH)

    # Very basic 16x16 tiling
    gemmini = divide_loop(gemmini, "i", 16, ["i", "i_in"], perfect=True)
    gemmini = divide_loop(gemmini, "j", 16, ["j", "j_in"], perfect=True)
    gemmini = divide_loop(gemmini, "k", 16, ["k", "ki"], perfect=True)

    # Basic memory setup - do not use complex gemmini instructions
    # Just map memory to gemmini memory space, but keep simple loop structure
    
    return cpu, gemmini


cpu_matmul_4, matmul_4 = schedule_matmul_4()
cpu_matmul_flexible, matmul_flexible = schedule_matmul_flexible("matmul_4_flexible", 64, 64, 64)

__all__ = [
    "cpu_matmul_4",
    "matmul_4",
    "cpu_matmul_flexible",
    "matmul_flexible",
]
