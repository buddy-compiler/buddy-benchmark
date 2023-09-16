import tvm
import tvm.testing
from tvm import te 
import numpy as np

from tvm.script import tir as T
from tvm import meta_schedule as ms
from tvm.script.parser.tir import evaluate



def BroadcastAdd_autoschedule(shape1, shape2):
  assert len(shape1) == 2 and len(shape2) == 2, "broadcast tensors should both be 2-dimension"

  m = shape1[0] if shape2[0] == 1 else shape2[0]
  n = shape1[1] if shape2[1] == 1 else shape2[1]

  @tvm.script.ir_module
  class MyBroadcastAddModule:
    @T.prim_func
    def main(A: T.Buffer((shape1[0], shape1[1]), "float32"),
            B: T.Buffer((shape2[0], shape2[1]), "float32"),
            C: T.Buffer((m, n), "float32"),):
      T.func_attr({"global_symbol": "main", "tir.noalias": True})

  

      for i, j in T.grid(m, n):
        with T.block("C"):
          vi, vj = T.axis.remap("SS", [i, j])
          with T.init():
            C[vi, vj] = 0.0
          C[vi, vj] = C[vi, vj] + A[0 if shape1[0] == 1 else vi, 0 if shape1[1] == 1 else vj] + \
                      B[0 if shape2[0] == 1 else vi, 0 if shape2[1] == 1 else vj]

  return MyBroadcastAddModule

def broadcastAdd_autoschedule(shape1, shape2):
  my_module = BroadcastAdd_autoschedule(shape1,shape2)
  database = ms.tune_tir(
  mod=my_module,
  target="nvidia/nvidia-a100",
  work_dir="./tune_tmp",
  max_trials_global=64,
  num_trials_per_iter=64,
  )
  sch_tuned = ms.tir_integration.compile_tir(database, my_module, "cuda")

  return sch_tuned





