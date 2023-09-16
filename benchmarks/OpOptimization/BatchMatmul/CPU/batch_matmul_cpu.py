import sys
import tvm
from tvm import topi
import numpy as np
from tvm import te, auto_scheduler


@auto_scheduler.register_workload
def batchMatmul_default(batch,M,K,N):
  A = tvm.te.placeholder((batch, M, K), name='A')
  B = tvm.te.placeholder((batch, K, N), name='B')
  k = tvm.te.reduce_axis((0, K), 'k')
  C = tvm.te.compute((batch, M, N),
          lambda b, y, x: tvm.te.sum(A[b, y, k] * B[b, k, x], axis = k),
          name = 'C')

  # schedule optimization
  s = tvm.te.create_schedule(C.op)

  return [A,B,C]


def batchMatmul_auto_tuning(shape, target):
  target = tvm.target.Target(target)
  batch, M, K, N = shape
  task = tvm.auto_scheduler.SearchTask(func=batchMatmul_default, args=(batch, M, K, N), target=target)

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
  
  return sch,args


def batchMatmul_manual(batch,M,K,N):
    A = tvm.te.placeholder((batch, M, K), name='A')
    B = tvm.te.placeholder((batch, K, N), name='B')
    k = tvm.te.reduce_axis((0, K), 'k')
    C = tvm.te.compute((batch, M, N),
    lambda b, y, x: tvm.te.sum(A[b, y, k] * B[b, k, x], axis = k),
    name = 'C')

    # schedule optimization
    s = tvm.te.create_schedule(C.op)
    bn = 32
    xo, yo, xi, yi = s[C].tile(C.op.axis[1], C.op.axis[2], bn, bn)
    (k,) = s[C].op.reduce_axis
    ko, ki = s[C].split(k, factor=4)
    # re-ordering
    s[C].reorder(xo, yo, ko, xi, ki, yi)
    s[C].vectorize(yi)
    arg_bufs = [A, B, C]
    return s, arg_bufs



def batchMatmul_numpy(shape,a_np,b_np):
  batch_size,M,K,N = shape
  c_np = np.zeros((batch_size, M, N), dtype=np.float32)
  for bs in range(batch_size):
      c_np[bs, :, :] = np.dot(a_np[bs, :, :], b_np[bs, :, :])




def main():
  target = tvm.target.Target(target="llvm", host="llvm")
  batch_size = 64
  M = 64
  K = 128
  N = 256

  a_np = np.random.rand(batch_size, M, K).astype(np.float32)
  b_np = np.random.rand(batch_size, K, N).astype(np.float32)
  c_np = np.zeros((batch_size, M, N), dtype=np.float32)
  
  a = tvm.nd.array(a_np)
  b = tvm.nd.array(b_np)
  c = tvm.nd.array(c_np)
  shape = batch_size,M,K,N

  s, arg_bufs = batchMatmul_manual(batch_size, M, K, N)

  func = tvm.build(s, arg_bufs)
  func(a, b, c)



if __name__ == '__main__':
  main()