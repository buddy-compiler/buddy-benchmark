import tvm
import tvm.testing
from tvm import te # tensor expression
import numpy as np

from tvm.script import tir as T
from tvm import meta_schedule as ms
from tvm.script.parser.tir import evaluate


nt = 1024  # number of threads in a block
nb = 1728 # number of blocks

with tvm.target.Target('cuda'):
    assert nt <= tvm.target.Target.current(allow_none=False).max_num_threads, \
        'the number of threads in a block exceed the hardware limit'




def BroadcastAdd_default(shape1,shape2):
  M, K = shape1
  K, N = shape2
  assert len(shape1) == 2 and len(shape2) == 2, "broadcast tensors should both be 2-dimension"
  for i in range(len(shape1)):
    assert shape1[i] == shape2[i] or shape1[i] == 1 or shape2[i] == 1,"tensor shapes do not fit for broadcasting"

  A = te.placeholder(shape1, name='A')
  B = te.placeholder(shape2, name='B')
  m = shape1[0] if shape2[0] == 1 else shape2[0]
  n = shape1[1] if shape2[1] == 1 else shape2[1]
  f = lambda x, y: A[0 if shape1[0]==1 else x, 0 if shape1[1]==1 else y] + \
      B[0 if shape2[0]==1 else x, 0 if shape2[1]==1 else y]
  C = te.compute((m, n), f, name='C')
  # s = te.create_schedule(C.op)
  # arg_bufs = [A, B, C]
  return [A,B,C]


def continuous_parallel(n):
    A, B, C = BroadcastAdd_default((n,1), (n,n))
    total_size = n * n
    need_further_split = total_size > nb * nt
    s = te.create_schedule(C.op)
    x, y = C.op.axis
    fused = s[C].fuse(x, y)
    if need_further_split:
        bx, tx = s[C].split(fused, nparts=nb)
        tx, xi = s[C].split(tx, nparts=nt)
        s[C].bind(bx, te.thread_axis("blockIdx.x"))
        s[C].bind(tx, te.thread_axis("threadIdx.x"))
    else:
        bx, tx = s[C].split(fused, factor=nt)
        s[C].bind(bx, te.thread_axis("blockIdx.x"))
        s[C].bind(tx, te.thread_axis("threadIdx.x"))
    return s, (A, B, C)


def alternate_parallel(n):
    A, B, C = BroadcastAdd_default((n,1), (n,n))
    total_size = n * n
    need_further_split = total_size > nb * nt
    s = te.create_schedule(C.op)
    x, y = C.op.axis
    fused = s[C].fuse(x, y)
    if need_further_split:
        xo, xi = s[C].split(fused, factor=nb * nt)
        bx, tx = s[C].split(xi, factor=nt)
        # bring the outermost axis to the innermost
        # for alternate data access of a CUDA thread
        s[C].reorder(bx, tx, xo)
        s[C].bind(bx, te.thread_axis("blockIdx.x"))
        s[C].bind(tx, te.thread_axis("threadIdx.x"))
    else:
        bx, tx = s[C].split(fused, factor=nt)
        s[C].bind(bx, te.thread_axis("blockIdx.x"))
        s[C].bind(tx, te.thread_axis("threadIdx.x"))
    return s, (A, B, C)



dev = tvm.cuda(0)
A_np = np.random.uniform(size=(256, 1)).astype("float32")
B_np = np.random.uniform(size=(256, 256)).astype("float32")

A_nd = tvm.nd.array(A_np, dev)
B_nd = tvm.nd.array(B_np, dev)
C_nd = tvm.nd.array(np.zeros((256, 256), dtype="float32"), dev)

s, args = continuous_parallel(256)
tvm.lower(s, args, simple_mode=True)
mod = tvm.build(s, args, 'cuda')
mod(A_nd,B_nd,C_nd)
