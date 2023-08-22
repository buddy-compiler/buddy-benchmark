import tvm
from tvm import te
import numpy as np


# ------------------------------------------------------------------------------
# Default Scheduling
# ------------------------------------------------------------------------------
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
  s = te.create_schedule(C.op)
  arg_bufs = [A, B, C]
  return s,arg_bufs


def BroadcastAdd_Good_Schedule(shape1,shape2):
  s, (A,B,C) = BroadcastAdd_default(shape1,shape2)
  x, y = C.op.axis
  s[C].parallel(x)
  s[C].vectorize(y)
  arg_bufs = (A,B,C)
  return s,arg_bufs
  
def BroadcastAdd_Bad_Schedule(shape1,shape2):
  s, (A,B,C) = BroadcastAdd_default(shape1,shape2)
  x, y = C.op.axis
  s[C].reorder(y, x)
  s[C].parallel(y)
  s[C].vectorize(x)
  arg_bufs = (A,B,C)
  return s,arg_bufs

# Save to the d2ltvm package.
def get_bcast_data(shape1, shape2, constructor=None):
    """Return random tensors a, b
    and empty tensor c to store broadcast results between a and b

    shape1, shape2: shapes of input tensors
    constructor : user-defined tensor constructor
    """
    np.random.seed(0)
    a = np.random.normal(size=shape1).astype("float32")
    b = np.random.normal(size=shape2).astype("float32")
    out_shape = (shape1[0] if shape2[0] == 1 else shape2[0],
                 shape1[1] if shape2[1] == 1 else shape2[1])
    c = np.empty(out_shape, dtype='float32')
    if constructor:
        a, b, c = [constructor(x) for x in (a, b, c)]
    return a, b, c






def main():
  m = 3
  n = 4
  shape1 = (m, 1)
  shape2 = (1, n)
  s, arg_bufs= BroadcastAdd_default(shape1,shape2)
  A , B ,C = arg_bufs
  a, b, c = get_bcast_data(shape1, shape2, tvm.nd.array)
  mod = tvm.build(s, [A, B, C])
  mod(a, b, c)
  np.testing.assert_allclose(np.add(a.asnumpy(), b.asnumpy()), c.asnumpy(), atol=1e-5)
  print(a.shape, b.shape, c.shape)

if __name__ == "__main__":
    main()





  