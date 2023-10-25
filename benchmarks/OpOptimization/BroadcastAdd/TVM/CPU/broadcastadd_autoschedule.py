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
# This file implements the auto optimization for benchmark broadcastAdd on CPU.
# Autoscheduler is TVM's next-generation performance tuning tool, 
# which can automatically generate search spaces for optimizing tensor expressions.
# TVM is an Apache-2.0 licensed project.
# See the TVM license at: https://github.com/apache/tvm/blob/main/LICENSE
#
# ===---------------------------------------------------------------------------

import tvm
from tvm import autotvm
from tvm import te, auto_scheduler
import numpy as np

# ------------------------------------------------------------------------------
# Template Function
# ------------------------------------------------------------------------------
@auto_scheduler.register_workload
def BroadcastAdd(shape1,shape2):
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
  C = te.compute((m, n),f, name='C',attrs={"layout_free_placeholders": [B]},)
  return [A, B, C]

def BroadcastAdd_auto_tuning_plus(args, target):
  target = tvm.target.Target(target)
  shape1, shape2 = args
  task = tvm.auto_scheduler.SearchTask(func=BroadcastAdd, args=(shape1,shape2), target=target)
  print("==========broadcastadd_auto_tuning_plus=========")
  log_file = "autotvm-plus-broadcastadd.log"
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
  args = shape1,shape2
  target = tvm.target.Target(target="llvm", host="llvm")
  s, arg_bufs= BroadcastAdd_auto_tuning_plus(args,target)
  A , B ,C = arg_bufs
  a, b, c = get_bcast_data(shape1, shape2, tvm.nd.array)
  mod = tvm.build(s, [A, B, C])
  mod(a, b, c)
  np.testing.assert_allclose(np.add(a.asnumpy(), b.asnumpy()), c.asnumpy(), atol=1e-5)
  print(a.shape, b.shape, c.shape)

if __name__ == "__main__":
    main()
