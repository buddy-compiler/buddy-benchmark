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
# This file implements the entry for benchmark Broadcast on CPU.
# Autoscheduler is TVM's next-generation performance tuning tool, 
# which can automatically generate search spaces for optimizing tensor expressions.
# TVM is an Apache-2.0 licensed project.
# See the TVM license at: https://github.com/apache/tvm/blob/main/LICENSE
#
# ===---------------------------------------------------------------------------
import numpy
import timeit
import tvm.testing
from broadcastadd_manual import *
from broadcastadd_autoschedule import *

# ------------------------------------------------------------------------------
# User Configurable Variables
# ------------------------------------------------------------------------------
# Define the size of the matrix.
# (M, K) x (K, N)
M = 1024
K = 1
N = 4096
# Choose the target for your target hardware plateform.
target = tvm.target.Target(target="llvm", host="llvm")
# target = tvm.target.Target("llvm -mcpu=core-avx2")
# target = tvm.target.Target("llvm -mcpu=skylake-avx512")
# target = tvm.target.arm_cpu("llvm -mattr=+neon")
# Define the tensor data type.
dtype = "float32"

# ------------------------------------------------------------------------------
# Helper Function
# ------------------------------------------------------------------------------
def evaluate_operation(s, vars, target, inputs, standard, optimization, log):
  """Evaluate operation correctness and print the performance information.
  Args:
    s: The schedule to be built.
    vars: The argument lists to the function.
    target: The target and option of the compilation.
    inputs: The input tensors.
    standard: The standard result for correctness evaluation.
    optimization: The name of the optimization.
    log: The log list.
  """
  func = tvm.build(s, vars, target=target)
  dev = tvm.device(target.kind.name, 0)
  a, b, c= inputs
  func(a, b, c)
  # Evaluate correctness.
  tvm.testing.assert_allclose(c.numpy(), standard, rtol=1e-5)
  # Evaluate performance.
  evaluator = func.time_evaluator(func.entry_name, dev, number=10)
  mean_time = evaluator(a, b, c).mean * 1000  # Convert to milliseconds
  log.append((optimization, mean_time))

def report_performance(log):
  """Convert the log into a performance table.
  Args:
    log: The log list.
  """
  baseline = log[0][1]
  header = "Benchmark".ljust(30) + "\t" + "Time".rjust(
      10) + "\t" + "SpeedUp".rjust(10)
  split_line = "-" * 70
  print(split_line)
  print(header)
  print(split_line)
  for result in log:
    formatted_time = "{:.2f}".format(result[1])
    formatted_performance = "{:.2f}".format(baseline / result[1])
    print("\033[32m%s\033[0m\t\033[33m%s\033[0m\t\033[34m%s\033[0m" %
          (result[0].ljust(30), str(formatted_time + " ms").rjust(10),
           str(formatted_performance).rjust(10)))

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
  # ----------------------------------------------------------------------------
  # Initialization and Baseline
  # ----------------------------------------------------------------------------
  # Initialize the log list.
  log = []
  dev = tvm.device(target.kind.name, 0)
  shape1 = (M, K)
  shape2 = (K, N)
  # Generate random tensor for testing.
  a, b, c = get_bcast_data(shape1, shape2, tvm.nd.array)
  np_repeat = 100
  np_running_time = timeit.timeit(
      setup="import numpy\n"
      "M = " + str(M) + "\n"
      "K = " + str(K) + "\n"
      "N = " + str(N) + "\n"
      'dtype = "float32"\n'
      "a = numpy.random.rand(M, K).astype(dtype)\n"
      "b = numpy.random.rand(K, N).astype(dtype)\n",
      stmt="answer = a+b",
      number=np_repeat,
  )
  standard_res = a.numpy() +  b.numpy()

  # ----------------------------------------------------------------------------
  # Register Benchmarks and Dump Report
  # ----------------------------------------------------------------------------
  # Register default schedule.
  s, arg_bufs = BroadcastAdd_default(shape1,shape2)
  evaluate_operation(s,
                      arg_bufs,
                      target=target,
                      inputs=(a, b, c),
                      standard=standard_res,
                      optimization="BroadcastAdd_default",
                      log=log)

  s, arg_bufs = BroadcastAdd_Good_Schedule(shape1,shape2)
  evaluate_operation(s,
                      arg_bufs,
                      target=target,
                      inputs=(a, b, c),
                      standard=standard_res,
                      optimization="BroadcastAdd_G_Sch",
                      log=log)

  s, arg_bufs = BroadcastAdd_Bad_Schedule(shape1,shape2)
  evaluate_operation(s,
                      arg_bufs,
                      target=target,
                      inputs=(a, b, c),
                      standard=standard_res,
                      optimization="BroadcastAdd_B_Sch",
                      log=log)

  s, arg_bufs = BroadcastAdd_auto_tuning_plus((shape1,shape2),target)
  evaluate_operation(s,
                     arg_bufs,
                     target=target,
                     inputs=(a, b, c),
                     standard=standard_res,
                     optimization="BroadcastAdd_auto_tuning_plus",
                     log=log)

  # Register numpy case.
  log.append(("NUMPY_BroadcastAdd", np_running_time / np_repeat * 1000))  # Milliseconds
  # Dump the performance table.
  report_performance(log)

if __name__ == "__main__":
  main()
