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
# This file implements the entry for benchmark DepthwiseConvolution on GPU.
# Autoscheduler is TVM's next-generation performance tuning tool, 
# which can automatically generate search spaces for optimizing tensor expressions.
# TVM is an Apache-2.0 licensed project.
# See the TVM license at: https://github.com/apache/tvm/blob/main/LICENSE
#
# ===---------------------------------------------------------------------------
from gpu_dep_conv_manual import *
from gpu_dep_conv_auto import *
from dep_conv_mxnet import *
import numpy
import timeit
import tvm.testing
import mxnet as mx

dev = tvm.cuda(0)
target = 'cuda'
# ------------------------------------------------------------------------------
# Helper Function
# ------------------------------------------------------------------------------
def evaluate_operation(s, vars, target, inputs, optimization, log):
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
#   dev = tvm.device(target.kind.name, 0)
  data_x, data_k, data_y= inputs
  evaluator = func.time_evaluator(func.entry_name, dev, number=10)
  mean_time = evaluator(data_x, data_k, data_y).mean * 1000  # Convert to milliseconds
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
    formatted_time = "{:.4f}".format(result[1])
    formatted_performance = "{:.4f}".format(baseline / result[1])
    print("\033[32m%s\033[0m\t\033[33m%s\033[0m\t\033[34m%s\033[0m" %
          (result[0].ljust(30), str(formatted_time + " ms").rjust(10),
           str(formatted_performance).rjust(10)))

def main():
  # ----------------------------------------------------------------------------
  # Initialization and Baseline
  # ----------------------------------------------------------------------------
  # Initialize the log list.
  log = []
  c, n, k, p, s, tc = 512, 64, 3, 1, 1, 16
  size = c,n,k
  data_x, data_k, data_y = get_conv_data(c, c, n, k, p, s,tvm.nd.array,conv_type='depthwise')
  data_x = tvm.nd.array(data_x, dev)
  data_k = tvm.nd.array(data_k, dev)
  data_y = tvm.nd.array(data_y, dev)
  mxnet_times = bench_depthwise_conv_mxnet(size)
  # ----------------------------------------------------------------------------
  # Register Benchmarks and Dump Report
  # ----------------------------------------------------------------------------
  # Register default schedule.
  sch, arg_bufs = default_sch(c, n, k, p, s)
  evaluate_operation(sch,
                      arg_bufs,
                      target=target,
                      inputs=(data_x, data_k, data_y),
                      optimization="default_dep_conv",
                      log=log)

  sch, arg_bufs = gpu_block_schedule(c, n, k, p, s) 
  evaluate_operation(sch,
                      arg_bufs,
                      target=target,
                      inputs=(data_x, data_k, data_y),
                      optimization="gpu_block_schedule",
                      log=log)

  sch, arg_bufs = gpu_depthwise_conv_auto(c, n, k, p, s)
  evaluate_operation(sch,
                      arg_bufs,
                      target=target,
                      inputs=(data_x, data_k, data_y),
                      optimization="depthwise_conv_auto",
                      log=log)

  # Register numpy case.
  log.append(("mxnet_baseline", mxnet_times  * 1000))  # Milliseconds
  # Dump the performance table.
  report_performance(log)

if __name__ == "__main__":
  main()
