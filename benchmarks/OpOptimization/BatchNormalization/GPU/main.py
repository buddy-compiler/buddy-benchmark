# ===- main.py -----------------------------------------------------------------
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
# This is the entry of the TVM MatMul benchmark.
#
# ===---------------------------------------------------------------------------

import numpy
import timeit
import tvm.testing
import mxnet as mx

from gpu_batch_norm_auto import *
from gpu_batch_norm_manual import *



# ------------------------------------------------------------------------------
# User Configurable Variables
# ------------------------------------------------------------------------------
target = 'cuda'
dev = tvm.cuda(0)
size = (1024, 28)
data, mean, var, gamma, beta, out = get_bn_data(size[0], size[1], tvm.nd.array)

data = tvm.nd.array(data, dev)
mean = tvm.nd.array(mean, dev)  
var = tvm.nd.array(var, dev)
gamma = tvm.nd.array(gamma, dev)
beta = tvm.nd.array(beta, dev)
out = tvm.nd.array(out, dev)

test_input = data, mean, var, gamma, beta, out


# ------------------------------------------------------------------------------
# Helper Function
# ------------------------------------------------------------------------------
# def evaluate_operation(s, vars, target, inputs, standard, optimization, log):
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
  target = 'cuda'
  dev = tvm.cuda(0)
  func = tvm.build(s, vars, target=target)
  data, mean, var, gamma, beta, out = inputs
  func(data, mean, var, gamma, beta, out)

  evaluator = func.time_evaluator(func.entry_name, dev, number=300)
  mean_time = evaluator(data, mean, var, gamma, beta, out).mean * 1000  # Convert to milliseconds
  log.append((optimization, mean_time))


def report_performance(log):
  """Convert the log into a performance table.
  Args:
    log: The log list.
  """
  baseline = log[-1][1]
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

  # dev = tvm.device(target.kind.name, 0)
  # Generate random tensor for testing.

  
  
  ctx = getattr(mx, "cpu")()
  mxnet_times = bench_bn_mxnet(size)

  # ----------------------------------------------------------------------------
  # Register Benchmarks and Dump Report
  # ----------------------------------------------------------------------------
  # Register default schedule.
  sch, arg_bufs = default_bn(size)
  evaluate_operation(sch,
                      arg_bufs,
                      target=target,
                      inputs=test_input,
                      optimization="default_bn",
                      log=log)

  sch, arg_bufs = optimized_bn(size)  
  evaluate_operation(sch,
                      arg_bufs,
                      target=target,
                      inputs=test_input,
                      optimization="optimized_bn",
                      log=log)

  sch, arg_bufs = gpu_pooling_autoschedule(size)
  evaluate_operation(sch,
                      arg_bufs,
                      target=target,
                      inputs=test_input,
                      optimization="auto_bn",
                      log=log)

  
  # Register numpy case.
  log.append(("mxnet_pooling", mxnet_times  * 1000))  # Milliseconds
  # Dump the performance table.
  report_performance(log)


if __name__ == "__main__":
  main()
