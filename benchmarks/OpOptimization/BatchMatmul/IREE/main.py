# You may obtain a copy of the License at
#
#     https://github.com/openxla/iree/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ===---------------------------------------------------------------------------
#
# This file implements the IREE optimization for benchmark BatchMatmul on GPU.
# IREE (Intermediate Representation Execution Environment, pronounced as "eerie") 
# is an MLIR-based end-to-end compiler and runtime that lowers Machine Learning (ML) 
# models to a unified IR that scales up to meet the needs of the datacenter and down 
# to satisfy the constraints and special considerations of mobile and edge deployments.
#
# ===---------------------------------------------------------------------------
import numpy as np
import time
from batch_matmul_iree import *

dtype = "float32"
def iree_evaluator(s, inputs, num):
  result = s.forward(inputs)
  all_time = []
  for i in range(num):
      start = time.time()
      s.forward(inputs)
      end = time.time()
      elapsed_time = end - start
      all_time.append(elapsed_time)
  average_time = sum(all_time) / num
  return average_time

def numpy_evaluator(a_tensor,b_tensor,num):
  a_tensor_np = a_tensor.numpy()
  b_tensor_np = b_tensor.numpy()
  batch_size = a_tensor.shape[0]
  result_size1 =  a_tensor.shape[1]
  result_size2 = b_tensor.shape[2]
  result = np.random.randn(batch_size, result_size1, result_size2)
  all_time = []
  for i in range(num):
    for j in range(batch_size):
      start = time.time()
      result[j] = np.dot(a_tensor_np[j],b_tensor_np[j])
      end = time.time()
      elapsed_time = end - start
      all_time.append(elapsed_time)
  average_time = sum(all_time) / num
  return average_time

def evaluator(s, inputs, num):
  result = s(inputs)
  all_time = []
  for i in range(num):
      start = time.time()
      s(inputs)
      end = time.time()
      elapsed_time = end - start
      all_time.append(elapsed_time)
  average_time = sum(all_time) / num
  return average_time

def evaluate_operation(s, inputs, optimization, log):
  """Evaluate operation correctness and print the performance information.
  Args:
    s: The schedule to be built.
    inputs: The input tensors.
    optimization: The name of the optimization.
    log: The log list.
  """
  if optimization == "IREE":
    mean_time = iree_evaluator(s, inputs, 10)
  else:
    mean_time = evaluator(s, inputs, 10)
  log.append((optimization, mean_time))

def report_performance(log):
  """Convert the log into a performance table.
  Args:
    log: The log list.
  """
  baseline = log[-1][1]
  header = "Benchmark".ljust(20) + "\t" + "Time".rjust(
      10) + "\t" + "SpeedUp".rjust(10)
  split_line = "-" * 50
  print(split_line)
  print(header)
  print(split_line)
  for result in log:
    formatted_time = "{:.2f}".format(result[1])
    formatted_performance = "{:.2f}".format(baseline / result[1])
    print("\033[32m%s\033[0m\t\033[33m%s\033[0m\t\033[34m%s\033[0m" %
          (result[0].ljust(20), str(formatted_time + " ms").rjust(10),
           str(formatted_performance).rjust(10)))

def main():
  # ----------------------------------------------------------------------------
  # Initialization and Baseline
  # ----------------------------------------------------------------------------
  # Initialize the log list.
  log = []
  # Generate random tensor for testing.
  batch_num = 10
  a_dim1 = 2048
  a_dim2 = 1024
  b_dim1 = a_dim2
  b_dim2 = 2048
  model, b_tensor= torch_matrix_multiply(batch_num, b_dim1, b_dim2)
  a_tensor = torch.randn(batch_num, a_dim1, a_dim2)
  example_input = a_tensor
  # ----------------------------------------------------------------------------
  # Register Benchmarks and Dump Report
  # ----------------------------------------------------------------------------
  # Register default schedule.
  invoker = iree_matrix_multiply(model, example_input)
  evaluate_operation(invoker,
                     inputs=example_input,
                     optimization="IREE",
                     log=log)
  s = model
  evaluate_operation(s,
                     inputs=example_input,
                     optimization="torch_cpu",
                     log=log)
  np_average_time = numpy_evaluator(a_tensor,b_tensor,10)
  log.append(("numpy", np_average_time))
  report_performance(log)

if __name__ == "__main__":
  main()
