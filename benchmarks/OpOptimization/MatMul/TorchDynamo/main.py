# You may obtain a copy of the License at
#
#     https://github.com/pytorch/pytorch/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ===---------------------------------------------------------------------------
#
# This file implements the dynamo entry for benchmark Matmul on GPU.
# torchdynamo is an internal API that uses a CPython feature called the Frame Evaluation 
# API to safely capture PyTorch graphs. Methods that are available externally for PyTorch 
# users are surfaced through the torch.compiler namespace.
# which can automatically generate search spaces for optimizing tensor expressions.
# See the pytorch license at: https://github.com/pytorch/pytorch/blob/main/LICENSE
#
# ===---------------------------------------------------------------------------
import numpy
import time
from torchDynamo import *
# ------------------------------------------------------------------------------
# User Configurable Variables
# ------------------------------------------------------------------------------
# Define the size of the matrix.
# (M, K) x (K, N)
M = 4
N = 4
K = 4
dtype = "float32"

# ------------------------------------------------------------------------------
# Helper Function
# ------------------------------------------------------------------------------
def evaluator(s, inputs, num):
  a, b = inputs
  result = s(a, b)  #warm up
  all_time = []
  for i in range(num):
      torch.cuda.synchronize()
      start = time.time()
      result = s(a, b)
      torch.cuda.synchronize()
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
  mean_time = evaluator(s, inputs, 10)
  log.append((optimization, mean_time))

def report_performance(log):
  """Convert the log into a performance table.
  Args:
    log: The log list.
  """
  baseline = log[0][1]
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
  a_np = numpy.random.rand(M, K)
  b_np = numpy.random.rand(K, N)
  a_tensor = torch.tensor(a_np)
  b_tensor = torch.tensor(b_np)
  # ----------------------------------------------------------------------------
  # Register Benchmarks and Dump Report
  # ----------------------------------------------------------------------------
  # Register default schedule.
  s = default_matrix_multiply()
  evaluate_operation(s,
                     inputs=(a_tensor, b_tensor),
                     optimization="torch_default",
                     log=log)

  s = dynamo_matrix_multiply()
  evaluate_operation(s,
                     inputs=(a_tensor, b_tensor),
                     optimization="torch_dynamo",
                     log=log)

  report_performance(log)

if __name__ == "__main__":
  main()