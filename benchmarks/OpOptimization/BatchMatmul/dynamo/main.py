import numpy
import time
from torchDynamo import *
# ------------------------------------------------------------------------------
# User Configurable Variables
# ------------------------------------------------------------------------------

# Define the size of the matrix.
# (M, K) x (K, N)
B = 4
M = 4
N = 4
K = 4

dtype = "float32"


# ------------------------------------------------------------------------------
# Helper Function
# ------------------------------------------------------------------------------

def evaluator(s, inputs, num):
  a, b = inputs
  all_time = []
  s(a, b)
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
  a_np = numpy.random.rand(B, M, K)
  b_np = numpy.random.rand(B, K, N)
  a_tensor = torch.tensor(a_np)
  b_tensor = torch.tensor(b_np)


  # ----------------------------------------------------------------------------
  # Register Benchmarks and Dump Report
  # ----------------------------------------------------------------------------
  # Register default schedule.

  s = default_matrix_multiply()
  evaluate_operation(s,
                     inputs=(a_tensor, b_tensor),
                     optimization="torch_batchmm_default",
                     log=log)


  s = dynamo_batch_matrix_multiply()
  evaluate_operation(s,
                     inputs=(a_tensor, b_tensor),
                     optimization="torch_batchmm_dynamo",
                     log=log)

 

  report_performance(log)


if __name__ == "__main__":
  main()
