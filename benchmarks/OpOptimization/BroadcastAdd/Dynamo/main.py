import numpy
import time
from torchDynamo import *
# ------------------------------------------------------------------------------
# User Configurable Variables
# ------------------------------------------------------------------------------
dtype = "float32"

# ------------------------------------------------------------------------------
# Helper Function
# ------------------------------------------------------------------------------

def evaluator(s, inputs, num):
  all_time = []
  a,b = inputs
  for i in range(num):
      torch.cuda.synchronize()
      start = time.time()
      result = s(a,b)
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
  mean_time = evaluator(s, inputs, 1)
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
  m = 256
  n = 512
  shape1 = (m, 1)
  shape2 = (1, n)
  a, b, c = get_bcast_data(shape1, shape2)


  # ----------------------------------------------------------------------------
  # Register Benchmarks and Dump Report
  # ----------------------------------------------------------------------------
  # Register default schedule.

  s_1 = broadcastAdd_torch()
  evaluate_operation(s_1,
                     inputs=(a,b),
                     optimization="torch_broAdd_default",
                     log=log)


  s_2 = broadcastAdd_compiled()
  evaluate_operation(s_2,
                     inputs=(a,b),
                     optimization="torch_broAdd_compiled",
                     log=log)

 

  report_performance(log)


if __name__ == "__main__":
  main()
