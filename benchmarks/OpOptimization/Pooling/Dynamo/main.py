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
  for i in range(num):
      torch.cuda.synchronize()
      start = time.time()
      result = s(inputs)
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
  size = (512, 64, 3)
  c, n, k, p, s = size[0], size[0], size[1], size[2], 1
  oc, ic, n, k, p, s = size[0], size[0], size[1], size[2], 1, 1
  data, out_max = get_pool_data_torch(c, n, k, p, s)


  # ----------------------------------------------------------------------------
  # Register Benchmarks and Dump Report
  # ----------------------------------------------------------------------------
  # Register default schedule.

  s_1 = pool_torch(k,p,s)
  evaluate_operation(s_1,
                     inputs=data,
                     optimization="torch_pooling_default",
                     log=log)


  s_2 = pool_compiled(k,p,s)
  evaluate_operation(s_2,
                     inputs=data,
                     optimization="torch_pooling_dynamo",
                     log=log)

 

  report_performance(log)


if __name__ == "__main__":
  main()
