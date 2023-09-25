import numpy
import time
from pooling_iree import *
# ------------------------------------------------------------------------------
# User Configurable Variables
# ------------------------------------------------------------------------------
dtype = "float32"
# ------------------------------------------------------------------------------
# Helper Function
# ------------------------------------------------------------------------------
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
  size = (512, 64, 3)
  c, n, k, p, s = size[0], size[0], size[1], size[2], 1
  oc, ic, n, k, p, s = size[0], size[0], size[1], size[2], 1, 1
  data, out_max = get_pool_data_torch(c, n, k, p, s)
  model = torch_pooling(k, p, s)
  model(data)
  example_input = data
  invoker = iree_pooling(model,example_input)
  # ----------------------------------------------------------------------------
  # Register Benchmarks and Dump Report
  # ----------------------------------------------------------------------------
  # Register default schedule.
  
  evaluate_operation(invoker,
                     inputs=example_input,
                     optimization="IREE",
                     log=log)
  evaluate_operation(model,
                     inputs=data,
                     optimization="torch_cpu",
                     log=log)
  report_performance(log)

if __name__ == "__main__":
  main()