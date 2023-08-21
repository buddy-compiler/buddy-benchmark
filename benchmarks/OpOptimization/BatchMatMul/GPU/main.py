from batch_matmul_gpu import *
import numpy
import timeit
import tvm.testing
import mxnet as mx
import time

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
  target = tvm.target.Target(target="cuda", host="llvm")
  batch_size = 64
  M = 64
  K = 128
  N = 256
  ctx = tvm.cuda(0)
  a_np = np.random.rand(batch_size, M, K).astype(np.float32)
  b_np = np.random.rand(batch_size, K, N).astype(np.float32)
  c_np = np.zeros((batch_size, M, N), dtype=np.float32)
  
  a = tvm.nd.array(a_np, ctx)
  b = tvm.nd.array(b_np, ctx)
  c = tvm.nd.array(c_np, ctx)
  shape = batch_size,M,K,N
  

  

  # 记录开始时间
  start_time = time.time()

  # 执行需要测试运行时间的代码
  # 例如，这里是一个简单的循环
  shape = batch_size,M,K,N
  batchMatmul_numpy(shape,a_np,b_np)

  end_time = time.time()

  # 计算运行时间
  numpy_time = end_time - start_time

  sch, arg_bufs = batchMatmul_manual(batch_size, M, K, N)
  evaluate_operation(sch,
                      arg_bufs,
                      target=target,
                      inputs=(a,b,c),
                      optimization="gpu_batchMatmul",
                      log=log)

  sch, arg_bufs = batchMatmul_auto_tuning(shape,"cuda")
  evaluate_operation(sch,
                      arg_bufs,
                      target=target,
                      inputs=(a,b,c),
                      optimization="batchMatmul_auto_tuning",
                      log=log)

  

  
  # Register numpy case.
  log.append(("numpy_time", numpy_time  * 1000))  # Milliseconds
  # Dump the performance table.
  report_performance(log)


if __name__ == "__main__":
  main()