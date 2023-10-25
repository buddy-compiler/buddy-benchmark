import tvm
import tvm.testing
from tvm import te
import numpy as np

from tvm.script import tir as T
from tvm import meta_schedule as ms
from tvm.script.parser.tir import evaluate

from gpu_matmul import *


def report_performance(log):
    """Convert the log into a performance table.
    Args:
      log: The log list.
    """
    baseline = log[0][1]
    header = (
        "Benchmark".ljust(20) + "\t" + "Time".rjust(10) + "\t" + "SpeedUp".rjust(10)
    )
    split_line = "-" * 50
    print(split_line)
    print(header)
    print(split_line)
    for result in log:
        formatted_time = "{:.8f}".format(result[1])
        formatted_performance = "{:.2f}".format(baseline / result[1])
        print(
            "\033[32m%s\033[0m\t\033[33m%s\033[0m\t\033[34m%s\033[0m"
            % (
                result[0].ljust(20),
                str(formatted_time + " ms").rjust(10),
                str(formatted_performance).rjust(10),
            )
        )


def main():
    log = []
    outputs = []
    M = 64
    N = 3136
    K = 576
    num_flop = 2 * M * N * K
    dev = tvm.cuda(0)
    A_np = np.random.uniform(size=(M, K)).astype("float32")
    B_np = np.random.uniform(size=(K, N)).astype("float32")
    A_nd = tvm.nd.array(A_np, dev)
    B_nd = tvm.nd.array(B_np, dev)
    C_nd = tvm.nd.array(np.zeros((M, N), dtype="float32"), dev)

    sch = matmul_default((M, K, N))
    rt_mod = tvm.build(sch.mod, target="cuda")
    evaluator = rt_mod.time_evaluator("main", dev, number=10)
    log.append(("Default schedule", (evaluator(A_nd, B_nd, C_nd).mean)))

    sch = matmul_blocking((M, K, N))
    rt_mod = tvm.build(sch.mod, target="cuda")
    evaluator = rt_mod.time_evaluator("main", dev, number=10)
    log.append(("BlockingSchedule", (evaluator(A_nd, B_nd, C_nd).mean)))

    sch = matmul_blocking_with_shared((M, K, N))
    rt_mod = tvm.build(sch.mod, target="cuda")
    evaluator = rt_mod.time_evaluator("main", dev, number=10)
    log.append(("BlockingSharedSchedule", (evaluator(A_nd, B_nd, C_nd).mean)))

    sch_tuned = matmul_autoschedule((M, K, N))
    rt_mod = tvm.build(sch_tuned.mod, target="cuda")
    log.append(("autoschedule", (evaluator(A_nd, B_nd, C_nd).mean)))

    report_performance(log)


if __name__ == "__main__":
    main()
