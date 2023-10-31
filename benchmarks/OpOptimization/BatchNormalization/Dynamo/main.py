# ===- main.py --------------------------------------------------------
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
# This file implements the dynamo entry for benchmark BatchNormalization on GPU.
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
dtype = "float32"


# ------------------------------------------------------------------------------
# Helper Function
# ------------------------------------------------------------------------------
def evaluator(s, inputs, num):
    all_time = []
    data, mean, var, gamma, beta, out = inputs
    for i in range(num):
        torch.cuda.synchronize()
        start = time.time()
        result = s(data, mean, var, gamma, beta, out)
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
      vars: The argument lists to the function.
      target: The target and option of the compilation.
      inputs: The input tensors.
      standard: The standard result for correctness evaluation.
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
    header = (
        "Benchmark".ljust(20) + "\t" + "Time".rjust(10) + "\t" + "SpeedUp".rjust(10)
    )
    split_line = "-" * 50
    print(split_line)
    print(header)
    print(split_line)
    for result in log:
        formatted_time = "{:.2f}".format(result[1])
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
    # ----------------------------------------------------------------------------
    # Initialization and Baseline
    # ----------------------------------------------------------------------------
    # Initialize the log list.
    log = []
    # Generate random tensor for testing.
    size = (4096, 1024)
    data, mean, var, gamma, beta, out = get_bn_data(size[0], size[1])
    # ----------------------------------------------------------------------------
    # Register Benchmarks and Dump Report
    # ----------------------------------------------------------------------------
    # Register default schedule.
    s_1 = batch_norm_default()
    evaluate_operation(
        s_1,
        inputs=(data, mean, var, gamma, beta, out),
        optimization="torch_broAdd_default",
        log=log,
    )

    s_2 = batch_norm_compiled()
    evaluate_operation(
        s_2,
        inputs=(data, mean, var, gamma, beta, out),
        optimization="torch_broAdd_compiled",
        log=log,
    )

    report_performance(log)


if __name__ == "__main__":
    main()
