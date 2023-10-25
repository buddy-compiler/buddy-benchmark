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
# This file implements the auto optimization for benchmark BatchNormalization on GPU.
# Autoscheduler is TVM's next-generation performance tuning tool,
# which can automatically generate search spaces for optimizing tensor expressions.
# TVM is an Apache-2.0 licensed project.
# See the TVM license at: https://github.com/apache/tvm/blob/main/LICENSE
#
# ===---------------------------------------------------------------------------
from gpu_batch_norm_manual import *
import tvm
from tvm import autotvm
from tvm import te, auto_scheduler
import numpy as np


@auto_scheduler.register_workload
def batch_norm(c, n, eps=1e-5):
    """batch normalization
    c : channels
    N : input width and height
    eps : small positive value to prevent divide 0
    """
    X = te.placeholder((c, n, n), name="X")
    Mean = te.placeholder((c, 1, 1), name="Mean")
    Var = te.placeholder((c, 1, 1), name="Var")
    Gamma = te.placeholder((c, 1, 1), name="Gamma")
    Beta = te.placeholder((c, 1, 1), name="Beta")
    C1 = topi.subtract(X, Mean)
    C2 = topi.sqrt(Var + eps)
    Y = C1 / C2 * Gamma + Beta
    return X, Mean, Var, Gamma, Beta, Y


def gpu_batch_norm_auto_tuning_plus(args, target):
    target = tvm.target.Target(target)
    size = args
    c, n = size[:]
    task = tvm.auto_scheduler.SearchTask(func=batch_norm, args=(c, n), target=target)
    print("==========batch_norm_auto_tuning_plus=========")
    log_file = "batch_norm_auto_tuning_plus.log"
    measure_ctx = None
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=60,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        verbose=1,
    )
    # vervose to determine whether output or not
    task.tune(tune_option)
    sch, args = task.apply_best(log_file)
    return sch, args


def gpu_pooling_autoschedule(size):
    target = tvm.target.Target(target="cuda", host="llvm")
    sch, arg_bufs = gpu_batch_norm_auto_tuning_plus(size, target)
    return sch, arg_bufs
