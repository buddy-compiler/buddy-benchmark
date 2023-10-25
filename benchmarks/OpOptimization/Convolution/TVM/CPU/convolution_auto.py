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
# This file implements the auto optimization for benchmark Convolution on CPU.
# Autoscheduler is TVM's next-generation performance tuning tool, 
# which can automatically generate search spaces for optimizing tensor expressions.
# TVM is an Apache-2.0 licensed project.
# See the TVM license at: https://github.com/apache/tvm/blob/main/LICENSE
#
# ===---------------------------------------------------------------------------
import tvm
from tvm import autotvm
from tvm import te, auto_scheduler
from convolution_manual import *
import numpy as np

@auto_scheduler.register_workload
def conv_default(oc, ic, nh, nw, kh, kw, ph=0, pw=0, sh=1, sw=1):
    """Convolution
    oc, ic : output and input channels
    nh, nw : input width and height
    kh, kw : kernel width and height
    ph, pw : height and width padding sizes, default 0
    sh, sw : height and width strides, default 1
    """
    # reduction axes
    ric = te.reduce_axis((0, ic), name='ric')
    rkh = te.reduce_axis((0, kh), name='rkh')
    rkw = te.reduce_axis((0, kw), name='rkw')
    # output height and width
    oh = conv_out_size(nh, kh, ph, sh)
    ow = conv_out_size(nw, kw, pw, sw)
    # pad X and then compute Y
    X = te.placeholder((ic, nh, nw), name='X')
    K = te.placeholder((oc, ic, kh, kw), name='K')
    PaddedX = padding(X, ph, pw) if ph * pw != 0 else X
    Y = te.compute(
        (oc, oh, ow),
        lambda c, i, j: te.sum(
            PaddedX[ric, i*sh+rkh, j*sw+rkw] * K[c, ric, rkh, rkw],
            axis=[ric, rkh, rkw]), name='Y')
    return X, K, Y, PaddedX

def conv_auto(oc, ic, n, k, p, s):
    target = tvm.target.Target(target="llvm", host="llvm")
    task = tvm.auto_scheduler.SearchTask(func=conv_default, args= (oc, ic, n, n, k, k, p, p, s, s), target=target)
    print("==========conv_autoschedule=========")
    log_file = "conv_autoschedule.log"
    measure_ctx = None
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=120,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        verbose=1,
        early_stopping = 20,
        )
        # vervose to determine whether output or not
    task.tune(tune_option)
    sch, args = task.apply_best(log_file)
    X, K, Y, PaddedX = args
    return sch, (X, K, Y)
