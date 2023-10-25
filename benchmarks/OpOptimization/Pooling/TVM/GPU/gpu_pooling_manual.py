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
# This file implements the manual optimization for benchmark pooling on GPU.
# Autoscheduler is TVM's next-generation performance tuning tool, 
# which can automatically generate search spaces for optimizing tensor expressions.
# TVM is an Apache-2.0 licensed project.
# See the TVM license at: https://github.com/apache/tvm/blob/main/LICENSE
#
# ===---------------------------------------------------------------------------
import numpy as np
import timeit
import tvm
from tvm import te
import mxnet as mx

target = 'cuda'
dev = tvm.cuda(0)
# attain the maximal number of threads of a CUDA block
nt = 0
with tvm.target.Target(target):
    nt = tvm.target.Target.current(allow_none=False).max_num_threads

def pool_mxnet(pool_type, data, out, k, p, s):
    mx.nd.Pooling(data, kernel=(k,k), stride=(s,s),
                      pad=(p,p), pool_type=pool_type, out=out)

def get_pool_data_mxnet(c, n, k, p, s, ctx='cpu'):
    ctx = getattr(mx, ctx)()
    data, _, out = get_conv_data(c, c, n, k, p, s,
                                      lambda x: mx.nd.array(x, ctx=ctx))
    data, out = data.expand_dims(axis=0), out.expand_dims(axis=0)
    return data, out

def pooling_timer_mxnet(pool_type, c, n, k, ctx):
    """Benchmark pooling in MXNet
    c : channels
    n : input width and height
    k : kernel width and height
    """
    timer = timeit.Timer(
        setup='import mxnet as mx\n'
        'from gpu_pooling_manual import get_pool_data_mxnet\n'
        'from gpu_pooling_manual import pool_mxnet\n'
        'c, n, k, p, s = %d, %d, %d, 1, 1\n'
        'data, out = get_pool_data_mxnet(\n'
        '    c, n, k, p, s, "%s")'%(c, n, k, ctx),
        stmt='pool_mxnet("%s", data, out, k, p, s);'
        'out.wait_to_read()'%(pool_type)
        )
    return timer.timeit

def bench_workload(workload):
    """Benchmark a workload
    workload: a method that accept a num_repeat argument
    and return its total execution time
    """
    workload(1)  # warmup
    time = workload(1)  # the time to run once
    if time > 1: return time
    # The number of repeats to measure at least 1 second
    num_repeats = max(int(1.0 / time), 5)
    return workload(num_repeats) / num_repeats

def bench_pooling_mxnet(pool_type, sizes, ctx='cpu'):
    """Return the execution times of MXNet pooling"""
    return bench_workload(pooling_timer_mxnet(pool_type, sizes[0], sizes[1], sizes[2], ctx))

def get_conv_data(oc, ic, n, k, p=0, s=1, constructor=None):
    """Return random 3-D data tensor, 3-D kernel tenor and empty 3-D output
    tensor with the shapes specified by input arguments.
    oc, ic : output and input channels
    n : input width and height
    k : kernel width and height
    p : padding size, default 0
    s : stride, default 1
    constructor : user-defined tensor constructor
    """
    np.random.seed(0)
    data = np.random.normal(size=(ic, n, n)).astype('float32')
    weight = np.random.normal(size=(oc, ic, k, k)).astype('float32')
    on = conv_out_size(n, k, p, s)
    out = np.empty((oc, on, on), dtype='float32')
    if constructor:
        data, weight, out = (constructor(x) for x in [data, weight, out])
    return data, weight, out

def padding(X, ph, pw, val=0):
    """Pad X with the given value in 2-D
    ph, pw : height and width padding
    val : padding value, default 0
    """
    assert len(X.shape) >= 2
    nh, nw = X.shape[-2], X.shape[-1]
    return te.compute(
            (*X.shape[0:-2], nh+ph*2, nw+pw*2),
            lambda *i: te.if_then_else(
                te.any(i[-2]<ph, i[-2]>=nh+ph, i[-1]<pw, i[-1]>=nw+pw),
                val, X[i[:-2]+(i[-2]-ph, i[-1]-pw)]),
            name='PaddedX')

def conv_out_size(n, k, p, s):
    """Compute the output size by given input size n (width or height),
    kernel size k, padding p, and stride s
    Return output size (width or height)
    """
    return (n - k + 2 * p)//s + 1

def pool(pool_type, c, nh, nw, kh, kw, ph=0, pw=0, sh=1, sw=1):
    """2D pooling
    pool_type: pooling type, 'max' or 'avg'
    c : channels
    nh, nw : input width and height
    kh, kw : kernel width and height
    ph, pw : height and width padding sizes, default 0
    sh, sw : height and width strides, default 1
    """
    # reduction axes
    rkh = te.reduce_axis((0, kh), name='rkh')
    rkw = te.reduce_axis((0, kw), name='rkw')
    # output height and weights
    oh = conv_out_size(nh, kh, ph, sh)
    ow = conv_out_size(nw, kw, pw, sw)
    # pad X and then compute Y
    X = te.placeholder((c, nh, nw), name='X')

    if pool_type == 'max':
        PaddedX = padding(X, ph, pw, val=te.min_value(X.dtype)) \
            if ph * pw != 0 else X
        Y = te.compute((c, oh, ow), \
                            lambda c, h, w: \
                            te.max(PaddedX[c, h*sh+rkh, w*sw+rkw], \
                                axis=[rkh, rkw]), \
                            tag="pool_max", name='PoolMax')
    elif pool_type == 'avg':
        PaddedX = padding(X, ph, pw) if ph * pw != 0 else X
        tsum = te.compute((c, oh, ow), \
                            lambda c, h, w: \
                            te.sum(PaddedX[c, h*sh+rkh, w*sw+rkw], \
                                axis=[rkh, rkw]), \
                            tag="pool_avg1", name='PoolSum')
        Y = te.compute((c, oh, ow), \
                            lambda c, h, w: \
                            tsum[c, h, w] / (kh*kw), \
                            tag='pool_avg2', name='PoolAvg')
    else:
        raise ValueError("Pool type should be 'avg' or 'max'.")
    return X, Y, PaddedX

def schedule_max(size):
    c, n, k = size[:]
    X, Y, PaddedX = pool('max', c, n, n, k, k, 1, 1, 1, 1)
    sch = te.create_schedule(Y.op)
    sch[PaddedX].compute_inline()
    # traversal axes binding
    fused = sch[Y].fuse(*sch[Y].op.axis)
    bx, tx = sch[Y].split(fused, factor=nt)
    sch[Y].bind(bx, te.thread_axis("blockIdx.x"))
    sch[Y].bind(tx, te.thread_axis("threadIdx.x"))
    return sch, (X, Y)

def schedule_avg(size):
    c, n, k = size[:]
    X, Y, PaddedX = pool('avg', c, n, n, k, k, 1, 1, 1, 1)
    sch = te.create_schedule(Y.op)
    sch[PaddedX].compute_inline()
    # traversal axes binding
    fused = sch[Y].fuse(*sch[Y].op.axis)
    bx, tx = sch[Y].split(fused, factor=nt)
    sch[Y].bind(bx, te.thread_axis("blockIdx.x"))
    sch[Y].bind(tx, te.thread_axis("threadIdx.x"))
    # merging two stages
    PoolSum = Y.op.input_tensors[0]
    sch[PoolSum].compute_at(sch[Y], tx)
    return sch, (X, Y)

size = (512, 512, 3)
c, n, k, p, s = size[0], size[0], size[1], size[2], 1
data, _, out_max = get_conv_data(size[0], size[0], size[1], size[2], 1, 1, tvm.nd.array)
data = tvm.nd.array(data, dev)
out_max = tvm.nd.array(out_max, dev)
sch, args = schedule_avg(size)
mod = tvm.build(sch, args, target)
mod(data, out_max)
