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
# This file implements the manual optimization for benchmark DepthwiseConvolution on GPU.
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

target = "cuda"


def padding(X, ph, pw, val=0):
    """Pad X with the given value in 2-D

    ph, pw : height and width padding
    val : padding value, default 0
    """
    assert len(X.shape) >= 2
    nh, nw = X.shape[-2], X.shape[-1]
    return te.compute(
        (*X.shape[0:-2], nh + ph * 2, nw + pw * 2),
        lambda *i: te.if_then_else(
            te.any(i[-2] < ph, i[-2] >= nh + ph, i[-1] < pw, i[-1] >= nw + pw),
            val,
            X[i[:-2] + (i[-2] - ph, i[-1] - pw)],
        ),
        name="PaddedX",
    )


def conv_out_size(n, k, p, s):
    """Compute the output size by given input size n (width or height),
    kernel size k, padding p, and stride s
    Return output size (width or height)
    """
    return (n - k + 2 * p) // s + 1


def get_conv_data(oc, ic, n, k, p=0, s=1, constructor=None, conv_type="direct"):
    """Return random 3-D data tensor, 3-D kernel tenor and empty 3-D output
    tensor with the shapes specified by input arguments.
    oc, ic : output and input channels
    n : input width and height
    k : kernel width and height
    p : padding size, default 0
    s : stride, default 1
    conv_type: either direct 2D or depthwise, default direct
    constructor : user-defined tensor constructor
    """
    np.random.seed(0)
    data = np.random.normal(size=(ic, n, n)).astype("float32")
    ic_weight = ic
    if conv_type == "depthwise":
        ic_weight = 1
    weight = np.random.normal(size=(oc, ic_weight, k, k)).astype("float32")
    on = conv_out_size(n, k, p, s)
    out = np.empty((oc, on, on), dtype="float32")
    if constructor:
        data, weight, out = (constructor(x) for x in [data, weight, out])
    return data, weight, out


def split_axis(factors, sch, op, axis):
    """Splitting an axis into factors
    Parameters
    ----------
    factors: array of integers
        The factors that the split applies
    sch: tvm.te.schedule.Schedule
        The tvm schedule
    op: tvm.te.tensor.Operation
        The stage to be applied
    axis: tvm.te.schedule.IterVar
        axis to split
    Returns
    -------
    axes : list of Axis
        The transformed axes.
    """
    ret = []
    for i in range(0, len(factors)):
        ax0, ax1 = sch[op].split(axis, factor=int(np.prod(factors[i:])))
        ret.append(ax0)
        axis = ax1
    return ret + [axis]


def depthwise_conv(ic, nh, nw, kh, kw, ph=0, pw=0, sh=1, sw=1):
    """Convolution
    ic : number of channels for both input and output
    nh, nw : input width and height
    kh, kw : kernel width and height
    ph, pw : height and width padding sizes, default 0
    sh, sw : height and width strides, default 1
    """
    # reduction axes
    rkh = te.reduce_axis((0, kh), name="rkh")
    rkw = te.reduce_axis((0, kw), name="rkw")
    # output height and weights
    oh = conv_out_size(nh, kh, ph, sh)
    ow = conv_out_size(nw, kw, pw, sw)
    # pad X and then compute Y
    X = te.placeholder((ic, nh, nw), name="X")
    K = te.placeholder((ic, 1, kh, kw), name="K")
    PaddedX = padding(X, ph, pw) if ph * pw != 0 else X
    Y = te.compute(
        (ic, oh, ow),
        lambda c, i, j: te.sum(
            (PaddedX[c, i * sh + rkh, j * sw + rkw] * K[c, 0, rkh, rkw]),
            axis=[rkh, rkw],
        ),
        name="Y",
    )

    return X, K, Y, PaddedX


def default_sch(ic, n, k, p, s):
    X, K, Y, PaddedX = depthwise_conv(ic, n, n, k, k, p, p, s, s)
    sch = te.create_schedule(Y.op)
    sch[PaddedX].compute_inline()
    _, y, x = sch[Y].op.axis
    sch[Y].bind(y, te.thread_axis("blockIdx.x"))
    sch[Y].bind(x, te.thread_axis("threadIdx.x"))
    return sch, (X, K, Y)


def gpu_block_schedule(ic, n, k, p, s):
    tile_c = [1, 1]  # making each block take 1 channel
    tile_h = [2, 8]  # making each thread take 8 rows
    tile_w = [64, 1]  # making each thread take 1 column
    X, K, Y, PaddedX = depthwise_conv(ic, n, n, k, k, p, p, s, s)
    sch = te.create_schedule(Y.op)
    sch[PaddedX].compute_inline()
    YL = sch.cache_write(Y, "local")
    # create cache stage
    XX = sch.cache_read(PaddedX, "shared", [YL])
    KK = sch.cache_read(K, "shared", [YL])
    XL = sch.cache_read(XX, "local", [YL])
    KL = sch.cache_read(KK, "local", [YL])
    # tile and bind spatial axes
    c, h, w = sch[Y].op.axis
    bc, tc, ic = split_axis(tile_c, sch, Y, c)
    bh, th, ih = split_axis(tile_h, sch, Y, h)
    bw, tw, iw = split_axis(tile_w, sch, Y, w)
    sch[Y].bind(bc, te.thread_axis("blockIdx.z"))
    sch[Y].bind(bh, te.thread_axis("blockIdx.y"))
    sch[Y].bind(bw, te.thread_axis("blockIdx.x"))
    sch[Y].bind(tc, te.thread_axis("threadIdx.z"))
    sch[Y].bind(th, te.thread_axis("threadIdx.y"))
    sch[Y].bind(tw, te.thread_axis("threadIdx.x"))
    sch[Y].reorder(bc, bh, bw, tc, th, tw, ic, ih, iw)
    sch[YL].compute_at(sch[Y], tw)
    sch[XX].compute_at(sch[Y], bw)
    sch[KK].compute_at(sch[Y], bw)
    sch[XL].compute_at(sch[Y], tw)
    sch[KL].compute_at(sch[Y], tw)

    # cooperative fetching
    for load in [XX, KK]:
        args = sch[load].op.axis
        fused = sch[load].fuse(*args)
        # align thread layout
        tz, fused = sch[load].split(fused, nparts=tile_c[0])
        ty, fused = sch[load].split(fused, nparts=tile_h[0])
        tx, _ = sch[load].split(fused, nparts=tile_w[0])
        sch[load].bind(tz, te.thread_axis("threadIdx.z"))
        sch[load].bind(ty, te.thread_axis("threadIdx.y"))
        sch[load].bind(tx, te.thread_axis("threadIdx.x"))
    return sch, (X, K, Y)
