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
# This file implements the manual optimization for benchmark DepthwiseConvolution on CPU.
# Autoscheduler is TVM's next-generation performance tuning tool,
# which can automatically generate search spaces for optimizing tensor expressions.
# TVM is an Apache-2.0 licensed project.
# See the TVM license at: https://github.com/apache/tvm/blob/main/LICENSE
#
# ===---------------------------------------------------------------------------
import numpy as np
import tvm
from tvm import te


def depthwise_conv_default(ic, nh, nw, kh, kw, ph=0, pw=0, sh=1, sw=1):
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
    sch = te.create_schedule(Y.op)
    return sch, (X, K, Y)


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


def depthwise_conv_pack(c, nh, nw, kh, kw, ph, pw, tc):
    """Pack data and weight for depthwise convolution
       Note that the input channel of kernel is specified as 1,
       and the output channel of kernel equals the input channel of data
    c : input channel of data and output channel of kernel
    nh, nw : input width and height
    kh, kw : kernel width and height
    ph, pw : height and width padding
    tc : the tiling size of channels
    """
    X = te.placeholder((c, nh, nw), name="X")
    K = te.placeholder((c, 1, kh, kw), name="K")
    PaddedX = padding(X, ph, pw) if ph * pw != 0 else X
    # make sure the channel tiling is valid
    if c < tc:
        tc = c
    assert c % tc == 0
    # pack X and K
    PackedX = te.compute(
        (c // tc, nh + ph * 2, nw + pw * 2, tc),
        lambda c_out, x, y, c_in: PaddedX[c_out * tc + c_in, x, y],
        name="PackedX",
    )
    PackedK = te.compute(
        (c // tc, 1, kh, kw, 1, tc),
        lambda c_out, _, x, y, __, c_in: K[c_out * tc + c_in, 0, x, y],
        name="PackedK",
    )
    return X, K, PaddedX, PackedX, PackedK


def depthwise_conv(c, nh, nw, kh, kw, ph, pw, sh, sw, tc):
    """depthwise conv
    c : number of channels for both input and output.
    nh, nw : input width and height
    kh, kw : kernel width and height
    ph, pw : height and width padding
    sh, sw : height and width strides
    tc : the tiling sizes of channels
    """
    X, K, PaddedX, PackedX, PackedK = depthwise_conv_pack(c, nh, nw, kh, kw, ph, pw, tc)
    # reduction axes
    rkh = te.reduce_axis((0, kh), name="rkh")
    rkw = te.reduce_axis((0, kw), name="rkw")
    # output height and weights
    oh = conv_out_size(nh, kh, ph, sh)
    ow = conv_out_size(nw, kw, pw, sw)
    # compute Y in the packed layout
    PackedY = te.compute(
        (c // tc, oh, ow, tc),
        lambda c_out, x, y, c_in: te.sum(
            (
                PackedX[c_out, x * sh + rkh, y * sw + rkw, c_in]
                * PackedK[c_out, 0, rkh, rkw, 0, c_in]
            ),
            axis=[rkh, rkw],
        ),
        name="PackedY",
    )

    # Unpack the result
    Y = te.compute(
        (c, oh, ow), lambda c, x, y: PackedY[c // tc, x, y, c % tc], name="Y"
    )
    return X, K, Y, PaddedX, PackedX, PackedK, PackedY


def depthwise_cached_block(c, n, k, p, s):
    # tiling sizes for channel and width
    tc, tw = 16, 4
    X, K, Y, PaddedX, PackedX, PackedK, PackedY = depthwise_conv(
        c, n, n, k, k, p, p, s, s, tc
    )
    sch = te.create_schedule(Y.op)
    CachedY = sch.cache_write(PackedY, "global")
    c_out, h, w, c_in = sch[PackedY].op.axis
    w_out, w_in = sch[PackedY].split(w, factor=tw)
    sch[PackedY].reorder(c_out, h, w_out, w_in, c_in)
    c_out_h = sch[PackedY].fuse(c_out, h)
    sch[PackedY].parallel(c_out_h)
    sch[CachedY].compute_at(sch[PackedY], w_out)
    cc_out, ch, cw, cc_in = sch[CachedY].op.axis
    kh, kw = sch[CachedY].op.reduce_axis
    sch[CachedY].reorder(cc_out, ch, kh, kw, cw, cc_in)
    sch[CachedY].vectorize(cc_in)
    sch[CachedY].unroll(cw)
    # Schedule the padding by adding thread-level parallelism
    if PaddedX != X:
        sch[PaddedX].parallel(PaddedX.op.axis[0])
    # Optimize the packing of X and K
    sch[PackedX].parallel(sch[PackedX].fuse(*PackedX.op.axis[0:2]))
    sch[PackedX].unroll(PackedX.op.axis[-1])
    sch[PackedK].parallel(sch[PackedK].fuse(*PackedK.op.axis[0:2]))
    sch[PackedK].unroll(PackedK.op.axis[-1])
    # Optimize the unpacking of Y
    sch[Y].parallel(sch[Y].fuse(*Y.op.axis[0:2]))
    sch[Y].unroll(Y.op.axis[-1])
    return sch, (X, K, Y)
