# ===- convolution_manual.py ---------------------------------------------------
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
# This file implements the TVM manual optimization for Conv on CPU.
# This file is based on the TVM tutorial:
# https://tvm.apache.org/docs/tutorial/tensor_expr_get_started.html
# TVM is an Apache-2.0 licensed project.
# See the TVM license at: https://github.com/apache/tvm/blob/main/LICENSE
#
# ===---------------------------------------------------------------------------

import numpy as np
import tvm
from tvm import te


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
    data = np.random.normal(size=(ic, n, n)).astype("float32")
    weight = np.random.normal(size=(oc, ic, k, k)).astype("float32")
    on = conv_out_size(n, k, p, s)
    out = np.empty((oc, on, on), dtype="float32")
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


def conv_default(oc, ic, nh, nw, kh, kw, ph=0, pw=0, sh=1, sw=1):
    """Convolution
    oc, ic : output and input channels
    nh, nw : input width and height
    kh, kw : kernel width and height
    ph, pw : height and width padding sizes, default 0
    sh, sw : height and width strides, default 1
    """
    # reduction axes
    ric = te.reduce_axis((0, ic), name="ric")
    rkh = te.reduce_axis((0, kh), name="rkh")
    rkw = te.reduce_axis((0, kw), name="rkw")
    # output height and width
    oh = conv_out_size(nh, kh, ph, sh)
    ow = conv_out_size(nw, kw, pw, sw)
    # pad X and then compute Y
    X = te.placeholder((ic, nh, nw), name="X")
    K = te.placeholder((oc, ic, kh, kw), name="K")
    PaddedX = padding(X, ph, pw) if ph * pw != 0 else X
    Y = te.compute(
        (oc, oh, ow),
        lambda c, i, j: te.sum(
            PaddedX[ric, i * sh + rkh, j * sw + rkw] * K[c, ric, rkh, rkw],
            axis=[ric, rkh, rkw],
        ),
        name="Y",
    )
    return X, K, Y, PaddedX


th, tw = 8, 8  # Tile sizes for height and weight


def blocked_cached_conv(oc, ic, n, k, p, s):
    X, K, Y, PaddedX = conv_default(oc, ic, n, n, k, k, p, p, s, s)
    s = te.create_schedule(Y.op)
    CachedY = s.cache_write(Y, "local")
    # Compute the output block for every output channel in parallel
    oc, h, w = Y.op.axis
    ho, wo, hi, wi = s[Y].tile(h, w, th, tw)
    ochw = s[Y].fuse(oc, ho, wo)
    s[Y].parallel(ochw)
    # Cache the output block, and move the inner height and width axes
    # to innermost, so we can vectorize and unroll them
    s[CachedY].compute_at(s[Y], ochw)
    _, ch, cw = CachedY.op.axis
    ric, rkh, rkw = CachedY.op.reduce_axis
    s[CachedY].reorder(ric, rkh, rkw, ch, cw)
    s[CachedY].vectorize(cw)
    s[CachedY].unroll(ch)
    # Schedule the padding by adding thread-level parallelism
    if PaddedX != X:
        s[PaddedX].parallel(PaddedX.op.axis[0])
    return s, (X, K, Y)


def conv_pack(oc, ic, nh, nw, kh, kw, ph, pw, toc, tic):
    """Pack data and weight for convolution

    oc, ic : output and input channels
    nh, nw : input width and height
    kh, kw : kernel width and height
    ph, pw : height and width padding
    toc, tic : the tiling sizes of the output and input channels
    """
    X = te.placeholder((ic, nh, nw), name="X")
    K = te.placeholder((oc, ic, kh, kw), name="K")
    PaddedX = padding(X, ph, pw) if ph * pw != 0 else X
    # pack X and K
    assert ic % tic == 0 and oc % toc == 0
    PackedX = te.compute(
        (ic // tic, nh + ph * 2, nw + pw * 2, tic),
        lambda ic_out, x, y, ic_in: PaddedX[ic_out * tic + ic_in, x, y],
        name="PackedX",
    )
    PackedK = te.compute(
        (oc // toc, ic // tic, kh, kw, tic, toc),
        lambda oc_out, ic_out, x, y, ic_in, oc_in: K[
            oc_out * toc + oc_in, ic_out * tic + ic_in, x, y
        ],
        name="PackedK",
    )
    return X, K, PaddedX, PackedX, PackedK


def conv_using_pack(oc, ic, nh, nw, kh, kw, ph, pw, sh, sw, toc, tic):
    """2-D conv
    oc, ic : output and input channels.
    nh, nw : input width and height
    kh, kw : kernel width and height
    ph, pw : height and width padding
    sh, sw : height and width strides
    toc, tic : the tiling sizes of output channel and input channel
    """
    X, K, PaddedX, PackedX, PackedK = conv_pack(
        oc, ic, nh, nw, kh, kw, ph, pw, toc, tic
    )
    # reduction axes
    ric_in = te.reduce_axis((0, tic), name="ric_in")
    ric_out = te.reduce_axis((0, ic // tic), name="ric_out")
    rkh = te.reduce_axis((0, kh), name="rkh")
    rkw = te.reduce_axis((0, kw), name="rkw")
    # output height and weights
    oh = conv_out_size(nh, kh, ph, sh)
    ow = conv_out_size(nw, kw, pw, sw)
    # Compuated Y in the packed layout
    PackedY = te.compute(
        (oc // toc, oh, ow, toc),
        lambda oc_out, x, y, oc_in: te.sum(
            PackedX[ric_out, x * sh + rkh, y * sw + rkw, ric_in]
            * PackedK[oc_out, ric_out, rkh, rkw, ric_in, oc_in],
            axis=[ric_out, rkh, rkw, ric_in],
        ),
        name="Y",
    )
    # Unpack the result
    Y = te.compute(
        (oc, oh, ow), lambda oc, x, y: PackedY[oc // toc, x, y, oc % toc], name="Y"
    )
    return X, K, Y, PaddedX, PackedX, PackedK, PackedY


def packed_cached_conv(oc, ic, n, k, p, s):
    toc, tic, tw = 16, 16, 4
    X, K, Y, PaddedX, PackedX, PackedK, PackedY = conv_using_pack(
        oc, ic, n, n, k, k, p, p, s, s, toc, tic
    )
    s = te.create_schedule(Y.op)
    CachedY = s.cache_write(PackedY, "local")
    oc_out, h, w, oc_in = s[PackedY].op.axis
    oc_out_h = s[PackedY].fuse(oc_out, h)
    # Parallel on the first two dimensions oc_out and h
    s[PackedY].parallel(oc_out_h)
    # Optimize the computation of a cached output block
    w_out, w_in = s[PackedY].split(w, factor=tw)  # Split the columns
    s[CachedY].compute_at(s[PackedY], w_out)
    _, _, cw, oc_in = CachedY.op.axis
    ric_out, rkh, rkw, ric_in = CachedY.op.reduce_axis
    s[CachedY].reorder(ric_out, rkh, rkw, ric_in, cw, oc_in)
    s[CachedY].unroll(ric_in)
    s[CachedY].unroll(cw)
    s[CachedY].vectorize(oc_in)
    # Schedule the padding by adding thread-level parallelism
    if PaddedX != X:
        s[PaddedX].parallel(PaddedX.op.axis[0])
    # Optimize the packing of X and K
    s[PackedX].parallel(s[PackedX].fuse(*PackedX.op.axis[0:2]))
    s[PackedX].unroll(PackedX.op.axis[-1])
    s[PackedK].parallel(s[PackedK].fuse(*PackedK.op.axis[0:2]))
    s[PackedK].unroll(PackedK.op.axis[-1])
    # Optimize the unpacking of Y
    s[Y].parallel(s[Y].fuse(*Y.op.axis[0:2]))
    s[Y].unroll(Y.op.axis[-1])
    return s, (X, K, Y)


def default_conv(oc, ic, n, k, p, s):
    X, K, Y, PaddedX = conv_default(oc, ic, n, n, k, k, p, p, s, s)
    s = te.create_schedule(Y.op)
    return s, (X, K, Y)
