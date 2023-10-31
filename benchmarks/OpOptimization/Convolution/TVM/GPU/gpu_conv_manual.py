# ===- gpu_conv_auto.py --------------------------------------------------------
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
# This file implements the TVM manual optimization for conv on GPU.
# This file is based on the TVM tutorial:
# https://tvm.apache.org/docs/tutorial/tensor_expr_get_started.html
# TVM is an Apache-2.0 licensed project.
# See the TVM license at: https://github.com/apache/tvm/blob/main/LICENSE
#
# ===---------------------------------------------------------------------------

import numpy as np
import tvm
from tvm import te


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


def conv(oc, ic, nh, nw, kh, kw, ph=0, pw=0, sh=1, sw=1):
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


def gpu_conv_default(oc, ic, n, k, p, s):
    X, K, Y, PaddedX = conv(oc, ic, n, n, k, k, p, p, s, s)
    sch = te.create_schedule(Y.op)
    sch[PaddedX].compute_inline()
    _, y, x = sch[Y].op.axis
    sch[Y].bind(y, te.thread_axis("blockIdx.x"))
    sch[Y].bind(x, te.thread_axis("threadIdx.x"))
    return sch, (X, K, Y)


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


def gpu_conv_tiling(oc, ic, n, k, p, s):
    tile_c = [4, 8]
    tile_h = [2, 2]
    tile_w = [16, 4]
    tile_rc = [1, 1]
    tile_rh = [1, 1]
    tile_rw = [1, 3]
    X, K, Y, PaddedX = conv(oc, ic, n, n, k, k, p, p, s, s)
    sch = te.create_schedule(Y.op)
    sch[PaddedX].compute_inline()

    YL = sch.cache_write(Y, "local")

    # create cache stage
    XX = sch.cache_read(PaddedX, "shared", [YL])
    KK = sch.cache_read(K, "shared", [YL])
    XL = sch.cache_read(XX, "local", [YL])
    KL = sch.cache_read(KK, "local", [YL])

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

    # tile reduction axes
    c, h, w = sch[YL].op.axis
    rc, rh, rw = sch[YL].op.reduce_axis
    rco, rcm, rci = split_axis(tile_rc, sch, YL, rc)
    rho, rhm, rhi = split_axis(tile_rh, sch, YL, rh)
    rwo, rwm, rwi = split_axis(tile_rw, sch, YL, rw)
    sch[YL].reorder(rco, rho, rwo, rcm, rhm, rwm, rci, rhi, rwi, c, h, w)

    sch[XX].compute_at(sch[YL], rwo)
    sch[KK].compute_at(sch[YL], rwo)
    sch[XL].compute_at(sch[YL], rwm)
    sch[KL].compute_at(sch[YL], rwm)

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


def gpu_conv_vthread(oc, ic, n, k, p, s):
    tile_c = [1, 4, 8]
    tile_h = [1, 2, 2]
    tile_w = [2, 16, 2]  # making 2 virtual thread along the ow dimension
    tile_rc = [1, 1]
    tile_rh = [1, 3]  # making the data access in columns
    tile_rw = [1, 1]
    X, K, Y, PaddedX = conv(oc, ic, n, n, k, k, p, p, s, s)
    sch = te.create_schedule(Y.op)
    sch[PaddedX].compute_inline()

    YL = sch.cache_write(Y, "local")

    # create cache stage
    XX = sch.cache_read(PaddedX, "shared", [YL])
    KK = sch.cache_read(K, "shared", [YL])
    XL = sch.cache_read(XX, "local", [YL])
    KL = sch.cache_read(KK, "local", [YL])

    c, h, w = sch[Y].op.axis

    bc, vc, tc, ic = split_axis(tile_c, sch, Y, c)
    bh, vh, th, ih = split_axis(tile_h, sch, Y, h)
    bw, vw, tw, iw = split_axis(tile_w, sch, Y, w)

    sch[Y].bind(bc, te.thread_axis("blockIdx.z"))
    sch[Y].bind(bh, te.thread_axis("blockIdx.y"))
    sch[Y].bind(bw, te.thread_axis("blockIdx.x"))
    sch[Y].bind(vc, te.thread_axis("vthread"))
    sch[Y].bind(vh, te.thread_axis("vthread"))
    sch[Y].bind(vw, te.thread_axis("vthread"))
    sch[Y].bind(tc, te.thread_axis("threadIdx.z"))
    sch[Y].bind(th, te.thread_axis("threadIdx.y"))
    sch[Y].bind(tw, te.thread_axis("threadIdx.x"))
    sch[Y].reorder(bc, bh, bw, vc, vh, vw, tc, th, tw, ic, ih, iw)

    sch[YL].compute_at(sch[Y], tw)

    # tile reduction axes
    c, h, w = sch[YL].op.axis
    rc, rh, rw = sch[YL].op.reduce_axis
    rco, rcm, rci = split_axis(tile_rc, sch, YL, rc)
    rho, rhm, rhi = split_axis(tile_rh, sch, YL, rh)
    rwo, rwm, rwi = split_axis(tile_rw, sch, YL, rw)
    sch[YL].reorder(rco, rho, rwo, rcm, rhm, rwm, rci, rhi, rwi, c, h, w)

    sch[XX].compute_at(sch[YL], rwo)
    sch[KK].compute_at(sch[YL], rwo)
    sch[XL].compute_at(sch[YL], rwm)
    sch[KL].compute_at(sch[YL], rwm)

    # cooperative fetching
    for load in [XX, KK]:
        args = sch[load].op.axis
        fused = sch[load].fuse(*args)
        # align thread layout
        tz, fused = sch[load].split(fused, nparts=tile_c[1])
        ty, fused = sch[load].split(fused, nparts=tile_h[1])
        tx, _ = sch[load].split(fused, nparts=tile_w[1])
        sch[load].bind(tz, te.thread_axis("threadIdx.z"))
        sch[load].bind(ty, te.thread_axis("threadIdx.y"))
        sch[load].bind(tx, te.thread_axis("threadIdx.x"))

    return sch, (X, K, Y)
