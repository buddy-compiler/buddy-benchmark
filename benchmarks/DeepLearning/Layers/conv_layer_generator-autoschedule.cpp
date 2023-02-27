#include "Halide.h"

namespace {

using namespace Halide;

class ConvolutionLayer : public Halide::Generator<ConvolutionLayer> {
public:
    Input<Buffer<float, 4>> input{"input"};
    Input<Buffer<float, 4>> filter{"filter"};
    Input<Buffer<float, 1>> bias{"bias"};
    Output<Buffer<float, 4>> relu{"relu"};

    void generate() {
        const int N = 5, CI = 128, CO = 128, W = 100, H = 80;

        /* THE ALGORITHM */

        Var x("x"), y("y"), c("c"), n("n");

        Func conv("conv");
        RDom r(0, CI, 0, 3, 0, 3);

        conv(c, x, y, n) = bias(c);
        conv(c, x, y, n) += filter(c, r.y, r.z, r.x) * input(r.x, x + r.y, y + r.z, n);

        relu(c, x, y, n) = max(0, conv(c, x, y, n));

        /* THE SCHEDULE */

        relu.dim(0).set_bounds(0, CO).set_stride(1);
        relu.dim(1).set_bounds(0, W).set_stride(CO);
        relu.dim(2).set_bounds(0, H).set_stride(CO * W);
        relu.dim(3).set_bounds(0, N).set_stride(CO * H * W);

        input.dim(0).set_bounds(0, CI).set_stride(1);
        input.dim(1).set_bounds(0, W + 2).set_stride(CI);
        input.dim(2).set_bounds(0, H + 2).set_stride(CI * (W + 2));
        input.dim(3).set_bounds(0, N).set_stride(CI * (W + 2) * (H + 2));

        filter.dim(0).set_bounds(0, CO).set_stride(1);
        filter.dim(1).set_bounds(0, 3).set_stride(CO);
        filter.dim(2).set_bounds(0, 3).set_stride(CO * 3);
        filter.dim(3).set_bounds(0, CI).set_stride(CO * 3 * 3);

        bias.dim(0).set_bounds(0, CO).set_stride(1);

        if (using_autoscheduler()) {
            input.dim(0).set_estimate(0, CI);
            input.dim(1).set_estimate(0, W + 2);
            input.dim(2).set_estimate(0, H + 2);
            input.dim(3).set_estimate(0, N);

            filter.dim(0).set_estimate(0, CO);
            filter.dim(1).set_estimate(0, 3);
            filter.dim(2).set_estimate(0, 3);
            filter.dim(3).set_estimate(0, CI);

            bias.dim(0).set_estimate(0, CO);

            relu.dim(0).set_estimate(0, W);
            relu.dim(1).set_estimate(0, H);
            relu.dim(2).set_estimate(0, CO);
            relu.dim(3).set_estimate(0, N);
        }
    }
};

}  // namespace

HALIDE_REGISTER_GENERATOR(ConvolutionLayer, conv_layer_autoschedule)
