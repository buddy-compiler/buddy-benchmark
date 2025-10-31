#include <cfloat>
#include <riscv_vector.h>

#define CV_PI   3.1415926535897932384626433832795

static const float atan2_p1 = 0.9997878412794807f*(float)(180/CV_PI);
static const float atan2_p3 = -0.3258083974640975f*(float)(180/CV_PI);
static const float atan2_p5 = 0.1555786518463281f*(float)(180/CV_PI);
static const float atan2_p7 = -0.04432655554792128f*(float)(180/CV_PI);

void fastAtan32f(const float* y, const float* x, float* dst, size_t n, bool angle_in_deg) {
    const float scale = angle_in_deg ? 1.f : CV_PI / 180.f;
    const float p1 = atan2_p1 * scale;
    const float p3 = atan2_p3 * scale;
    const float p5 = atan2_p5 * scale;
    const float p7 = atan2_p7 * scale;
    const float angle_90_deg = 90.F * scale;

    static size_t vlmax = __riscv_vsetvlmax_e32m4();
    auto vp1 = __riscv_vfmv_v_f_f32m4(p1, vlmax);
    auto vp3 = __riscv_vfmv_v_f_f32m4(p3, vlmax);
    auto vp5 = __riscv_vfmv_v_f_f32m4(p5, vlmax);

    for (size_t vl; n > 0; n -= vl) {
        vl = __riscv_vsetvl_e32m4(n);

        auto vy = __riscv_vle32_v_f32m4(y, vl);
        auto vx = __riscv_vle32_v_f32m4(x, vl);

        const auto ax = __riscv_vfabs(vx, vl);
        const auto ay = __riscv_vfabs(vy, vl);
        const auto c = __riscv_vfdiv(
            __riscv_vfmin(ax, ay, vl),
            __riscv_vfadd(__riscv_vfmax(ax, ay, vl), FLT_EPSILON, vl), vl);
        const auto c2 = __riscv_vfmul(c, c, vl);

        auto a = __riscv_vfmadd(c2, p7, vp5, vl);
        a = __riscv_vfmadd(a, c2, vp3, vl);
        a = __riscv_vfmadd(a, c2, vp1, vl);
        a = __riscv_vfmul(a, c, vl);

        const auto mask = __riscv_vmflt(ax, ay, vl);
        a = __riscv_vfrsub_mu(mask, a, a, angle_90_deg, vl);
        a = __riscv_vfrsub_mu(__riscv_vmflt(vx, 0.F, vl), a, a, angle_90_deg * 2, vl);
        a = __riscv_vfrsub_mu(__riscv_vmflt(vy, 0.F, vl), a, a, angle_90_deg * 4, vl);

        __riscv_vse32(dst, a, vl);

        x += vl;
        y += vl;
        dst += vl;
    }
}
