#include <arm_neon.h>
#include <cmath>

static const float atan2_p1 = 0.9997878412794807f;
static const float atan2_p3 = -0.3258083974640975f;
static const float atan2_p5 = 0.1555786518463281f;
static const float atan2_p7 = -0.04432655554792128f;

void fastAtan32f(const float* y, const float* x, float* dst, size_t n, bool angle_in_deg) {
    const float deg_scale = angle_in_deg ? 1.0f : static_cast<float>(M_PI / 180.0);
    const float rad_scale = static_cast<float>(180.0 / M_PI);

    const float P1 = atan2_p1 * rad_scale * deg_scale;
    const float P3 = atan2_p3 * rad_scale * deg_scale;
    const float P5 = atan2_p5 * rad_scale * deg_scale;
    const float P7 = atan2_p7 * rad_scale * deg_scale;
    const float A_90 = 90.0f * deg_scale;
    const float A_180 = 180.0f * deg_scale;
    const float A_360 = 360.0f * deg_scale;
    const float eps_val = static_cast<float>(1e-6);

    const float32x4_t v_eps = vdupq_n_f32(eps_val);
    const float32x4_t v_90 = vdupq_n_f32(A_90);
    const float32x4_t v_180 = vdupq_n_f32(A_180);
    const float32x4_t v_360 = vdupq_n_f32(A_360);
    const float32x4_t v_zero = vdupq_n_f32(0.0f);
    const float32x4_t v_p1 = vdupq_n_f32(P1);
    const float32x4_t v_p3 = vdupq_n_f32(P3);
    const float32x4_t v_p5 = vdupq_n_f32(P5);
    const float32x4_t v_p7 = vdupq_n_f32(P7);

    size_t i = 0;

    for (; i + 3 < n; i += 4) {
        float32x4_t v_y = vld1q_f32(y + i);
        float32x4_t v_x = vld1q_f32(x + i);

        float32x4_t v_ax = vabsq_f32(v_x);
        float32x4_t v_ay = vabsq_f32(v_y);

        float32x4_t v_min = vminq_f32(v_ax, v_ay);
        float32x4_t v_max = vmaxq_f32(v_ax, v_ay);

        float32x4_t v_c = vminq_f32(
            vdivq_f32(v_min, vaddq_f32(v_max, v_eps)),
            vdupq_n_f32(1.0f)
        );
        float32x4_t v_c2 = vmulq_f32(v_c, v_c);
        float32x4_t v_a = vmlaq_f32(v_p5, v_c2, v_p7);
        v_a = vmlaq_f32(v_p3, v_c2, v_a);
        v_a = vmlaq_f32(v_p1, v_c2, v_a);
        v_a = vmulq_f32(v_a, v_c);

        uint32x4_t v_swap_mask = vcltq_f32(v_ax, v_ay);
        v_a = vbslq_f32(v_swap_mask,
                        vsubq_f32(v_90, v_a),
                        v_a);

        uint32x4_t v_negx_mask = vcltq_f32(v_x, v_zero);
        v_a = vbslq_f32(v_negx_mask,
                        vsubq_f32(v_180, v_a),
                        v_a);

        uint32x4_t v_negy_mask = vcltq_f32(v_y, v_zero);
        v_a = vbslq_f32(v_negy_mask,
                        vsubq_f32(v_360, v_a),
                        v_a);
        vst1q_f32(dst + i, v_a);
    }
    for (; i < n; i++) {
        float ax = std::abs(x[i]);
        float ay = std::abs(y[i]);
        float c, c2, a;

        if (ax >= ay) {
            c = ay / (ax + eps_val);
            c2 = c * c;
            a = (((P7 * c2 + P5) * c2 + P3) * c2 + P1) * c;
        } else {
            c = ax / (ay + eps_val);
            c2 = c * c;
            a = A_90 - (((P7 * c2 + P5) * c2 + P3) * c2 + P1) * c;
        }

        if (x[i] < 0)
            a = A_180 - a;
        if (y[i] < 0)
            a = A_360 - a;

        dst[i] = a;
    }
}