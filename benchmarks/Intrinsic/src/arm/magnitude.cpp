#include <arm_neon.h>
#include <cmath>
#include <cstdio>
static inline float32x4_t vsqrtq_f32_impl(float32x4_t val)
{
    float32x4_t e = vrsqrteq_f32(val);
    e = vmulq_f32(vrsqrtsq_f32(vmulq_f32(e, e), val), e);
    e = vmulq_f32(vrsqrtsq_f32(vmulq_f32(e, e), val), e);

    float32x4_t reciprocal = vrecpeq_f32(e);
    reciprocal = vmulq_f32(vrecpsq_f32(e, reciprocal), reciprocal);
    reciprocal = vmulq_f32(vrecpsq_f32(e, reciprocal), reciprocal);
    return reciprocal;
}

static inline float32x2_t vsqrt_f32_impl(float32x2_t val)
{
    float32x2_t e = vrsqrte_f32(val);
    e = vmul_f32(vrsqrts_f32(vmul_f32(e, e), val), e);
    e = vmul_f32(vrsqrts_f32(vmul_f32(e, e), val), e);

    float32x2_t reciprocal = vrecpe_f32(e);
    reciprocal = vmul_f32(vrecps_f32(e, reciprocal), reciprocal);
    reciprocal = vmul_f32(vrecps_f32(e, reciprocal), reciprocal);
    return reciprocal;
}

void magnitude32f(const float *xx, const float *yy, float *dstBase, int len)
{

    const int step_base = 32 / sizeof(float);
    int roiw_base = len >= (step_base - 1) ? len - step_base + 1 : 0;
    const int step_tail = 8 / sizeof(float);
    int roiw_tail = len >= (step_tail - 1) ? len - step_tail + 1 : 0;

    const float *src0 = xx;
    const float *src1 = yy;
    float *dst = dstBase;
    int x = 0;
    for (; x < roiw_base; x += step_base)
    {
        float32x4_t v_src00 = vld1q_f32(src0 + x);
        float32x4_t v_src01 = vld1q_f32(src0 + x + 16 / sizeof(float));
        float32x4_t v_src10 = vld1q_f32(src1 + x);
        float32x4_t v_src11 = vld1q_f32(src1 + x + 16 / sizeof(float));
        float32x4_t v_dst;
        v_dst = vsqrtq_f32_impl(vaddq_f32(vmulq_f32(v_src00, v_src00), vmulq_f32(v_src10, v_src10)));
        vst1q_f32(dst + x, v_dst);

        v_dst = vsqrtq_f32_impl(vaddq_f32(vmulq_f32(v_src01, v_src01), vmulq_f32(v_src11, v_src11)));
        vst1q_f32(dst + x + 16 / sizeof(float), v_dst);
    }
    for (; x < roiw_tail; x += step_tail)
    {
        float32x2_t v_src0 = vld1_f32(src0 + x);
        float32x2_t v_src1 = vld1_f32(src1 + x);
        float32x2_t v_dst;
        v_dst = vsqrt_f32_impl(vadd_f32(vmul_f32(v_src0, v_src0), vmul_f32(v_src1, v_src1)));
        vst1_f32(dst + x, v_dst);
    }
    for (; x < len; ++x)
    {
        dst[x] = sqrtf(src0[x] * src0[x] + src1[x] * src1[x]);
    }
}
