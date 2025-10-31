#include <arm_neon.h>

#include <cmath>

#define c_exp_hi 88.3762626647949f
#define c_exp_lo -88.3762626647949f

#define c_cephes_LOG2EF 1.44269504088896341
#define c_cephes_exp_C1 0.693359375
#define c_cephes_exp_C2 -2.12194440e-4

#define c_cephes_exp_p0 1.9875691500E-4
#define c_cephes_exp_p1 1.3981999507E-3
#define c_cephes_exp_p2 8.3334519073E-3
#define c_cephes_exp_p3 4.1665795894E-2
#define c_cephes_exp_p4 1.6666665459E-1
#define c_cephes_exp_p5 5.0000001201E-1

static inline float32x4_t exp_ps(float32x4_t x)
{
    float32x4_t tmp, fx;

    float32x4_t one = vdupq_n_f32(1);
    x = vminq_f32(x, vdupq_n_f32(c_exp_hi));
    x = vmaxq_f32(x, vdupq_n_f32(c_exp_lo));

    fx = vmlaq_f32(vdupq_n_f32(0.5f), x, vdupq_n_f32(c_cephes_LOG2EF));

    tmp = vcvtq_f32_s32(vcvtq_s32_f32(fx));

    uint32x4_t mask = vcgtq_f32(tmp, fx);
    mask = vandq_u32(mask, vreinterpretq_u32_f32(one));

    fx = vsubq_f32(tmp, vreinterpretq_f32_u32(mask));

    tmp = vmulq_f32(fx, vdupq_n_f32(c_cephes_exp_C1));
    float32x4_t z = vmulq_f32(fx, vdupq_n_f32(c_cephes_exp_C2));
    x = vsubq_f32(x, tmp);
    x = vsubq_f32(x, z);

    z = vmulq_f32(x, x);

    float32x4_t y = vdupq_n_f32(c_cephes_exp_p0);
    y = vmlaq_f32(vdupq_n_f32(c_cephes_exp_p1), y, x);
    y = vmlaq_f32(vdupq_n_f32(c_cephes_exp_p2), y, x);
    y = vmlaq_f32(vdupq_n_f32(c_cephes_exp_p3), y, x);
    y = vmlaq_f32(vdupq_n_f32(c_cephes_exp_p4), y, x);
    y = vmlaq_f32(vdupq_n_f32(c_cephes_exp_p5), y, x);

    y = vmlaq_f32(x, y, z);
    y = vaddq_f32(y, one);

    int32x4_t mm;
    mm = vcvtq_s32_f32(fx);
    mm = vaddq_s32(mm, vdupq_n_s32(0x7f));
    mm = vshlq_n_s32(mm, 23);
    float32x4_t pow2n = vreinterpretq_f32_s32(mm);

    y = vmulq_f32(y, pow2n);
    return y;
}

void selu(float* data, int channels, int size, float alpha, float lambda)
{
    float alphaxlambda = alpha * lambda;
    for (int q = 0; q < channels; q++)
    {
        float* ptr = data + q * size;
        int nn = size >> 2;
        int remain = size - (nn << 2);
        float32x4_t _one = vdupq_n_f32(1.f);
        float32x4_t _zero = vdupq_n_f32(0.f);
        float32x4_t _alphaxlambda = vdupq_n_f32(alphaxlambda);
        float32x4_t _lambda = vdupq_n_f32(lambda);
        for (; nn > 0; nn--)
        {
            float32x4_t _p = vld1q_f32(ptr);
            uint32x4_t _lemask = vcleq_f32(_p, _zero);

            float32x4_t _nps = exp_ps(_p);
            _nps = vsubq_f32(_nps, _one);
            _nps = vmulq_f32(_nps, _alphaxlambda);

            _p = vmulq_f32(_p, _lambda);

            _p = vbslq_f32(_lemask, _nps, _p);
            vst1q_f32(ptr, _p);

            ptr += 4;
        }
        for (; remain > 0; remain--)
        {
            if (*ptr < 0.f)
                *ptr = (expf(*ptr) - 1.f) * alphaxlambda;
            else
                *ptr *= lambda;

            ptr++;
        }
    }
}