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

void swish(float* data, int channels, int size)
{
    for (int q = 0; q < channels; q++)
    {
        float* ptr = data + q * size;
        int i = 0;
        float32x4_t _one = vdupq_n_f32(1.f);
        for (; i + 15 < size; i += 16)
        {
            float32x4_t _p0 = vld1q_f32(ptr);
            float32x4_t _p1 = vld1q_f32(ptr + 4);
            float32x4_t _p2 = vld1q_f32(ptr + 8);
            float32x4_t _p3 = vld1q_f32(ptr + 12);
            _p0 = vdivq_f32(_p0, vaddq_f32(_one, exp_ps(vnegq_f32(_p0))));
            _p1 = vdivq_f32(_p1, vaddq_f32(_one, exp_ps(vnegq_f32(_p1))));
            _p2 = vdivq_f32(_p2, vaddq_f32(_one, exp_ps(vnegq_f32(_p2))));
            _p3 = vdivq_f32(_p3, vaddq_f32(_one, exp_ps(vnegq_f32(_p3))));
            vst1q_f32(ptr, _p0);
            vst1q_f32(ptr + 4, _p1);
            vst1q_f32(ptr + 8, _p2);
            vst1q_f32(ptr + 12, _p3);
            ptr += 16;
        }
        for (; i + 7 < size; i += 8)
        {
            float32x4_t _p0 = vld1q_f32(ptr);
            float32x4_t _p1 = vld1q_f32(ptr + 4);
            _p0 = vdivq_f32(_p0, vaddq_f32(_one, exp_ps(vnegq_f32(_p0))));
            _p1 = vdivq_f32(_p1, vaddq_f32(_one, exp_ps(vnegq_f32(_p1))));
            vst1q_f32(ptr, _p0);
            vst1q_f32(ptr + 4, _p1);
            ptr += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            float32x4_t _p = vld1q_f32(ptr);
            _p = vdivq_f32(_p, vaddq_f32(_one, exp_ps(vnegq_f32(_p))));
            vst1q_f32(ptr, _p);
            ptr += 4;
        }
        for (; i < size; i++)
        {
            *ptr = *ptr / (1.f + expf(-*ptr));

            ptr++;
        }
    }
}