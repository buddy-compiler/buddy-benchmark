#include <arm_neon.h>

#include <cmath>

#define c_tanh_tiny 1e-4f
#define c_tanh_hi   9.0f

#define c_tanh_alpha_1  4.89352455891786e-3f
#define c_tanh_alpha_3  6.37261928875436e-4f
#define c_tanh_alpha_5  1.48572235717979e-5f
#define c_tanh_alpha_7  5.12229709037114e-8f
#define c_tanh_alpha_9  -8.60467152213735e-11f
#define c_tanh_alpha_11 2.00018790482477e-13f
#define c_tanh_alpha_13 -2.76076847742355e-16f

#define c_tanh_beta_0 4.89352518554385e-3f
#define c_tanh_beta_2 2.26843463243900e-3f
#define c_tanh_beta_4 1.18534705686654e-4f
#define c_tanh_beta_6 1.19825839466702e-6f

static inline float32x4_t tanh_ps(float32x4_t x)
{
    float32x4_t x2 = vabsq_f32(x);

    uint32x4_t tiny_mask = vcgeq_f32(x2, vdupq_n_f32(c_tanh_tiny));

    x2 = vreinterpretq_f32_u32(vbslq_u32(vcgeq_f32(vdupq_n_f32(c_tanh_hi), x2), vreinterpretq_u32_f32(x2), vreinterpretq_u32_f32(vdupq_n_f32(c_tanh_hi))));

    float32x4_t z = vmulq_f32(x2, x2);

    float32x4_t y = vdupq_n_f32(c_tanh_alpha_13);
    y = vmlaq_f32(vdupq_n_f32(c_tanh_alpha_11), y, z);
    y = vmlaq_f32(vdupq_n_f32(c_tanh_alpha_9), y, z);
    y = vmlaq_f32(vdupq_n_f32(c_tanh_alpha_7), y, z);
    y = vmlaq_f32(vdupq_n_f32(c_tanh_alpha_5), y, z);
    y = vmlaq_f32(vdupq_n_f32(c_tanh_alpha_3), y, z);
    y = vmlaq_f32(vdupq_n_f32(c_tanh_alpha_1), y, z);
    y = vmulq_f32(y, x2);

    float32x4_t w = vdupq_n_f32(c_tanh_beta_6);
    w = vmlaq_f32(vdupq_n_f32(c_tanh_beta_4), w, z);
    w = vmlaq_f32(vdupq_n_f32(c_tanh_beta_2), w, z);
    w = vmlaq_f32(vdupq_n_f32(c_tanh_beta_0), w, z);

    y = vdivq_f32(y, w);

    y = vreinterpretq_f32_u32(vbslq_u32(vdupq_n_u32(1u << 31), vreinterpretq_u32_f32(x), vreinterpretq_u32_f32(y)));

    y = vreinterpretq_f32_u32(vbslq_u32(tiny_mask, vreinterpretq_u32_f32(y), vreinterpretq_u32_f32(x)));

    return y;
}

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

static inline float32x4_t sigmoid_ps(float32x4_t _v)
{
    float32x4_t _one = vdupq_n_f32(1.f);
    _v = vnegq_f32(_v);
    _v = exp_ps(_v);
    _v = vaddq_f32(_v, _one);
    float32x4_t _outp = vrecpeq_f32(_v);
    _outp = vmulq_f32(vrecpsq_f32(_v, _outp), _outp);
    return vmulq_f32(vrecpsq_f32(_v, _outp), _outp);
}

void sigmoid(float* data, int channels, int size)
{
    for (int q = 0; q < channels; q++)
    {
        float* ptr = data + q * size;

        int i = 0;
        for (; i + 15 < size; i += 16)
        {
            float32x4_t _p0 = vld1q_f32(ptr);
            float32x4_t _p1 = vld1q_f32(ptr + 4);
            float32x4_t _p2 = vld1q_f32(ptr + 8);
            float32x4_t _p3 = vld1q_f32(ptr + 12);
            _p0 = sigmoid_ps(_p0);
            _p1 = sigmoid_ps(_p1);
            _p2 = sigmoid_ps(_p2);
            _p3 = sigmoid_ps(_p3);
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
            _p0 = sigmoid_ps(_p0);
            _p1 = sigmoid_ps(_p1);
            vst1q_f32(ptr, _p0);
            vst1q_f32(ptr + 4, _p1);
            ptr += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            float32x4_t _p = vld1q_f32(ptr);
            _p = sigmoid_ps(_p);
            vst1q_f32(ptr, _p);
            ptr += 4;
        }
        for (; i < size; i++)
        {
            *ptr = 1.f / (1.f + expf(-*ptr));
            ptr++;
        }
    }
}