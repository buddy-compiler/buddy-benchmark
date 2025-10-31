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

void gelu(float* data, int channels, int size, bool fast_gelu)
{
    if (!fast_gelu)
    {
        for (int q = 0; q < channels; q++)
        {
            float* ptr = data + q * size;

            for (int i = 0; i < size; i++)
            {

                ptr[i] = 0.5f * ptr[i] * erfcf(-0.70710678f * ptr[i]);
            }
        }
    }

    else
    {
        for (int q = 0; q < channels; q++)
        {
            float* ptr = data + q * size;
            int i = 0;
            for (; i + 3 < size; i += 4)
            {
                float32x4_t _pLoad = vld1q_f32(ptr);

                float32x4_t _blob = vmulq_f32(_pLoad, _pLoad);
                _blob = vmulq_f32(_pLoad, _blob);
                _blob = vmulq_f32(vdupq_n_f32(0.044715f * 0.79788452f), _blob);
                _blob = vmlaq_f32(_blob, vdupq_n_f32(0.79788452f), _pLoad);
                _blob = tanh_ps(_blob);
                _blob = vaddq_f32(vdupq_n_f32(1.f), _blob);
                _blob = vmulq_f32(vdupq_n_f32(0.5f), vmulq_f32(_blob, _pLoad));
                vst1q_f32(ptr, _blob);
                ptr += 4;
            }
            for (; i < size; i++)
            {

                *ptr = 0.5f * *ptr * (1.0f + tanhf(0.79788452f * (*ptr + 0.044715f * *ptr * *ptr * *ptr)));
                ptr++;
            }
        }
    }
}
