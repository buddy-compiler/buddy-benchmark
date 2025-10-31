#include <arm_neon.h>

void hardsigmoid(float* data, int channels, int size, float alpha, float beta)
{
    float lower = -beta / alpha;
    float upper = (1.f / alpha) + lower;
    for (int q = 0; q < channels; q++)
    {
        float* ptr = data + q * size;
        int i = 0;
        float32x4_t _zero = vdupq_n_f32(0.f);
        float32x4_t _one = vdupq_n_f32(1.f);
        float32x4_t _alpha = vdupq_n_f32(alpha);
        float32x4_t _beta = vdupq_n_f32(beta);
        for (; i + 15 < size; i += 16)
        {
            float32x4_t _p0 = vld1q_f32(ptr);
            float32x4_t _p1 = vld1q_f32(ptr + 4);
            float32x4_t _p2 = vld1q_f32(ptr + 8);
            float32x4_t _p3 = vld1q_f32(ptr + 12);
            _p0 = vmlaq_f32(_beta, _p0, _alpha);
            _p1 = vmlaq_f32(_beta, _p1, _alpha);
            _p2 = vmlaq_f32(_beta, _p2, _alpha);
            _p3 = vmlaq_f32(_beta, _p3, _alpha);
            _p0 = vmaxq_f32(_p0, _zero);
            _p1 = vmaxq_f32(_p1, _zero);
            _p2 = vmaxq_f32(_p2, _zero);
            _p3 = vmaxq_f32(_p3, _zero);
            _p0 = vminq_f32(_p0, _one);
            _p1 = vminq_f32(_p1, _one);
            _p2 = vminq_f32(_p2, _one);
            _p3 = vminq_f32(_p3, _one);
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
            _p0 = vmlaq_f32(_beta, _p0, _alpha);
            _p1 = vmlaq_f32(_beta, _p1, _alpha);
            _p0 = vmaxq_f32(_p0, _zero);
            _p1 = vmaxq_f32(_p1, _zero);
            _p0 = vminq_f32(_p0, _one);
            _p1 = vminq_f32(_p1, _one);
            vst1q_f32(ptr, _p0);
            vst1q_f32(ptr + 4, _p1);
            ptr += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            float32x4_t _p = vld1q_f32(ptr);
            _p = vmlaq_f32(_beta, _p, _alpha);
            _p = vmaxq_f32(_p, _zero);
            _p = vminq_f32(_p, _one);
            vst1q_f32(ptr, _p);
            ptr += 4;
        }
        for (; i < size; i++)
        {
            if (*ptr < lower)
                *ptr = 0.f;
            else if (*ptr > upper)
                *ptr = 1.f;
            else
                *ptr = *ptr * alpha + beta;

            ptr++;
        }
    }
}