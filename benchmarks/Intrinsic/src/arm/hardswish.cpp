#include <arm_neon.h>

void hardswish(float* data, int channels, int size, float alpha, float beta)
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
            float32x4_t _ans0 = vmlaq_f32(_beta, _p0, _alpha);
            float32x4_t _ans1 = vmlaq_f32(_beta, _p1, _alpha);
            float32x4_t _ans2 = vmlaq_f32(_beta, _p2, _alpha);
            float32x4_t _ans3 = vmlaq_f32(_beta, _p3, _alpha);
            _ans0 = vmaxq_f32(_ans0, _zero);
            _ans1 = vmaxq_f32(_ans1, _zero);
            _ans2 = vmaxq_f32(_ans2, _zero);
            _ans3 = vmaxq_f32(_ans3, _zero);
            _ans0 = vminq_f32(_ans0, _one);
            _ans1 = vminq_f32(_ans1, _one);
            _ans2 = vminq_f32(_ans2, _one);
            _ans3 = vminq_f32(_ans3, _one);
            _p0 = vmulq_f32(_ans0, _p0);
            _p1 = vmulq_f32(_ans1, _p1);
            _p2 = vmulq_f32(_ans2, _p2);
            _p3 = vmulq_f32(_ans3, _p3);
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
            float32x4_t _ans0 = vmlaq_f32(_beta, _p0, _alpha);
            float32x4_t _ans1 = vmlaq_f32(_beta, _p1, _alpha);
            _ans0 = vmaxq_f32(_ans0, _zero);
            _ans1 = vmaxq_f32(_ans1, _zero);
            _ans0 = vminq_f32(_ans0, _one);
            _ans1 = vminq_f32(_ans1, _one);
            _p0 = vmulq_f32(_ans0, _p0);
            _p1 = vmulq_f32(_ans1, _p1);
            vst1q_f32(ptr, _p0);
            vst1q_f32(ptr + 4, _p1);
            ptr += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            float32x4_t _p = vld1q_f32(ptr);
            float32x4_t _ans = vmlaq_f32(_beta, _p, _alpha);
            _ans = vmaxq_f32(_ans, _zero);
            _ans = vminq_f32(_ans, _one);
            _p = vmulq_f32(_ans, _p);
            vst1q_f32(ptr, _p);
            ptr += 4;
        }
        for (; i < size; i++)
        {
            if (*ptr < lower)
                *ptr = 0.f;
            else if (*ptr > upper)
                ;
            else
                *ptr = *ptr * (*ptr * alpha + beta);

            ptr++;
        }
    }
}