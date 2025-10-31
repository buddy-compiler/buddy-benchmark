#include <arm_neon.h>
#include <cmath>

void relu(float* data, int channels, int size, float slope = 0.f)
{
    if (slope == 0.f)
    {
        for (int q = 0; q < channels; q++)
        {
            float* ptr = data + size * q;

            int i = 0;
            float32x4_t _zero = vdupq_n_f32(0.f);
            for (; i + 15 < size; i += 16)
            {
                float32x4_t _p0 = vld1q_f32(ptr);
                float32x4_t _p1 = vld1q_f32(ptr + 4);
                float32x4_t _p2 = vld1q_f32(ptr + 8);
                float32x4_t _p3 = vld1q_f32(ptr + 12);
                _p0 = vmaxq_f32(_p0, _zero);
                _p1 = vmaxq_f32(_p1, _zero);
                _p2 = vmaxq_f32(_p2, _zero);
                _p3 = vmaxq_f32(_p3, _zero);
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
                _p0 = vmaxq_f32(_p0, _zero);
                _p1 = vmaxq_f32(_p1, _zero);
                vst1q_f32(ptr, _p0);
                vst1q_f32(ptr + 4, _p1);
                ptr += 8;
            }
            for (; i + 3 < size; i += 4)
            {
                float32x4_t _ptr = vld1q_f32(ptr);
                _ptr = vmaxq_f32(_ptr, _zero);
                vst1q_f32(ptr, _ptr);
                ptr += 4;
            }
            for (; i < size; i++)
            {
                *ptr = std::max(*ptr, 0.f);
                ptr++;
            }
        }
    }
    else
    {
        for (int q = 0; q < channels; q++)
        {
            float* ptr = data + q * size;
            int i = 0;
            float32x4_t _zero = vdupq_n_f32(0.f);
            float32x4_t _slope = vdupq_n_f32(slope);
            for (; i + 15 < size; i += 16)
            {
                float32x4_t _p0 = vld1q_f32(ptr);
                float32x4_t _p1 = vld1q_f32(ptr + 4);
                float32x4_t _p2 = vld1q_f32(ptr + 8);
                float32x4_t _p3 = vld1q_f32(ptr + 12);
                uint32x4_t _lemask0 = vcleq_f32(_p0, _zero);
                uint32x4_t _lemask1 = vcleq_f32(_p1, _zero);
                uint32x4_t _lemask2 = vcleq_f32(_p2, _zero);
                uint32x4_t _lemask3 = vcleq_f32(_p3, _zero);
                float32x4_t _ps0 = vmulq_f32(_p0, _slope);
                float32x4_t _ps1 = vmulq_f32(_p1, _slope);
                float32x4_t _ps2 = vmulq_f32(_p2, _slope);
                float32x4_t _ps3 = vmulq_f32(_p3, _slope);
                _p0 = vbslq_f32(_lemask0, _ps0, _p0);
                _p1 = vbslq_f32(_lemask1, _ps1, _p1);
                _p2 = vbslq_f32(_lemask2, _ps2, _p2);
                _p3 = vbslq_f32(_lemask3, _ps3, _p3);
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
                uint32x4_t _lemask0 = vcleq_f32(_p0, _zero);
                uint32x4_t _lemask1 = vcleq_f32(_p1, _zero);
                float32x4_t _ps0 = vmulq_f32(_p0, _slope);
                float32x4_t _ps1 = vmulq_f32(_p1, _slope);
                _p0 = vbslq_f32(_lemask0, _ps0, _p0);
                _p1 = vbslq_f32(_lemask1, _ps1, _p1);
                vst1q_f32(ptr, _p0);
                vst1q_f32(ptr + 4, _p1);
                ptr += 8;
            }
            for (; i + 3 < size; i += 4)
            {
                float32x4_t _p = vld1q_f32(ptr);
                uint32x4_t _lemask = vcleq_f32(_p, _zero);
                float32x4_t _ps = vmulq_f32(_p, _slope);
                _p = vbslq_f32(_lemask, _ps, _p);
                vst1q_f32(ptr, _p);
                ptr += 4;
            }
            for (; i < size; i++)
            {
                if (*ptr < 0)
                    *ptr *= slope;
                ptr++;
            }
        }
    }
}