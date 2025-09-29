#include "x86_activation.h"
#include <cmath>

void tanh(float* data, int channels, int size)
{
    for (int q = 0; q < channels; q++)
    {
        float* ptr = data + q * size;

        int i = 0;
#if __AVX512F__
        for (; i + 15 < size; i += 16)
        {
            __m512 _p = _mm512_loadu_ps(ptr);
            _p = tanh_avx512(_p);
            _mm512_storeu_ps(ptr, _p);
            ptr += 16;
        }
        if (i < size)
        {
            const unsigned int remain = size - i;
            __mmask16 _mask = (__mmask16)((1u << remain) - 1);
            __m512 _p = _mm512_maskz_loadu_ps(_mask, ptr);
            _p = tanh_avx512(_p);
            _mm512_mask_storeu_ps(ptr, _mask, _p);
        }
#else
#if __SSE2__
#if __AVX__
        for (; i + 7 < size; i += 8)
        {
            __m256 _p = _mm256_loadu_ps(ptr);
            _p = tanh_avx(_p);
            _mm256_storeu_ps(ptr, _p);
            ptr += 8;
        }
#endif
        for (; i + 3 < size; i += 4)
        {
            __m128 _p = _mm_loadu_ps(ptr);
            _p = tanh_sse(_p);
            _mm_storeu_ps(ptr, _p);
            ptr += 4;
        }
#endif
        for (; i < size; i++)
        {
            *ptr = tanhf(*ptr);
            ptr++;
        }
#endif
    }
}