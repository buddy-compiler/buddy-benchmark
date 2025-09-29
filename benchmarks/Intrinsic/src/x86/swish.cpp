#include "x86_activation.h"
#include <cmath>

void swish(float* data, int channels, int size)
{
    for (int q = 0; q < channels; q++)
    {
        float* ptr = data + q * size;

        int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
        __m512 _one_avx512 = _mm512_set1_ps(1.f);
        __m512 _zero_avx512 = _mm512_setzero_ps();
        for (; i + 15 < size; i += 16)
        {
            __m512 _p = _mm512_loadu_ps(ptr);
            _p = _mm512_div_ps(_p, _mm512_add_ps(_one_avx512, exp512_ps(_mm512_sub_ps(_zero_avx512, _p))));
            _mm512_storeu_ps(ptr, _p);
            ptr += 16;
        }
#endif
        __m256 _one_avx = _mm256_set1_ps(1.f);
        __m256 _zero_avx = _mm256_setzero_ps();
        for (; i + 7 < size; i += 8)
        {
            __m256 _p = _mm256_loadu_ps(ptr);
            _p = _mm256_div_ps(_p, _mm256_add_ps(_one_avx, exp256_ps(_mm256_sub_ps(_zero_avx, _p))));
            _mm256_storeu_ps(ptr, _p);
            ptr += 8;
        }
#endif
        __m128 _one = _mm_set1_ps(1.f);
        __m128 _zero = _mm_setzero_ps();
        for (; i + 3 < size; i += 4)
        {
            __m128 _p = _mm_loadu_ps(ptr);
            _p = _mm_div_ps(_p, _mm_add_ps(_one, exp_ps(_mm_sub_ps(_zero, _p))));
            _mm_storeu_ps(ptr, _p);
            ptr += 4;
        }
#endif
        for (; i < size; i++)
        {
            *ptr = *ptr / (1.f + expf(-*ptr));
            ptr++;
        }
    }
}