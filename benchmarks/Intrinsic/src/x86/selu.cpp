#include "x86_activation.h"
#include <cmath>

void selu(float* data, int channels, int size, float alpha, float lambda)
{
    for (int q = 0; q < channels; q++)
    {
        float* ptr = data + q * size;

        int i = 0;

#if __SSE2__
#if __AVX__
#if __AVX512F__
        __m512 _zero512 = _mm512_setzero_ps();
        __m512 _one512 = _mm512_set1_ps(1.f);
        __m512 _alpha512 = _mm512_set1_ps(alpha);
        __m512 _lambda512 = _mm512_set1_ps(lambda);
        for (; i + 15 < size; i += 16)
        {
            __m512 _p = _mm512_loadu_ps(ptr);

            __m512 _pos = _mm512_max_ps(_zero512, _p);
            __m512 _neg = _mm512_min_ps(_zero512, _p);

            __m512 _blob = exp512_ps(_neg);
            _blob = _mm512_sub_ps(_blob, _one512);
            _blob = _mm512_mul_ps(_alpha512, _blob);
            _blob = _mm512_mul_ps(_lambda512, _mm512_add_ps(_pos, _blob));

            _mm512_storeu_ps(ptr, _blob);

            ptr += 16;
        }
#endif
        __m256 _zero256 = _mm256_setzero_ps();
        __m256 _one256 = _mm256_set1_ps(1.f);
        __m256 _alpha256 = _mm256_set1_ps(alpha);
        __m256 _lambda256 = _mm256_set1_ps(lambda);
        for (; i + 7 < size; i += 8)
        {
            __m256 _p = _mm256_loadu_ps(ptr);

            __m256 _pos = _mm256_max_ps(_zero256, _p);
            __m256 _neg = _mm256_min_ps(_zero256, _p);

            __m256 _blob = exp256_ps(_neg);
            _blob = _mm256_sub_ps(_blob, _one256);
            _blob = _mm256_mul_ps(_alpha256, _blob);
            _blob = _mm256_mul_ps(_lambda256, _mm256_add_ps(_pos, _blob));

            _mm256_storeu_ps(ptr, _blob);

            ptr += 8;
        }
#endif
        __m128 _zero128 = _mm_setzero_ps();
        __m128 _one128 = _mm_set1_ps(1.f);
        __m128 _alpha128 = _mm_set1_ps(alpha);
        __m128 _lambda128 = _mm_set1_ps(lambda);
        for (; i + 3 < size; i += 4)
        {
            __m128 _p = _mm_loadu_ps(ptr);

            __m128 _pos = _mm_max_ps(_zero128, _p);
            __m128 _neg = _mm_min_ps(_zero128, _p);

            __m128 _blob = exp_ps(_neg);
            _blob = _mm_sub_ps(_blob, _one128);
            _blob = _mm_mul_ps(_alpha128, _blob);
            _blob = _mm_mul_ps(_lambda128, _mm_add_ps(_pos, _blob));

            _mm_storeu_ps(ptr, _blob);

            ptr += 4;
        }
#endif
        float alphaxlambda = alpha * lambda;
        for (; i < size; i++)
        {

            if (*ptr < 0)
                *ptr = (expf(*ptr) - 1.f) * alphaxlambda;
            else
                *ptr = *ptr * lambda;
            ptr++;
        }
    }
}
