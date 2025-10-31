#include "x86_activation.h"
#include <cmath>

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
    else {
        for (int q = 0; q < channels; q++)
        {
            float* ptr = data + q * size;

            int i = 0;

    #if __SSE2__
    #if __AVX__
    #if __AVX512F__
            __m512 _half512 = _mm512_set1_ps(0.5f);
            __m512 _one512 = _mm512_set1_ps(1.f);
            __m512 _fast1c512 = _mm512_set1_ps(0.79788452f);
            __m512 _fast2c512 = _mm512_set1_ps(0.044715f);
            for (; i + 15 < size; i += 16)
            {
                __m512 _pLoad = _mm512_loadu_ps(ptr);

                __m512 _cube = _mm512_mul_ps(_pLoad, _pLoad);
                _cube = _mm512_mul_ps(_pLoad, _cube);

                __m512 _blob = _mm512_mul_ps(_fast2c512, _cube);
                _blob = _mm512_add_ps(_pLoad, _blob);
                _blob = _mm512_mul_ps(_fast1c512, _blob);
                _blob = tanh512_ps(_blob);
                _blob = _mm512_add_ps(_one512, _blob);

                _blob = _mm512_mul_ps(_half512, _mm512_mul_ps(_blob, _pLoad));

                _mm512_storeu_ps(ptr, _blob);

                ptr += 16;
            }
    #endif
            __m256 _half256 = _mm256_set1_ps(0.5f);
            __m256 _one256 = _mm256_set1_ps(1.f);
            __m256 _fast1c256 = _mm256_set1_ps(0.79788452f);
            __m256 _fast2c256 = _mm256_set1_ps(0.044715f);
            for (; i + 7 < size; i += 8)
            {
                __m256 _pLoad = _mm256_loadu_ps(ptr);

                __m256 _cube = _mm256_mul_ps(_pLoad, _pLoad);
                _cube = _mm256_mul_ps(_pLoad, _cube);

                __m256 _blob = _mm256_mul_ps(_fast2c256, _cube);
                _blob = _mm256_add_ps(_pLoad, _blob);
                _blob = _mm256_mul_ps(_fast1c256, _blob);
                _blob = tanh256_ps(_blob);
                _blob = _mm256_add_ps(_one256, _blob);

                _blob = _mm256_mul_ps(_half256, _mm256_mul_ps(_blob, _pLoad));

                _mm256_storeu_ps(ptr, _blob);

                ptr += 8;
            }
    #endif
            __m128 _half128 = _mm_set1_ps(0.5f);
            __m128 _one128 = _mm_set1_ps(1.f);
            __m128 _fast1c128 = _mm_set1_ps(0.79788452f);
            __m128 _fast2c128 = _mm_set1_ps(0.044715f);
            for (; i + 3 < size; i += 4)
            {
                __m128 _pLoad = _mm_loadu_ps(ptr);

                __m128 _cube = _mm_mul_ps(_pLoad, _pLoad);
                _cube = _mm_mul_ps(_pLoad, _cube);

                __m128 _blob = _mm_mul_ps(_fast2c128, _cube);
                _blob = _mm_add_ps(_pLoad, _blob);
                _blob = _mm_mul_ps(_fast1c128, _blob);
                _blob = tanh_ps(_blob);
                _blob = _mm_add_ps(_one128, _blob);

                _blob = _mm_mul_ps(_half128, _mm_mul_ps(_blob, _pLoad));

                _mm_storeu_ps(ptr, _blob);

                ptr += 4;
            }
    #endif
            for (; i < size; i++)
            {

                *ptr = 0.5f * *ptr * (1.0f + tanhf(0.79788452f * (*ptr + 0.044715f * *ptr * *ptr * *ptr)));

                ptr++;
            }
        }
    }
}