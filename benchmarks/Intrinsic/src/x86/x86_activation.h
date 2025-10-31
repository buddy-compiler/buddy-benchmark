

#ifndef X86_ACTIVATION_H
#define X86_ACTIVATION_H

#ifdef _MSC_VER
    #define NCNN_FORCEINLINE __forceinline
#elif defined(__GNUC__)
    #define NCNN_FORCEINLINE inline __attribute__((__always_inline__))
#elif defined(__CLANG__)
    #if __has_attribute(__always_inline__)
        #define NCNN_FORCEINLINE inline __attribute__((__always_inline__))
    #else
        #define NCNN_FORCEINLINE inline
    #endif
#else
    #define NCNN_FORCEINLINE inline
#endif

#if __SSE2__
#include <emmintrin.h>
#include "sse_mathfun.h"

static NCNN_FORCEINLINE __m128 sigmoid_sse(__m128 inputs)
{
    const __m128 one = _mm_set1_ps(1.0f);
    return _mm_div_ps(one, _mm_add_ps(one, exp_ps(_mm_sub_ps(_mm_setzero_ps(), inputs))));
}

static NCNN_FORCEINLINE __m128 tanh_sse(__m128 inputs)
{
    const __m128 one = _mm_set1_ps(1.0f);
    const __m128 two = _mm_set1_ps(2.0f);
    return _mm_sub_ps(_mm_mul_ps(sigmoid_sse(_mm_mul_ps(inputs, two)), two), one);
}

static NCNN_FORCEINLINE __m128 mish_sse(__m128 inputs)
{
    return _mm_mul_ps(inputs, tanh_sse(log_ps(_mm_add_ps(exp_ps(inputs), _mm_set1_ps(1.f)))));
}

static NCNN_FORCEINLINE __m128 swish_sse(__m128 inputs)
{
    return _mm_mul_ps(inputs, sigmoid_sse(inputs));
}

static NCNN_FORCEINLINE __m128 hardswish_sse(__m128 inputs, __m128 a, __m128 b)
{
    const __m128 one = _mm_set1_ps(1.0f);
    b = _mm_add_ps(_mm_mul_ps(inputs, a), b);
    b = _mm_max_ps(b, _mm_setzero_ps());
    b = _mm_min_ps(b, one);
    return _mm_mul_ps(b, inputs);
}

static NCNN_FORCEINLINE __m128 lrelu_sse(__m128 inputs, float slope)
{
    __m128 pos = _mm_max_ps(_mm_setzero_ps(), inputs);
    __m128 neg = _mm_min_ps(_mm_setzero_ps(), inputs);
    return _mm_add_ps(pos, _mm_mul_ps(_mm_set1_ps(slope), neg));
}

static NCNN_FORCEINLINE __m128 prelu_sse(__m128 inputs, __m128 alphas)
{
    __m128 pos = _mm_max_ps(_mm_setzero_ps(), inputs);
    __m128 neg = _mm_min_ps(_mm_setzero_ps(), inputs);
    return _mm_add_ps(pos, _mm_mul_ps(alphas, neg));
}

static NCNN_FORCEINLINE __m128 elu_sse(__m128 inputs, __m128 alphas)
{
    __m128 pos = _mm_max_ps(_mm_setzero_ps(), inputs);
    __m128 neg = _mm_min_ps(_mm_setzero_ps(), inputs);
    neg = _mm_sub_ps(exp_ps(neg), _mm_set1_ps(1.f));
    return _mm_add_ps(pos, _mm_mul_ps(alphas, neg));
}

#if __AVX__
#include <immintrin.h>
#include "avx_mathfun.h"

static NCNN_FORCEINLINE __m256 sigmoid_avx(__m256 inputs)
{
    const __m256 one = _mm256_set1_ps(1.0f);
    return _mm256_div_ps(one, _mm256_add_ps(one, exp256_ps(_mm256_sub_ps(_mm256_setzero_ps(), inputs))));
}

static NCNN_FORCEINLINE __m256 tanh_avx(__m256 inputs)
{
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 two = _mm256_set1_ps(2.0f);
#if __FMA__
    return _mm256_fmsub_ps(sigmoid_avx(_mm256_mul_ps(inputs, two)), two, one);
#else
    return _mm256_sub_ps(_mm256_mul_ps(sigmoid_avx(_mm256_mul_ps(inputs, two)), two), one);
#endif
}

static NCNN_FORCEINLINE __m256 mish_avx(__m256 inputs)
{
    return _mm256_mul_ps(inputs, tanh_avx(log256_ps(_mm256_add_ps(exp256_ps(inputs), _mm256_set1_ps(1.f)))));
}

static NCNN_FORCEINLINE __m256 swish_avx(__m256 inputs)
{
    return _mm256_mul_ps(inputs, sigmoid_avx(inputs));
}

static NCNN_FORCEINLINE __m256 hardswish_avx(__m256 inputs, __m256 a, __m256 b)
{
    const __m256 one = _mm256_set1_ps(1.0f);
    b = _mm256_comp_fmadd_ps(inputs, a, b);
    b = _mm256_max_ps(b, _mm256_setzero_ps());
    b = _mm256_min_ps(b, one);
    return _mm256_mul_ps(b, inputs);
}

static NCNN_FORCEINLINE __m256 lrelu_avx(__m256 inputs, float slope)
{
    __m256 pos = _mm256_max_ps(_mm256_setzero_ps(), inputs);
    __m256 neg = _mm256_min_ps(_mm256_setzero_ps(), inputs);
    return _mm256_add_ps(pos, _mm256_mul_ps(_mm256_set1_ps(slope), neg));
}

static NCNN_FORCEINLINE __m256 prelu_avx(__m256 inputs, __m256 alphas)
{
    __m256 pos = _mm256_max_ps(_mm256_setzero_ps(), inputs);
    __m256 neg = _mm256_min_ps(_mm256_setzero_ps(), inputs);
    return _mm256_add_ps(pos, _mm256_mul_ps(alphas, neg));
}

static NCNN_FORCEINLINE __m256 elu_avx(__m256 inputs, __m256 alphas)
{
    __m256 pos = _mm256_max_ps(_mm256_setzero_ps(), inputs);
    __m256 neg = _mm256_min_ps(_mm256_setzero_ps(), inputs);
    neg = _mm256_sub_ps(exp256_ps(neg), _mm256_set1_ps(1.f));
    return _mm256_add_ps(pos, _mm256_mul_ps(alphas, neg));
}

#if __AVX512F__
#include "avx512_mathfun.h"

static NCNN_FORCEINLINE __m512 sigmoid_avx512(__m512 inputs)
{
    const __m512 one = _mm512_set1_ps(1.0f);
    return _mm512_div_ps(one, _mm512_add_ps(one, exp512_ps(_mm512_sub_ps(_mm512_setzero_ps(), inputs))));
}

static NCNN_FORCEINLINE __m512 tanh_avx512(__m512 inputs)
{
    const __m512 one = _mm512_set1_ps(1.0f);
    const __m512 two = _mm512_set1_ps(2.0f);
    return _mm512_fmsub_ps(sigmoid_avx512(_mm512_mul_ps(inputs, two)), two, one);
}

static NCNN_FORCEINLINE __m512 mish_avx512(__m512 inputs)
{
    return _mm512_mul_ps(inputs, tanh_avx512(log512_ps(_mm512_add_ps(exp512_ps(inputs), _mm512_set1_ps(1.f)))));
}

static NCNN_FORCEINLINE __m512 swish_avx512(__m512 inputs)
{
    return _mm512_mul_ps(inputs, sigmoid_avx512(inputs));
}

static NCNN_FORCEINLINE __m512 hardswish_avx512(__m512 inputs, __m512 a, __m512 b)
{
    const __m512 one = _mm512_set1_ps(1.0f);
    b = _mm512_fmadd_ps(inputs, a, b);
    b = _mm512_max_ps(b, _mm512_setzero_ps());
    b = _mm512_min_ps(b, one);
    return _mm512_mul_ps(b, inputs);
}

static NCNN_FORCEINLINE __m512 lrelu_avx512(__m512 inputs, float slope)
{
    __mmask16 _is_negative = _mm512_cmp_ps_mask(inputs, _mm512_setzero_ps(), _CMP_LT_OQ);
    return _mm512_mask_mul_ps(inputs, _is_negative, inputs, _mm512_set1_ps(slope));
}

static NCNN_FORCEINLINE __m512 elu_avx512(__m512 inputs, __m512 alphas)
{
    __m512 pos = _mm512_max_ps(_mm512_setzero_ps(), inputs);
    __m512 neg = _mm512_min_ps(_mm512_setzero_ps(), inputs);
    neg = _mm512_sub_ps(exp512_ps(neg), _mm512_set1_ps(1.f));
    return _mm512_add_ps(pos, _mm512_mul_ps(alphas, neg));
}

static NCNN_FORCEINLINE __m512 prelu_avx512(__m512 inputs, __m512 alphas)
{
    __m512 pos = _mm512_max_ps(_mm512_setzero_ps(), inputs);
    __m512 neg = _mm512_min_ps(_mm512_setzero_ps(), inputs);
    return _mm512_add_ps(pos, _mm512_mul_ps(alphas, neg));
}

#endif
#endif
#endif

#endif
