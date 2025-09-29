

#ifndef AVX_MATHFUN_H
#define AVX_MATHFUN_H

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

#include <immintrin.h>
#include <emmintrin.h>
#include "x86_usability.h"

#ifdef _MSC_VER
#define ALIGN32_BEG __declspec(align(32))
#define ALIGN32_END
#else
#define ALIGN32_BEG
#define ALIGN32_END __attribute__((aligned(32)))
#endif

#define _PI32AVX_CONST(Name, Val) \
    static const ALIGN32_BEG int _pi32avx_##Name[4] ALIGN32_END = {Val, Val, Val, Val}

_PI32AVX_CONST(1, 1);
_PI32AVX_CONST(inv1, ~1);
_PI32AVX_CONST(2, 2);
_PI32AVX_CONST(4, 4);

#define _PS256_CONST(Name, Val) \
    static const ALIGN32_BEG float _ps256_##Name[8] ALIGN32_END = {Val, Val, Val, Val, Val, Val, Val, Val}
#define _PI32_CONST256(Name, Val) \
    static const ALIGN32_BEG int _pi32_256_##Name[8] ALIGN32_END = {Val, Val, Val, Val, Val, Val, Val, Val}
#define _PS256_CONST_TYPE(Name, Type, Val) \
    static const ALIGN32_BEG Type _ps256_##Name[8] ALIGN32_END = {Val, Val, Val, Val, Val, Val, Val, Val}

_PS256_CONST(1, 1.0f);
_PS256_CONST(0p5, 0.5f);

_PS256_CONST_TYPE(min_norm_pos, int, 0x00800000);
_PS256_CONST_TYPE(mant_mask, int, 0x7f800000);
_PS256_CONST_TYPE(inv_mant_mask, int, ~0x7f800000);

_PS256_CONST_TYPE(sign_mask, int, (int)0x80000000);
_PS256_CONST_TYPE(inv_sign_mask, int, ~0x80000000);

_PI32_CONST256(0, 0);
_PI32_CONST256(1, 1);
_PI32_CONST256(inv1, ~1);
_PI32_CONST256(2, 2);
_PI32_CONST256(4, 4);
_PI32_CONST256(0x7f, 0x7f);

_PS256_CONST(cephes_SQRTHF, 0.707106781186547524f);
_PS256_CONST(cephes_log_p0, 7.0376836292E-2f);
_PS256_CONST(cephes_log_p1, -1.1514610310E-1f);
_PS256_CONST(cephes_log_p2, 1.1676998740E-1f);
_PS256_CONST(cephes_log_p3, -1.2420140846E-1f);
_PS256_CONST(cephes_log_p4, +1.4249322787E-1f);
_PS256_CONST(cephes_log_p5, -1.6668057665E-1f);
_PS256_CONST(cephes_log_p6, +2.0000714765E-1f);
_PS256_CONST(cephes_log_p7, -2.4999993993E-1f);
_PS256_CONST(cephes_log_p8, +3.3333331174E-1f);
_PS256_CONST(cephes_log_q1, -2.12194440e-4f);
_PS256_CONST(cephes_log_q2, 0.693359375f);

#ifndef __AVX2__
typedef union imm_xmm_union
{
    __m256i imm;
    __m128i xmm[2];
} imm_xmm_union;

#define COPY_IMM_TO_XMM(imm_, xmm0_, xmm1_)      \
    {                                            \
        ALIGN32_BEG imm_xmm_union u ALIGN32_END; \
        u.imm = imm_;                            \
        xmm0_ = u.xmm[0];                        \
        xmm1_ = u.xmm[1];                        \
    }

#define COPY_XMM_TO_IMM(xmm0_, xmm1_, imm_)      \
    {                                            \
        ALIGN32_BEG imm_xmm_union u ALIGN32_END; \
        u.xmm[0] = xmm0_;                        \
        u.xmm[1] = xmm1_;                        \
        imm_ = u.imm;                            \
    }

#define AVX2_BITOP_USING_SSE2(fn)                                      \
    static NCNN_FORCEINLINE __m256i _mm256_comp_##fn(__m256i x, int a) \
    {                                                                  \
           \
        __m128i x1, x2;                                                \
        __m256i ret;                                                   \
        COPY_IMM_TO_XMM(x, x1, x2);                                    \
        x1 = _mm_##fn(x1, a);                                          \
        x2 = _mm_##fn(x2, a);                                          \
        COPY_XMM_TO_IMM(x1, x2, ret);                                  \
        return (ret);                                                  \
    }
#define AVX2_INTOP_USING_SSE2(fn)                                          \
    static NCNN_FORCEINLINE __m256i _mm256_comp_##fn(__m256i x, __m256i y) \
    {                                                                      \
  \
        __m128i x1, x2;                                                    \
        __m128i y1, y2;                                                    \
        __m256i ret;                                                       \
        COPY_IMM_TO_XMM(x, x1, x2);                                        \
        COPY_IMM_TO_XMM(y, y1, y2);                                        \
        x1 = _mm_##fn(x1, y1);                                             \
        x2 = _mm_##fn(x2, y2);                                             \
        COPY_XMM_TO_IMM(x1, x2, ret);                                      \
        return (ret);                                                      \
    }
#else
#define AVX2_BITOP_USING_SSE2(fn)                                      \
    static NCNN_FORCEINLINE __m256i _mm256_comp_##fn(__m256i x, int a) \
    {                                                                  \
        return _mm256_##fn(x, a);                                      \
    }
#define AVX2_INTOP_USING_SSE2(fn)                                          \
    static NCNN_FORCEINLINE __m256i _mm256_comp_##fn(__m256i x, __m256i y) \
    {                                                                      \
        return _mm256_##fn(x, y);                                          \
    }
#endif

AVX2_BITOP_USING_SSE2(slli_epi32)
AVX2_BITOP_USING_SSE2(srli_epi32)
AVX2_INTOP_USING_SSE2(cmpeq_epi32)
AVX2_INTOP_USING_SSE2(sub_epi32)
AVX2_INTOP_USING_SSE2(add_epi32)

#ifndef __AVX2__
AVX2_INTOP_USING_SSE2(and_si128)
AVX2_INTOP_USING_SSE2(andnot_si128)
#endif

static NCNN_FORCEINLINE __m256 log256_ps(__m256 x)
{
    __m256i imm0;
    __m256 one = *(__m256*)_ps256_1;

    __m256 invalid_mask = _mm256_cmp_ps(x, _mm256_setzero_ps(), _CMP_LE_OS);

    x = _mm256_max_ps(x, *(__m256*)_ps256_min_norm_pos);

    imm0 = _mm256_comp_srli_epi32(_mm256_castps_si256(x), 23);

    x = _mm256_and_ps(x, *(__m256*)_ps256_inv_mant_mask);
    x = _mm256_or_ps(x, *(__m256*)_ps256_0p5);

    imm0 = _mm256_comp_sub_epi32(imm0, *(__m256i*)_pi32_256_0x7f);
    __m256 e = _mm256_cvtepi32_ps(imm0);

    e = _mm256_add_ps(e, one);

    __m256 mask = _mm256_cmp_ps(x, *(__m256*)_ps256_cephes_SQRTHF, _CMP_LT_OS);
    __m256 tmp = _mm256_and_ps(x, mask);
    x = _mm256_sub_ps(x, one);
    e = _mm256_sub_ps(e, _mm256_and_ps(one, mask));
    x = _mm256_add_ps(x, tmp);

    __m256 z = _mm256_mul_ps(x, x);

    __m256 y = *(__m256*)_ps256_cephes_log_p0;
    y = _mm256_comp_fmadd_ps(y, x, *(__m256*)_ps256_cephes_log_p1);
    y = _mm256_comp_fmadd_ps(y, x, *(__m256*)_ps256_cephes_log_p2);
    y = _mm256_comp_fmadd_ps(y, x, *(__m256*)_ps256_cephes_log_p3);
    y = _mm256_comp_fmadd_ps(y, x, *(__m256*)_ps256_cephes_log_p4);
    y = _mm256_comp_fmadd_ps(y, x, *(__m256*)_ps256_cephes_log_p5);
    y = _mm256_comp_fmadd_ps(y, x, *(__m256*)_ps256_cephes_log_p6);
    y = _mm256_comp_fmadd_ps(y, x, *(__m256*)_ps256_cephes_log_p7);
    y = _mm256_comp_fmadd_ps(y, x, *(__m256*)_ps256_cephes_log_p8);
    y = _mm256_mul_ps(y, x);

    y = _mm256_mul_ps(y, z);

    y = _mm256_comp_fmadd_ps(e, *(__m256*)_ps256_cephes_log_q1, y);

    y = _mm256_comp_fnmadd_ps(z, *(__m256*)_ps256_0p5, y);

    x = _mm256_add_ps(x, y);
    x = _mm256_comp_fmadd_ps(e, *(__m256*)_ps256_cephes_log_q2, x);
    y = _mm256_or_ps(x, invalid_mask);
    return y;
}

_PS256_CONST(exp_hi, 88.3762626647949f);
_PS256_CONST(exp_lo, -88.3762626647949f);

_PS256_CONST(cephes_LOG2EF, 1.44269504088896341f);
_PS256_CONST(cephes_exp_C1, 0.693359375f);
_PS256_CONST(cephes_exp_C2, -2.12194440e-4f);

_PS256_CONST(cephes_exp_p0, 1.9875691500E-4f);
_PS256_CONST(cephes_exp_p1, 1.3981999507E-3f);
_PS256_CONST(cephes_exp_p2, 8.3334519073E-3f);
_PS256_CONST(cephes_exp_p3, 4.1665795894E-2f);
_PS256_CONST(cephes_exp_p4, 1.6666665459E-1f);
_PS256_CONST(cephes_exp_p5, 5.0000001201E-1f);

static NCNN_FORCEINLINE __m256 exp256_ps(__m256 x)
{
    __m256 tmp = _mm256_setzero_ps(), fx;
    __m256i imm0;
    __m256 one = *(__m256*)_ps256_1;

    x = _mm256_min_ps(x, *(__m256*)_ps256_exp_hi);
    x = _mm256_max_ps(x, *(__m256*)_ps256_exp_lo);

    fx = _mm256_comp_fmadd_ps(x, *(__m256*)_ps256_cephes_LOG2EF, *(__m256*)_ps256_0p5);

    tmp = _mm256_floor_ps(fx);

    __m256 mask = _mm256_cmp_ps(tmp, fx, _CMP_GT_OS);
    mask = _mm256_and_ps(mask, one);
    fx = _mm256_sub_ps(tmp, mask);

    x = _mm256_comp_fnmadd_ps(fx, *(__m256*)_ps256_cephes_exp_C1, x);

    x = _mm256_comp_fnmadd_ps(fx, *(__m256*)_ps256_cephes_exp_C2, x);

    tmp = _mm256_mul_ps(x, x);

    __m256 y = *(__m256*)_ps256_cephes_exp_p0;
    y = _mm256_comp_fmadd_ps(y, x, *(__m256*)_ps256_cephes_exp_p1);
    y = _mm256_comp_fmadd_ps(y, x, *(__m256*)_ps256_cephes_exp_p2);
    y = _mm256_comp_fmadd_ps(y, x, *(__m256*)_ps256_cephes_exp_p3);
    y = _mm256_comp_fmadd_ps(y, x, *(__m256*)_ps256_cephes_exp_p4);
    y = _mm256_comp_fmadd_ps(y, x, *(__m256*)_ps256_cephes_exp_p5);
    y = _mm256_comp_fmadd_ps(y, tmp, x);
    y = _mm256_add_ps(y, one);

    imm0 = _mm256_cvttps_epi32(fx);

    imm0 = _mm256_comp_add_epi32(imm0, *(__m256i*)_pi32_256_0x7f);
    imm0 = _mm256_comp_slli_epi32(imm0, 23);
    __m256 pow2n = _mm256_castsi256_ps(imm0);
    y = _mm256_mul_ps(y, pow2n);
    return y;
}

_PS256_CONST(tanh_hi, 9.0f);
_PS256_CONST(tanh_lo, -9.0f);

_PS256_CONST(cephes_tanh_p0, -2.76076847742355E-16f);
_PS256_CONST(cephes_tanh_p1, 2.00018790482477E-13f);
_PS256_CONST(cephes_tanh_p2, -8.60467152213735E-11f);
_PS256_CONST(cephes_tanh_p3, 5.12229709037114E-08f);
_PS256_CONST(cephes_tanh_p4, 1.48572235717979E-05f);
_PS256_CONST(cephes_tanh_p5, 6.37261928875436E-04f);
_PS256_CONST(cephes_tanh_p6, 4.89352455891786E-03f);

_PS256_CONST(cephes_tanh_p7, 1.19825839466702e-06f);
_PS256_CONST(cephes_tanh_p8, 1.18534705686654e-04f);
_PS256_CONST(cephes_tanh_p9, 2.26843463243900e-03f);

static NCNN_FORCEINLINE __m256 tanh256_ps(__m256 x)
{
    __m256 value = x;
    value = _mm256_max_ps(*(__m256*)_ps256_tanh_lo, value);
    value = _mm256_min_ps(*(__m256*)_ps256_tanh_hi, value);

    __m256 value_squared = _mm256_mul_ps(value, value);

    __m256 p;
    p = _mm256_comp_fmadd_ps(value_squared, *(__m256*)_ps256_cephes_tanh_p0, *(__m256*)_ps256_cephes_tanh_p1);
    p = _mm256_comp_fmadd_ps(p, value_squared, *(__m256*)_ps256_cephes_tanh_p2);
    p = _mm256_comp_fmadd_ps(p, value_squared, *(__m256*)_ps256_cephes_tanh_p3);
    p = _mm256_comp_fmadd_ps(p, value_squared, *(__m256*)_ps256_cephes_tanh_p4);
    p = _mm256_comp_fmadd_ps(p, value_squared, *(__m256*)_ps256_cephes_tanh_p5);
    p = _mm256_comp_fmadd_ps(p, value_squared, *(__m256*)_ps256_cephes_tanh_p6);
    p = _mm256_mul_ps(p, value);

    __m256 q;
    q = _mm256_comp_fmadd_ps(value_squared, *(__m256*)_ps256_cephes_tanh_p7, *(__m256*)_ps256_cephes_tanh_p8);
    q = _mm256_comp_fmadd_ps(q, value_squared, *(__m256*)_ps256_cephes_tanh_p9);
    q = _mm256_comp_fmadd_ps(q, value_squared, *(__m256*)_ps256_cephes_tanh_p6);

    __m256 dst = _mm256_div_ps(p, q);
    return dst;
}

#endif
