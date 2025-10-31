

#ifndef AVX512_MATHFUN_H
#define AVX512_MATHFUN_H

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

#ifdef _MSC_VER
#define ALIGN64_BEG __declspec(align(64))
#define ALIGN64_END
#else
#define ALIGN64_BEG
#define ALIGN64_END __attribute__((aligned(64)))
#endif

#define _PS512_CONST(Name, Val) \
    static const ALIGN64_BEG float _ps512_##Name[16] ALIGN64_END = {Val, Val, Val, Val, Val, Val, Val, Val, Val, Val, Val, Val, Val, Val, Val, Val}
#define _PI32_CONST512(Name, Val) \
    static const ALIGN64_BEG int _pi32_512_##Name[16] ALIGN64_END = {Val, Val, Val, Val, Val, Val, Val, Val, Val, Val, Val, Val, Val, Val, Val, Val}
#define _PS512_CONST_TYPE(Name, Type, Val) \
    static const ALIGN64_BEG Type _ps512_##Name[16] ALIGN64_END = {Val, Val, Val, Val, Val, Val, Val, Val, Val, Val, Val, Val, Val, Val, Val, Val}

_PS512_CONST(1, 1.0f);
_PS512_CONST(0p5, 0.5f);

_PS512_CONST_TYPE(min_norm_pos, int, 0x00800000);
_PS512_CONST_TYPE(mant_mask, int, 0x7f800000);
_PS512_CONST_TYPE(inv_mant_mask, int, ~0x7f800000);

_PS512_CONST_TYPE(sign_mask, int, (int)0x80000000);

_PI32_CONST512(0, 0);
_PI32_CONST512(1, 1);
_PI32_CONST512(inv1, ~1);
_PI32_CONST512(2, 2);
_PI32_CONST512(4, 4);
_PI32_CONST512(0x7f, 0x7f);

_PS512_CONST(cephes_SQRTHF, 0.707106781186547524f);
_PS512_CONST(cephes_log_p0, 7.0376836292E-2f);
_PS512_CONST(cephes_log_p1, -1.1514610310E-1f);
_PS512_CONST(cephes_log_p2, 1.1676998740E-1f);
_PS512_CONST(cephes_log_p3, -1.2420140846E-1f);
_PS512_CONST(cephes_log_p4, +1.4249322787E-1f);
_PS512_CONST(cephes_log_p5, -1.6668057665E-1f);
_PS512_CONST(cephes_log_p6, +2.0000714765E-1f);
_PS512_CONST(cephes_log_p7, -2.4999993993E-1f);
_PS512_CONST(cephes_log_p8, +3.3333331174E-1f);
_PS512_CONST(cephes_log_q1, -2.12194440e-4f);
_PS512_CONST(cephes_log_q2, 0.693359375f);

static NCNN_FORCEINLINE __m512 log512_ps(__m512 x)
{
    __m512i imm0;
    __m512 one = *(__m512*)_ps512_1;

    __mmask16 invalid_mask = _mm512_cmp_ps_mask(x, _mm512_setzero_ps(), _CMP_LE_OQ);

    x = _mm512_max_ps(x, *(__m512*)_ps512_min_norm_pos);

    imm0 = _mm512_srli_epi32(_mm512_castps_si512(x), 23);

    x = _mm512_and_ps(x, *(__m512*)_ps512_inv_mant_mask);
    x = _mm512_or_ps(x, *(__m512*)_ps512_0p5);

    imm0 = _mm512_sub_epi32(imm0, *(__m512i*)_pi32_512_0x7f);
    __m512 e = _mm512_cvtepi32_ps(imm0);

    e = _mm512_add_ps(e, one);

    __mmask16 mask = _mm512_cmp_ps_mask(x, *(__m512*)_ps512_cephes_SQRTHF, _CMP_LT_OQ);
    __m512 tmp = _mm512_sub_ps(x, one);
    e = _mm512_mask_sub_ps(e, mask, e, one);
    x = _mm512_mask_add_ps(tmp, mask, tmp, x);

    __m512 z = _mm512_mul_ps(x, x);

    __m512 y = *(__m512*)_ps512_cephes_log_p0;
    y = _mm512_fmadd_ps(y, x, *(__m512*)_ps512_cephes_log_p1);
    y = _mm512_fmadd_ps(y, x, *(__m512*)_ps512_cephes_log_p2);
    y = _mm512_fmadd_ps(y, x, *(__m512*)_ps512_cephes_log_p3);
    y = _mm512_fmadd_ps(y, x, *(__m512*)_ps512_cephes_log_p4);
    y = _mm512_fmadd_ps(y, x, *(__m512*)_ps512_cephes_log_p5);
    y = _mm512_fmadd_ps(y, x, *(__m512*)_ps512_cephes_log_p6);
    y = _mm512_fmadd_ps(y, x, *(__m512*)_ps512_cephes_log_p7);
    y = _mm512_fmadd_ps(y, x, *(__m512*)_ps512_cephes_log_p8);
    y = _mm512_mul_ps(y, x);

    y = _mm512_mul_ps(y, z);

    y = _mm512_fmadd_ps(e, *(__m512*)_ps512_cephes_log_q1, y);

    y = _mm512_fnmadd_ps(z, *(__m512*)_ps512_0p5, y);

    x = _mm512_add_ps(x, y);
    x = _mm512_fmadd_ps(e, *(__m512*)_ps512_cephes_log_q2, x);
    y = _mm512_or_ps(x, _mm512_castsi512_ps(_mm512_movm_epi32(invalid_mask)));
    return y;
}

_PS512_CONST(exp_hi, 88.3762626647949f);
_PS512_CONST(exp_lo, -88.3762626647949f);

_PS512_CONST(cephes_LOG2EF, 1.44269504088896341f);
_PS512_CONST(cephes_exp_C1, 0.693359375f);
_PS512_CONST(cephes_exp_C2, -2.12194440e-4f);

_PS512_CONST(cephes_exp_p0, 1.9875691500E-4f);
_PS512_CONST(cephes_exp_p1, 1.3981999507E-3f);
_PS512_CONST(cephes_exp_p2, 8.3334519073E-3f);
_PS512_CONST(cephes_exp_p3, 4.1665795894E-2f);
_PS512_CONST(cephes_exp_p4, 1.6666665459E-1f);
_PS512_CONST(cephes_exp_p5, 5.0000001201E-1f);

static NCNN_FORCEINLINE __m512 exp512_ps(__m512 x)
{
    __m512 tmp = _mm512_setzero_ps(), fx;
    __m512i imm0;
    __m512 one = *(__m512*)_ps512_1;

    x = _mm512_min_ps(x, *(__m512*)_ps512_exp_hi);
    x = _mm512_max_ps(x, *(__m512*)_ps512_exp_lo);

    fx = _mm512_fmadd_ps(x, *(__m512*)_ps512_cephes_LOG2EF, *(__m512*)_ps512_0p5);

    tmp = _mm512_roundscale_ps(fx, _MM_FROUND_TO_NEG_INF);

    __mmask16 mask = _mm512_cmp_ps_mask(tmp, fx, _CMP_GT_OQ);
    fx = _mm512_mask_sub_ps(tmp, mask, tmp, one);

    x = _mm512_fnmadd_ps(fx, *(__m512*)_ps512_cephes_exp_C1, x);

    x = _mm512_fnmadd_ps(fx, *(__m512*)_ps512_cephes_exp_C2, x);

    tmp = _mm512_mul_ps(x, x);

    __m512 y = *(__m512*)_ps512_cephes_exp_p0;
    y = _mm512_fmadd_ps(y, x, *(__m512*)_ps512_cephes_exp_p1);
    y = _mm512_fmadd_ps(y, x, *(__m512*)_ps512_cephes_exp_p2);
    y = _mm512_fmadd_ps(y, x, *(__m512*)_ps512_cephes_exp_p3);
    y = _mm512_fmadd_ps(y, x, *(__m512*)_ps512_cephes_exp_p4);
    y = _mm512_fmadd_ps(y, x, *(__m512*)_ps512_cephes_exp_p5);
    y = _mm512_fmadd_ps(y, tmp, x);
    y = _mm512_add_ps(y, one);

    imm0 = _mm512_cvttps_epi32(fx);
    imm0 = _mm512_add_epi32(imm0, *(__m512i*)_pi32_512_0x7f);
    imm0 = _mm512_slli_epi32(imm0, 23);
    __m512 pow2n = _mm512_castsi512_ps(imm0);
    y = _mm512_mul_ps(y, pow2n);
    return y;
}

_PS512_CONST(tanh_hi, 9.0f);
_PS512_CONST(tanh_lo, -9.0f);

_PS512_CONST(cephes_tanh_p0, -2.76076847742355E-16f);
_PS512_CONST(cephes_tanh_p1, 2.00018790482477E-13f);
_PS512_CONST(cephes_tanh_p2, -8.60467152213735E-11f);
_PS512_CONST(cephes_tanh_p3, 5.12229709037114E-08f);
_PS512_CONST(cephes_tanh_p4, 1.48572235717979E-05f);
_PS512_CONST(cephes_tanh_p5, 6.37261928875436E-04f);
_PS512_CONST(cephes_tanh_p6, 4.89352455891786E-03f);

_PS512_CONST(cephes_tanh_p7, 1.19825839466702e-06f);
_PS512_CONST(cephes_tanh_p8, 1.18534705686654e-04f);
_PS512_CONST(cephes_tanh_p9, 2.26843463243900e-03f);

static inline __m512 tanh512_ps(const __m512 x)
{
    __m512 value = x;
    value = _mm512_max_ps(*(__m512*)_ps512_tanh_lo, value);
    value = _mm512_min_ps(*(__m512*)_ps512_tanh_hi, value);

    __m512 value_squared = _mm512_mul_ps(value, value);

    __m512 p;
    p = _mm512_fmadd_ps(value_squared, *(__m512*)_ps512_cephes_tanh_p0, *(__m512*)_ps512_cephes_tanh_p1);
    p = _mm512_fmadd_ps(p, value_squared, *(__m512*)_ps512_cephes_tanh_p2);
    p = _mm512_fmadd_ps(p, value_squared, *(__m512*)_ps512_cephes_tanh_p3);
    p = _mm512_fmadd_ps(p, value_squared, *(__m512*)_ps512_cephes_tanh_p4);
    p = _mm512_fmadd_ps(p, value_squared, *(__m512*)_ps512_cephes_tanh_p5);
    p = _mm512_fmadd_ps(p, value_squared, *(__m512*)_ps512_cephes_tanh_p6);
    p = _mm512_mul_ps(p, value);

    __m512 q;
    q = _mm512_fmadd_ps(value_squared, *(__m512*)_ps512_cephes_tanh_p7, *(__m512*)_ps512_cephes_tanh_p8);
    q = _mm512_fmadd_ps(q, value_squared, *(__m512*)_ps512_cephes_tanh_p9);
    q = _mm512_fmadd_ps(q, value_squared, *(__m512*)_ps512_cephes_tanh_p6);

    __m512 dst = _mm512_div_ps(p, q);
    return dst;
}

#endif
