

#ifndef SSE_MATHFUN_H
#define SSE_MATHFUN_H

#define USE_SSE2 1

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

#include <xmmintrin.h>
#include "x86_usability.h"

#ifdef _MSC_VER
#define ALIGN16_BEG __declspec(align(16))
#define ALIGN16_END
#else
#define ALIGN16_BEG
#define ALIGN16_END __attribute__((aligned(16)))
#endif

typedef __m128 v4sf;

#ifdef USE_SSE2
#include <emmintrin.h>
typedef __m128i v4si;
#else
typedef __m64 v2si;
#endif

#define _PS_CONST(Name, Val) \
    static const ALIGN16_BEG float _ps_##Name[4] ALIGN16_END = {Val, Val, Val, Val}
#define _PI32_CONST(Name, Val) \
    static const ALIGN16_BEG int _pi32_##Name[4] ALIGN16_END = {Val, Val, Val, Val}
#define _PS_CONST_TYPE(Name, Type, Val) \
    static const ALIGN16_BEG Type _ps_##Name[4] ALIGN16_END = {Val, Val, Val, Val}

_PS_CONST(1, 1.0f);
_PS_CONST(0p5, 0.5f);

_PS_CONST_TYPE(min_norm_pos, int, 0x00800000);
_PS_CONST_TYPE(mant_mask, int, 0x7f800000);
_PS_CONST_TYPE(inv_mant_mask, int, ~0x7f800000);

_PS_CONST_TYPE(sign_mask, int, (int)0x80000000);
_PS_CONST_TYPE(inv_sign_mask, int, ~0x80000000);

_PI32_CONST(1, 1);
_PI32_CONST(inv1, ~1);
_PI32_CONST(2, 2);
_PI32_CONST(4, 4);
_PI32_CONST(0x7f, 0x7f);

_PS_CONST(cephes_SQRTHF, 0.707106781186547524f);
_PS_CONST(cephes_log_p0, 7.0376836292E-2f);
_PS_CONST(cephes_log_p1, -1.1514610310E-1f);
_PS_CONST(cephes_log_p2, 1.1676998740E-1f);
_PS_CONST(cephes_log_p3, -1.2420140846E-1f);
_PS_CONST(cephes_log_p4, +1.4249322787E-1f);
_PS_CONST(cephes_log_p5, -1.6668057665E-1f);
_PS_CONST(cephes_log_p6, +2.0000714765E-1f);
_PS_CONST(cephes_log_p7, -2.4999993993E-1f);
_PS_CONST(cephes_log_p8, +3.3333331174E-1f);
_PS_CONST(cephes_log_q1, -2.12194440e-4f);
_PS_CONST(cephes_log_q2, 0.693359375f);

#ifndef USE_SSE2
typedef union xmm_mm_union
{
    __m128 xmm;
    __m64 mm[2];
} xmm_mm_union;

#define COPY_XMM_TO_MM(xmm_, mm0_, mm1_) \
    {                                    \
        xmm_mm_union u;                  \
        u.xmm = xmm_;                    \
        mm0_ = u.mm[0];                  \
        mm1_ = u.mm[1];                  \
    }

#define COPY_MM_TO_XMM(mm0_, mm1_, xmm_) \
    {                                    \
        xmm_mm_union u;                  \
        u.mm[0] = mm0_;                  \
        u.mm[1] = mm1_;                  \
        xmm_ = u.xmm;                    \
    }

#endif

static NCNN_FORCEINLINE v4sf log_ps(v4sf x)
{
#ifdef USE_SSE2
    v4si emm0;
#else
    v2si mm0, mm1;
#endif
    v4sf one = *(v4sf*)_ps_1;

    v4sf invalid_mask = _mm_cmple_ps(x, _mm_setzero_ps());

    x = _mm_max_ps(x, *(v4sf*)_ps_min_norm_pos);

#ifndef USE_SSE2

    COPY_XMM_TO_MM(x, mm0, mm1);
    mm0 = _mm_srli_pi32(mm0, 23);
    mm1 = _mm_srli_pi32(mm1, 23);
#else
    emm0 = _mm_srli_epi32(_mm_castps_si128(x), 23);
#endif

    x = _mm_and_ps(x, *(v4sf*)_ps_inv_mant_mask);
    x = _mm_or_ps(x, *(v4sf*)_ps_0p5);

#ifndef USE_SSE2

    mm0 = _mm_sub_pi32(mm0, *(v2si*)_pi32_0x7f);
    mm1 = _mm_sub_pi32(mm1, *(v2si*)_pi32_0x7f);
    v4sf e = _mm_cvtpi32x2_ps(mm0, mm1);
    _mm_empty();
#else
    emm0 = _mm_sub_epi32(emm0, *(v4si*)_pi32_0x7f);
    v4sf e = _mm_cvtepi32_ps(emm0);
#endif

    e = _mm_add_ps(e, one);

    v4sf mask = _mm_cmplt_ps(x, *(v4sf*)_ps_cephes_SQRTHF);
    v4sf tmp = _mm_and_ps(x, mask);
    x = _mm_sub_ps(x, one);
    e = _mm_sub_ps(e, _mm_and_ps(one, mask));
    x = _mm_add_ps(x, tmp);

    v4sf z = _mm_mul_ps(x, x);

    v4sf y = *(v4sf*)_ps_cephes_log_p0;
    y = _mm_comp_fmadd_ps(y, x, *(v4sf*)_ps_cephes_log_p1);
    y = _mm_comp_fmadd_ps(y, x, *(v4sf*)_ps_cephes_log_p2);
    y = _mm_comp_fmadd_ps(y, x, *(v4sf*)_ps_cephes_log_p3);
    y = _mm_comp_fmadd_ps(y, x, *(v4sf*)_ps_cephes_log_p4);
    y = _mm_comp_fmadd_ps(y, x, *(v4sf*)_ps_cephes_log_p5);
    y = _mm_comp_fmadd_ps(y, x, *(v4sf*)_ps_cephes_log_p6);
    y = _mm_comp_fmadd_ps(y, x, *(v4sf*)_ps_cephes_log_p7);
    y = _mm_comp_fmadd_ps(y, x, *(v4sf*)_ps_cephes_log_p8);
    y = _mm_mul_ps(y, x);

    y = _mm_mul_ps(y, z);

    y = _mm_comp_fmadd_ps(e, *(v4sf*)_ps_cephes_log_q1, y);

    y = _mm_comp_fnmadd_ps(z, *(v4sf*)_ps_0p5, y);

    x = _mm_add_ps(x, y);
    x = _mm_comp_fmadd_ps(e, *(v4sf*)_ps_cephes_log_q2, x);
    x = _mm_or_ps(x, invalid_mask);
    return x;
}

_PS_CONST(exp_hi, 88.3762626647949f);
_PS_CONST(exp_lo, -88.3762626647949f);

_PS_CONST(cephes_LOG2EF, 1.44269504088896341f);
_PS_CONST(cephes_exp_C1, 0.693359375f);
_PS_CONST(cephes_exp_C2, -2.12194440e-4f);

_PS_CONST(cephes_exp_p0, 1.9875691500E-4f);
_PS_CONST(cephes_exp_p1, 1.3981999507E-3f);
_PS_CONST(cephes_exp_p2, 8.3334519073E-3f);
_PS_CONST(cephes_exp_p3, 4.1665795894E-2f);
_PS_CONST(cephes_exp_p4, 1.6666665459E-1f);
_PS_CONST(cephes_exp_p5, 5.0000001201E-1f);

static NCNN_FORCEINLINE v4sf exp_ps(v4sf x)
{
    v4sf tmp = _mm_setzero_ps(), fx;
#ifdef USE_SSE2
    v4si emm0;
#else
    v2si mm0, mm1;
#endif
    v4sf one = *(v4sf*)_ps_1;

    x = _mm_min_ps(x, *(v4sf*)_ps_exp_hi);
    x = _mm_max_ps(x, *(v4sf*)_ps_exp_lo);

    fx = _mm_mul_ps(x, *(v4sf*)_ps_cephes_LOG2EF);
    fx = _mm_add_ps(fx, *(v4sf*)_ps_0p5);

#ifndef USE_SSE2

    tmp = _mm_movehl_ps(tmp, fx);
    mm0 = _mm_cvttps_pi32(fx);
    mm1 = _mm_cvttps_pi32(tmp);

    tmp = _mm_cvtpi32x2_ps(mm0, mm1);
#else
    emm0 = _mm_cvttps_epi32(fx);
    tmp = _mm_cvtepi32_ps(emm0);
#endif

    v4sf mask = _mm_cmpgt_ps(tmp, fx);
    mask = _mm_and_ps(mask, one);
    fx = _mm_sub_ps(tmp, mask);

    x = _mm_comp_fnmadd_ps(fx, *(v4sf*)_ps_cephes_exp_C1, x);

    x = _mm_comp_fnmadd_ps(fx, *(v4sf*)_ps_cephes_exp_C2, x);

    tmp = _mm_mul_ps(x, x);

    v4sf y = *(v4sf*)_ps_cephes_exp_p0;
    y = _mm_comp_fmadd_ps(y, x, *(v4sf*)_ps_cephes_exp_p1);
    y = _mm_comp_fmadd_ps(y, x, *(v4sf*)_ps_cephes_exp_p2);
    y = _mm_comp_fmadd_ps(y, x, *(v4sf*)_ps_cephes_exp_p3);
    y = _mm_comp_fmadd_ps(y, x, *(v4sf*)_ps_cephes_exp_p4);
    y = _mm_comp_fmadd_ps(y, x, *(v4sf*)_ps_cephes_exp_p5);
    y = _mm_comp_fmadd_ps(y, tmp, x);

    y = _mm_add_ps(y, one);

#ifndef USE_SSE2
    z = _mm_movehl_ps(z, fx);
    mm0 = _mm_cvttps_pi32(fx);
    mm1 = _mm_cvttps_pi32(z);
    mm0 = _mm_add_pi32(mm0, *(v2si*)_pi32_0x7f);
    mm1 = _mm_add_pi32(mm1, *(v2si*)_pi32_0x7f);
    mm0 = _mm_slli_pi32(mm0, 23);
    mm1 = _mm_slli_pi32(mm1, 23);

    v4sf pow2n;
    COPY_MM_TO_XMM(mm0, mm1, pow2n);
    _mm_empty();
#else
    emm0 = _mm_cvttps_epi32(fx);
    emm0 = _mm_add_epi32(emm0, *(v4si*)_pi32_0x7f);
    emm0 = _mm_slli_epi32(emm0, 23);
    v4sf pow2n = _mm_castsi128_ps(emm0);
#endif
    y = _mm_mul_ps(y, pow2n);
    return y;
}

_PS_CONST(tanh_hi, 9.0f);
_PS_CONST(tanh_lo, -9.0f);

_PS_CONST(cephes_tanh_p0, -2.76076847742355E-16f);
_PS_CONST(cephes_tanh_p1, 2.00018790482477E-13f);
_PS_CONST(cephes_tanh_p2, -8.60467152213735E-11f);
_PS_CONST(cephes_tanh_p3, 5.12229709037114E-08f);
_PS_CONST(cephes_tanh_p4, 1.48572235717979E-05f);
_PS_CONST(cephes_tanh_p5, 6.37261928875436E-04f);
_PS_CONST(cephes_tanh_p6, 4.89352455891786E-03f);
_PS_CONST(cephes_tanh_p7, 1.19825839466702e-06f);
_PS_CONST(cephes_tanh_p8, 1.18534705686654e-04f);
_PS_CONST(cephes_tanh_p9, 2.26843463243900e-03f);

static inline v4sf tanh_ps(const v4sf x)
{
    v4sf value = x;
    value = _mm_max_ps(*(v4sf*)_ps_tanh_lo, value);
    value = _mm_min_ps(*(v4sf*)_ps_tanh_hi, value);

    v4sf value_squared = _mm_mul_ps(value, value);

    v4sf p;
    p = _mm_comp_fmadd_ps(value_squared, *(v4sf*)_ps_cephes_tanh_p0, *(v4sf*)_ps_cephes_tanh_p1);
    p = _mm_comp_fmadd_ps(p, value_squared, *(v4sf*)_ps_cephes_tanh_p2);
    p = _mm_comp_fmadd_ps(p, value_squared, *(v4sf*)_ps_cephes_tanh_p3);
    p = _mm_comp_fmadd_ps(p, value_squared, *(v4sf*)_ps_cephes_tanh_p4);
    p = _mm_comp_fmadd_ps(p, value_squared, *(v4sf*)_ps_cephes_tanh_p5);
    p = _mm_comp_fmadd_ps(p, value_squared, *(v4sf*)_ps_cephes_tanh_p6);
    p = _mm_mul_ps(p, value);

    v4sf q;
    q = _mm_comp_fmadd_ps(value_squared, *(v4sf*)_ps_cephes_tanh_p7, *(v4sf*)_ps_cephes_tanh_p8);
    q = _mm_comp_fmadd_ps(q, value_squared, *(v4sf*)_ps_cephes_tanh_p9);
    q = _mm_comp_fmadd_ps(q, value_squared, *(v4sf*)_ps_cephes_tanh_p6);

    v4sf dst = _mm_div_ps(p, q);
    return dst;
}

_PS_CONST(minus_cephes_DP1, -0.78515625f);
_PS_CONST(minus_cephes_DP2, -2.4187564849853515625e-4f);
_PS_CONST(minus_cephes_DP3, -3.77489497744594108e-8f);
_PS_CONST(sincof_p0, -1.9515295891E-4f);
_PS_CONST(sincof_p1, 8.3321608736E-3f);
_PS_CONST(sincof_p2, -1.6666654611E-1f);
_PS_CONST(coscof_p0, 2.443315711809948E-005f);
_PS_CONST(coscof_p1, -1.388731625493765E-003f);
_PS_CONST(coscof_p2, 4.166664568298827E-002f);
_PS_CONST(cephes_FOPI, 1.27323954473516f);

#endif
