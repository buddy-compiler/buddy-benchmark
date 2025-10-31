#include <riscv_vector.h>

#define c_tanh_tiny 1e-4f
#define c_tanh_hi   9.0f

#define c_tanh_alpha_1  4.89352455891786e-3f
#define c_tanh_alpha_3  6.37261928875436e-4f
#define c_tanh_alpha_5  1.48572235717979e-5f
#define c_tanh_alpha_7  5.12229709037114e-8f
#define c_tanh_alpha_9  -8.60467152213735e-11f
#define c_tanh_alpha_11 2.00018790482477e-13f
#define c_tanh_alpha_13 -2.76076847742355e-16f

#define c_tanh_beta_0 4.89352518554385e-3f
#define c_tanh_beta_2 2.26843463243900e-3f
#define c_tanh_beta_4 1.18534705686654e-4f
#define c_tanh_beta_6 1.19825839466702e-6f

#define _RVV_FLOAT32_TANH_OP(LMUL, MLEN)                                                         \
    static inline vfloat32m##LMUL##_t tanh_ps(vfloat32m##LMUL##_t x, size_t vl)                  \
    {                                                                                            \
        vfloat32m##LMUL##_t x2 = __riscv_vfsgnj_vf_f32m##LMUL(x, 1.f, vl);                       \
                                                                                                 \
        vbool##MLEN##_t tiny_mask = __riscv_vmfge_vf_f32m##LMUL##_b##MLEN(x2, c_tanh_tiny, vl);  \
                                                                                                 \
                       \
                       \
        x2 = __riscv_vfmin_vf_f32m##LMUL(x2, c_tanh_hi, vl);                                     \
                                                                                                 \
                                  \
        vfloat32m##LMUL##_t z = __riscv_vfmul_vv_f32m##LMUL(x2, x2, vl);                         \
                                                                                                 \
                                               \
        vfloat32m##LMUL##_t y = __riscv_vfmul_vf_f32m##LMUL(z, c_tanh_alpha_13, vl);             \
        y = __riscv_vfadd_vf_f32m##LMUL(y, c_tanh_alpha_11, vl);                                 \
        y = __riscv_vfmul_vv_f32m##LMUL(y, z, vl);                                               \
        y = __riscv_vfadd_vf_f32m##LMUL(y, c_tanh_alpha_9, vl);                                  \
        y = __riscv_vfmul_vv_f32m##LMUL(y, z, vl);                                               \
        y = __riscv_vfadd_vf_f32m##LMUL(y, c_tanh_alpha_7, vl);                                  \
        y = __riscv_vfmul_vv_f32m##LMUL(y, z, vl);                                               \
        y = __riscv_vfadd_vf_f32m##LMUL(y, c_tanh_alpha_5, vl);                                  \
        y = __riscv_vfmul_vv_f32m##LMUL(y, z, vl);                                               \
        y = __riscv_vfadd_vf_f32m##LMUL(y, c_tanh_alpha_3, vl);                                  \
        y = __riscv_vfmul_vv_f32m##LMUL(y, z, vl);                                               \
        y = __riscv_vfadd_vf_f32m##LMUL(y, c_tanh_alpha_1, vl);                                  \
        y = __riscv_vfmul_vv_f32m##LMUL(y, x2, vl);                                              \
                                                                                                 \
                                             \
        vfloat32m##LMUL##_t w = __riscv_vfmul_vf_f32m##LMUL(z, c_tanh_beta_6, vl);               \
        w = __riscv_vfadd_vf_f32m##LMUL(w, c_tanh_beta_4, vl);                                   \
        w = __riscv_vfmul_vv_f32m##LMUL(w, z, vl);                                               \
        w = __riscv_vfadd_vf_f32m##LMUL(w, c_tanh_beta_2, vl);                                   \
        w = __riscv_vfmul_vv_f32m##LMUL(w, z, vl);                                               \
        w = __riscv_vfadd_vf_f32m##LMUL(w, c_tanh_beta_0, vl);                                   \
                                                                                                 \
                                           \
        y = __riscv_vfdiv_vv_f32m##LMUL(y, w, vl);                                               \
                                                                                                 \
                                                               \
        y = __riscv_vfsgnj_vv_f32m##LMUL(y, x, vl);                                              \
                                                                                                 \
 \
        y = __riscv_vmerge_vvm_f32m##LMUL(x, y, tiny_mask, vl);                                  \
                                                                                                 \
        return y;                                                                                \
    }

_RVV_FLOAT32_TANH_OP(1, 32)
_RVV_FLOAT32_TANH_OP(2, 16)
_RVV_FLOAT32_TANH_OP(4, 8)
_RVV_FLOAT32_TANH_OP(8, 4)

#define c_exp_hi 88.3762626647949f
#define c_exp_lo -88.3762626647949f

#define c_cephes_LOG2EF 1.44269504088896341
#define c_cephes_exp_C1 0.693359375
#define c_cephes_exp_C2 -2.12194440e-4

#define c_cephes_exp_p0 1.9875691500E-4
#define c_cephes_exp_p1 1.3981999507E-3
#define c_cephes_exp_p2 8.3334519073E-3
#define c_cephes_exp_p3 4.1665795894E-2
#define c_cephes_exp_p4 1.6666665459E-1
#define c_cephes_exp_p5 5.0000001201E-1

#define _RVV_FLOAT32_EXP_OP(LMUL, MLEN)                                                                   \
    static inline vfloat32m##LMUL##_t exp_ps(vfloat32m##LMUL##_t x, size_t vl)                            \
    {                                                                                                     \
        vfloat32m##LMUL##_t tmp, fx;                                                                      \
                                                                                                          \
        x = __riscv_vfmin_vf_f32m##LMUL(x, c_exp_hi, vl);                                                 \
        x = __riscv_vfmax_vf_f32m##LMUL(x, c_exp_lo, vl);                                                 \
                                                                                                          \
                                                         \
        fx = __riscv_vfmacc_vf_f32m##LMUL(__riscv_vfmv_v_f_f32m##LMUL(0.5f, vl), c_cephes_LOG2EF, x, vl); \
                                                                                                          \
                                                                            \
        tmp = __riscv_vfcvt_f_x_v_f32m##LMUL(__riscv_vfcvt_x_f_v_i32m##LMUL(fx, vl), vl);                 \
                                                                                                          \
                                                                     \
        vbool##MLEN##_t mask = __riscv_vmfgt_vv_f32m##LMUL##_b##MLEN(tmp, fx, vl);                        \
        fx = __riscv_vfsub_vf_f32m##LMUL##_mu(mask, tmp, tmp, 1.f, vl);                                   \
                                                                                                          \
        tmp = __riscv_vfmul_vf_f32m##LMUL(fx, c_cephes_exp_C1, vl);                                       \
        vfloat32m##LMUL##_t z = __riscv_vfmul_vf_f32m##LMUL(fx, c_cephes_exp_C2, vl);                     \
        x = __riscv_vfsub_vv_f32m##LMUL(x, tmp, vl);                                                      \
        x = __riscv_vfsub_vv_f32m##LMUL(x, z, vl);                                                        \
                                                                                                          \
        vfloat32m##LMUL##_t y = __riscv_vfmul_vf_f32m##LMUL(x, c_cephes_exp_p0, vl);                      \
        z = __riscv_vfmul_vv_f32m##LMUL(x, x, vl);                                                        \
                                                                                                          \
        y = __riscv_vfadd_vf_f32m##LMUL(y, c_cephes_exp_p1, vl);                                          \
        y = __riscv_vfmul_vv_f32m##LMUL(y, x, vl);                                                        \
        y = __riscv_vfadd_vf_f32m##LMUL(y, c_cephes_exp_p2, vl);                                          \
        y = __riscv_vfmul_vv_f32m##LMUL(y, x, vl);                                                        \
        y = __riscv_vfadd_vf_f32m##LMUL(y, c_cephes_exp_p3, vl);                                          \
        y = __riscv_vfmul_vv_f32m##LMUL(y, x, vl);                                                        \
        y = __riscv_vfadd_vf_f32m##LMUL(y, c_cephes_exp_p4, vl);                                          \
        y = __riscv_vfmul_vv_f32m##LMUL(y, x, vl);                                                        \
        y = __riscv_vfadd_vf_f32m##LMUL(y, c_cephes_exp_p5, vl);                                          \
                                                                                                          \
        y = __riscv_vfmul_vv_f32m##LMUL(y, z, vl);                                                        \
        y = __riscv_vfadd_vv_f32m##LMUL(y, x, vl);                                                        \
        y = __riscv_vfadd_vf_f32m##LMUL(y, 1.f, vl);                                                      \
                                                                                                          \
                                                                                   \
        vint32m##LMUL##_t mm = __riscv_vfcvt_x_f_v_i32m##LMUL(fx, vl);                                    \
        mm = __riscv_vadd_vx_i32m##LMUL(mm, 0x7f, vl);                                                    \
        mm = __riscv_vsll_vx_i32m##LMUL(mm, 23, vl);                                                      \
        vfloat32m##LMUL##_t pow2n = __riscv_vreinterpret_v_i32m##LMUL##_f32m##LMUL(mm);                   \
                                                                                                          \
        y = __riscv_vfmul_vv_f32m##LMUL(y, pow2n, vl);                                                    \
        return y;                                                                                         \
    }

_RVV_FLOAT32_EXP_OP(1, 32)
_RVV_FLOAT32_EXP_OP(2, 16)
_RVV_FLOAT32_EXP_OP(4, 8)
_RVV_FLOAT32_EXP_OP(8, 4)

#define c_erfc_erx_f 8.4506291151e-01f

#define c_erfc_efx  1.2837916613e-01f
#define c_erfc_efx8 1.0270333290e+00f

#define c_erfc_pp0 1.2837916613e-01f
#define c_erfc_pp1 -3.2504209876e-01f
#define c_erfc_pp2 -2.8481749818e-02f
#define c_erfc_pp3 -5.7702702470e-03f
#define c_erfc_pp4 -2.3763017452e-05f
#define c_erfc_qq1 3.9791721106e-01f
#define c_erfc_qq2 6.5022252500e-02f
#define c_erfc_qq3 5.0813062117e-03f
#define c_erfc_qq4 1.3249473704e-04f
#define c_erfc_qq5 -3.9602282413e-06f

#define c_erfc_pa0 -2.3621185683e-03f
#define c_erfc_pa1 4.1485610604e-01f
#define c_erfc_pa2 -3.7220788002e-01f
#define c_erfc_pa3 3.1834661961e-01f
#define c_erfc_pa4 -1.1089469492e-01f
#define c_erfc_pa5 3.5478305072e-02f
#define c_erfc_pa6 -2.1663755178e-03f
#define c_erfc_qa1 1.0642088205e-01f
#define c_erfc_qa2 5.4039794207e-01f
#define c_erfc_qa3 7.1828655899e-02f
#define c_erfc_qa4 1.2617121637e-01f
#define c_erfc_qa5 1.3637083583e-02f
#define c_erfc_qa6 1.1984500103e-02f

#define c_erfc_ra0 -9.8649440333e-03f
#define c_erfc_ra1 -6.9385856390e-01f
#define c_erfc_ra2 -1.0558626175e+01f
#define c_erfc_ra3 -6.2375331879e+01f
#define c_erfc_ra4 -1.6239666748e+02f
#define c_erfc_ra5 -1.8460508728e+02f
#define c_erfc_ra6 -8.1287437439e+01f
#define c_erfc_ra7 -9.8143291473e+00f
#define c_erfc_sa1 1.9651271820e+01f
#define c_erfc_sa2 1.3765776062e+02f
#define c_erfc_sa3 4.3456588745e+02f
#define c_erfc_sa4 6.4538726807e+02f
#define c_erfc_sa5 4.2900814819e+02f
#define c_erfc_sa6 1.0863500214e+02f
#define c_erfc_sa7 6.5702495575e+00f
#define c_erfc_sa8 -6.0424413532e-02f

#define c_erfc_rb0 -9.8649431020e-03f
#define c_erfc_rb1 -7.9928326607e-01f
#define c_erfc_rb2 -1.7757955551e+01f
#define c_erfc_rb3 -1.6063638306e+02f
#define c_erfc_rb4 -6.3756646729e+02f
#define c_erfc_rb5 -1.0250950928e+03f
#define c_erfc_rb6 -4.8351919556e+02f
#define c_erfc_sb1 3.0338060379e+01f
#define c_erfc_sb2 3.2579251099e+02f
#define c_erfc_sb3 1.5367296143e+03f
#define c_erfc_sb4 3.1998581543e+03f
#define c_erfc_sb5 2.5530502930e+03f
#define c_erfc_sb6 4.7452853394e+02f
#define c_erfc_sb7 -2.2440952301e+01f

#define _RVV_FLOAT32_FMA_HELPER(LMUL)                                                                             \
    static inline vfloat32m##LMUL##_t __riscv_vfmadd_vff_f32m##LMUL(vfloat32m##LMUL##_t a, float b,               \
                                                                    float c, size_t vl)                           \
    {                                                                                                             \
        vfloat32m##LMUL##_t ret = __riscv_vfmul_vf_f32m##LMUL(a, b, vl);                                          \
        ret = __riscv_vfadd_vf_f32m##LMUL(ret, c, vl);                                                            \
        return ret;                                                                                               \
    }                                                                                                             \
                                                                                                                  \
    static inline vfloat32m##LMUL##_t __riscv_vfmadd_vvf_f32m##LMUL(vfloat32m##LMUL##_t a, vfloat32m##LMUL##_t b, \
                                                                    float c, size_t vl)                           \
    {                                                                                                             \
        vfloat32m##LMUL##_t ret = __riscv_vfmul_vv_f32m##LMUL(a, b, vl);                                          \
        ret = __riscv_vfadd_vf_f32m##LMUL(ret, c, vl);                                                            \
        return ret;                                                                                               \
    }

_RVV_FLOAT32_FMA_HELPER(8)
_RVV_FLOAT32_FMA_HELPER(4)
_RVV_FLOAT32_FMA_HELPER(2)
_RVV_FLOAT32_FMA_HELPER(1)

#define _RVV_FLOAT32_ERFC_OP(LMUL, MLEN)                                                                                                                                                                                                                                                                                                                                                                   \
    static inline vfloat32m##LMUL##_t erfc_ps(vfloat32m##LMUL##_t x, size_t vl)                                                                                                                                                                                                                                                                                                                            \
    {                                                                                                                                                                                                                                                                                                                                                                                                      \
                                                                                                                                                                                                                                                                                                                                                                           \
        vfloat32m##LMUL##_t absx = __riscv_vfsgnjx_vv_f32m##LMUL(x, x, vl);                                                                                                                                                                                                                                                                                                                                \
        vfloat32m##LMUL##_t x2 = __riscv_vfmul_vv_f32m##LMUL(x, x, vl);                                                                                                                                                                                                                                                                                                                                    \
        vfloat32m##LMUL##_t t = __riscv_vfrdiv_vf_f32m##LMUL(x2, 1.0f, vl);                                                                                                                                                                                                                                                                                                                                \
        vfloat32m##LMUL##_t tt = __riscv_vfsub_vf_f32m##LMUL(absx, 1.0f, vl);                                                                                                                                                                                                                                                                                                                              \
                                                                                                                                                                                                                                                                                                                                                                          \
        t = __riscv_vmerge_vvm_f32m##LMUL(tt, t, __riscv_vmfge_vf_f32m##LMUL##_b##MLEN(absx, 1.25f, vl), vl);                                                                                                                                                                                                                                                                                              \
                                                                                                                                                                                                                                                                                                                                                                        \
        t = __riscv_vmerge_vvm_f32m##LMUL(x2, t, __riscv_vmfge_vf_f32m##LMUL##_b##MLEN(absx, 0.84375f, vl), vl);                                                                                                                                                                                                                                                                                           \
                                                                                                                                                                                                                                                                                                                                                                                                           \
        vfloat32m##LMUL##_t u = __riscv_vfmadd_vvf_f32m##LMUL(t, __riscv_vfmadd_vvf_f32m##LMUL(t, __riscv_vfmadd_vvf_f32m##LMUL(t, __riscv_vfmadd_vvf_f32m##LMUL(t, __riscv_vfmadd_vvf_f32m##LMUL(t, __riscv_vfmadd_vff_f32m##LMUL(t, c_erfc_rb6, c_erfc_rb5, vl), c_erfc_rb4, vl), c_erfc_rb3, vl), c_erfc_rb2, vl), c_erfc_rb1, vl), c_erfc_rb0, vl);                                                    \
        vfloat32m##LMUL##_t v = __riscv_vfmadd_vvf_f32m##LMUL(t, __riscv_vfmadd_vvf_f32m##LMUL(t, __riscv_vfmadd_vvf_f32m##LMUL(t, __riscv_vfmadd_vvf_f32m##LMUL(t, __riscv_vfmadd_vvf_f32m##LMUL(t, __riscv_vfmadd_vff_f32m##LMUL(t, c_erfc_sb7, c_erfc_sb6, vl), c_erfc_sb5, vl), c_erfc_sb4, vl), c_erfc_sb3, vl), c_erfc_sb2, vl), c_erfc_sb1, vl);                                                    \
                                                                                                                                                                                                                                                                                                                                                                                                           \
        vfloat32m##LMUL##_t tu = __riscv_vfmadd_vvf_f32m##LMUL(t, __riscv_vfmadd_vvf_f32m##LMUL(t, __riscv_vfmadd_vvf_f32m##LMUL(t, __riscv_vfmadd_vvf_f32m##LMUL(t, __riscv_vfmadd_vvf_f32m##LMUL(t, __riscv_vfmadd_vvf_f32m##LMUL(t, __riscv_vfmadd_vff_f32m##LMUL(t, c_erfc_ra7, c_erfc_ra6, vl), c_erfc_ra5, vl), c_erfc_ra4, vl), c_erfc_ra3, vl), c_erfc_ra2, vl), c_erfc_ra1, vl), c_erfc_ra0, vl); \
        vfloat32m##LMUL##_t tv = __riscv_vfmadd_vvf_f32m##LMUL(t, __riscv_vfmadd_vvf_f32m##LMUL(t, __riscv_vfmadd_vvf_f32m##LMUL(t, __riscv_vfmadd_vvf_f32m##LMUL(t, __riscv_vfmadd_vvf_f32m##LMUL(t, __riscv_vfmadd_vvf_f32m##LMUL(t, __riscv_vfmadd_vff_f32m##LMUL(t, c_erfc_sa8, c_erfc_sa7, vl), c_erfc_sa6, vl), c_erfc_sa5, vl), c_erfc_sa4, vl), c_erfc_sa3, vl), c_erfc_sa2, vl), c_erfc_sa1, vl); \
        u = __riscv_vmerge_vvm_f32m##LMUL(tu, u, __riscv_vmfge_vf_f32m##LMUL##_b##MLEN(absx, 2.857143f, vl), vl);                                                                                                                                                                                                                                                 \
        v = __riscv_vmerge_vvm_f32m##LMUL(tv, v, __riscv_vmfge_vf_f32m##LMUL##_b##MLEN(absx, 2.857143f, vl), vl);                                                                                                                                                                                                                                                 \
                                                                                                                                                                                                                                                                                                                                                                                                           \
        tu = __riscv_vfmadd_vvf_f32m##LMUL(t, __riscv_vfmadd_vvf_f32m##LMUL(t, __riscv_vfmadd_vvf_f32m##LMUL(t, __riscv_vfmadd_vvf_f32m##LMUL(t, __riscv_vfmadd_vvf_f32m##LMUL(t, __riscv_vfmadd_vff_f32m##LMUL(t, c_erfc_pa6, c_erfc_pa5, vl), c_erfc_pa4, vl), c_erfc_pa3, vl), c_erfc_pa2, vl), c_erfc_pa1, vl), c_erfc_pa0, vl);                                                                       \
        tv = __riscv_vfmadd_vvf_f32m##LMUL(t, __riscv_vfmadd_vvf_f32m##LMUL(t, __riscv_vfmadd_vvf_f32m##LMUL(t, __riscv_vfmadd_vvf_f32m##LMUL(t, __riscv_vfmadd_vff_f32m##LMUL(t, c_erfc_qa6, c_erfc_qa5, vl), c_erfc_qa4, vl), c_erfc_qa3, vl), c_erfc_qa2, vl), c_erfc_qa1, vl);                                                                                                                         \
        u = __riscv_vmerge_vvm_f32m##LMUL(tu, u, __riscv_vmfge_vf_f32m##LMUL##_b##MLEN(absx, 1.25f, vl),                                                                                                                                                                                                                                                                                                   \
                                          vl);                                                                                                                                                                                                                                                                                                                                 \
        v = __riscv_vmerge_vvm_f32m##LMUL(tv, v, __riscv_vmfge_vf_f32m##LMUL##_b##MLEN(absx, 1.25f, vl),                                                                                                                                                                                                                                                                                                   \
                                          vl);                                                                                                                                                                                                                                                                                                                                \
                                                                                                                                                                                                                                                                                                                                                                                                           \
        tu = __riscv_vfmadd_vvf_f32m##LMUL(t, __riscv_vfmadd_vvf_f32m##LMUL(t, __riscv_vfmadd_vvf_f32m##LMUL(t, __riscv_vfmadd_vff_f32m##LMUL(t, c_erfc_pp4, c_erfc_pp3, vl), c_erfc_pp2, vl), c_erfc_pp1, vl), c_erfc_pp0, vl);                                                                                                                                                                           \
        tv = __riscv_vfmadd_vvf_f32m##LMUL(t, __riscv_vfmadd_vvf_f32m##LMUL(t, __riscv_vfmadd_vvf_f32m##LMUL(t, __riscv_vfmadd_vff_f32m##LMUL(t, c_erfc_qq5, c_erfc_qq4, vl), c_erfc_qq3, vl), c_erfc_qq2, vl), c_erfc_qq1, vl);                                                                                                                                                                           \
        u = __riscv_vmerge_vvm_f32m##LMUL(tu, u, __riscv_vmfge_vf_f32m##LMUL##_b##MLEN(absx, 0.84375f, vl), vl);                                                                                                                                                                                                                                                            \
        v = __riscv_vmerge_vvm_f32m##LMUL(tv, v, __riscv_vmfge_vf_f32m##LMUL##_b##MLEN(absx, 0.84375f, vl), vl);                                                                                                                                                                                                                                                           \
                                                                                                                                                                                                                                                                                                                                                                                                           \
        v = __riscv_vfmadd_vvf_f32m##LMUL(t, v, 1.f, vl);                                                                                                                                                                                                                                                                                                                                                  \
                                                                                                                                                                                                                                                                                                                                                                                                           \
        vfloat32m##LMUL##_t q = __riscv_vfdiv_vv_f32m##LMUL(u, v, vl);                                                                                                                                                                                                                                                                                                                                     \
        vfloat32m##LMUL##_t ret = __riscv_vfmv_v_f_f32m##LMUL(0.f, vl);                                                                                                                                                                                                                                                                                                                                    \
                                                                                                                                                                                                                                                                                                                                                                                                           \
        vfloat32m##LMUL##_t z = __riscv_vreinterpret_v_u32m##LMUL##_f32m##LMUL(__riscv_vand_vx_u32m##LMUL(__riscv_vreinterpret_v_f32m##LMUL##_u32m##LMUL(absx), 0xfffff000, vl));                                                                                                                                                                                                                          \
                                                                                                                                                                                                                                                                                                                                                                                                           \
        vfloat32m##LMUL##_t r = __riscv_vfmul_vv_f32m##LMUL(exp_ps(__riscv_vfmadd_vvf_f32m##LMUL(__riscv_vfneg_v_f32m##LMUL(z, vl), z, -0.5625f, vl), vl), exp_ps(__riscv_vfmadd_vv_f32m##LMUL(__riscv_vfsub_vv_f32m##LMUL(z, absx, vl), __riscv_vfadd_vv_f32m##LMUL(z, absx, vl), q, vl), vl), vl);                                                                                                       \
        r = __riscv_vfdiv_vv_f32m##LMUL(r, absx, vl);                                                                                                                                                                                                                                                                                                                                                      \
        t = __riscv_vfrsub_vf_f32m##LMUL(r, 2.f, vl);                                                                                                                                                                                                                                                                                                                                                      \
        r = __riscv_vmerge_vvm_f32m##LMUL(t, r, __riscv_vmfge_vf_f32m##LMUL##_b##MLEN(x, 0.f, vl), vl);                                                                                                                                                                                                                                                                       \
        ret = __riscv_vmerge_vvm_f32m##LMUL(r, ret, __riscv_vmfge_vf_f32m##LMUL##_b##MLEN(absx, 28.f, vl), vl);                                                                                                                                                                                                                                                               \
                                                                                                                                                                                                                                                                                                                                                                                                           \
        r = __riscv_vfrsub_vf_f32m##LMUL(q, 1.f - c_erfc_erx_f, vl);                                                                                                                                                                                                                                                                                                                                       \
        t = __riscv_vfadd_vf_f32m##LMUL(q, 1.f + c_erfc_erx_f, vl);                                                                                                                                                                                                                                                                                                                                        \
        r = __riscv_vmerge_vvm_f32m##LMUL(t, r, __riscv_vmfge_vf_f32m##LMUL##_b##MLEN(x, 0.f, vl), vl);                                                                                                                                                                                                                                                                       \
        ret = __riscv_vmerge_vvm_f32m##LMUL(r, ret, __riscv_vmfge_vf_f32m##LMUL##_b##MLEN(absx, 1.25f, vl), vl);                                                                                                                                                                                                                                                              \
                                                                                                                                                                                                                                                                                                                                                                                                           \
        r = __riscv_vfrsub_vf_f32m##LMUL(__riscv_vfmadd_vv_f32m##LMUL(x, q, __riscv_vfsub_vf_f32m##LMUL(x, 0.5f, vl), vl), .5, vl);                                                                                                                                                                                                                                                                        \
        ret = __riscv_vmerge_vvm_f32m##LMUL(r, ret, __riscv_vmfge_vf_f32m##LMUL##_b##MLEN(absx, 0.84375f, vl), vl);                                                                                                                                                                                                                                                        \
                                                                                                                                                                                                                                                                                                                                                                                                           \
        ret = __riscv_vfmerge_vfm_f32m##LMUL(ret, 2.f, __riscv_vmflt_vf_f32m##LMUL##_b##MLEN(x, -6.0f, vl), vl);                                                                                                                                                                                                                                                                \
                                                                                                                                                                                                                                                                                                                                                                                                           \
        ret = __riscv_vmerge_vvm_f32m##LMUL(x, ret, __riscv_vmfeq_vv_f32m##LMUL##_b##MLEN(x, x, vl), vl);                                                                                                                                                                                                                                                                            \
                                                                                                                                                                                                                                                                                                                                                                                                           \
        return ret;                                                                                                                                                                                                                                                                                                                                                                                        \
    }

_RVV_FLOAT32_ERFC_OP(1, 32)
_RVV_FLOAT32_ERFC_OP(2, 16)
_RVV_FLOAT32_ERFC_OP(4, 8)
_RVV_FLOAT32_ERFC_OP(8, 4)

#define _RVV_FLOAT32_SIGMOID_OP(LMUL, MLEN)                                                                                                                        \
    static inline vfloat32m##LMUL##_t sigmoid_ps(vfloat32m##LMUL##_t _v, size_t vl)                                                                                \
    {                                                                                                                                                              \
        _v = __riscv_vfneg_v_f32m##LMUL(_v, vl);                                                                                                                   \
        _v = exp_ps(_v, vl);                                                                                                                                       \
        _v = __riscv_vfadd_vf_f32m##LMUL(_v, 1.f, vl);                                                                                                             \
        vfloat32m##LMUL##_t _reciprocal = __riscv_vfrec7_v_f32m##LMUL(_v, vl);                                                                                     \
        _reciprocal = __riscv_vfmul_vv_f32m##LMUL(__riscv_vfrsub_vf_f32m##LMUL(__riscv_vfmul_vv_f32m##LMUL(_v, _reciprocal, vl), 2.f, vl), _reciprocal, vl);       \
 \
        return _reciprocal;                                                                                                                                        \
    }

_RVV_FLOAT32_SIGMOID_OP(1, 32)
_RVV_FLOAT32_SIGMOID_OP(2, 16)
_RVV_FLOAT32_SIGMOID_OP(4, 8)
_RVV_FLOAT32_SIGMOID_OP(8, 4)
