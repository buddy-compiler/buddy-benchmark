#include <riscv_vector.h>

void hardswish(float* data, int channels, int size, float alpha, float beta)
{
    float lower = -beta / alpha;
    float upper = (1.f / alpha) + lower;
    for (int q = 0; q < channels; q++)
    {
        float* ptr = data + q * size;
        int n = size;
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e32m8(n);
            vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);

            vbool4_t _lower = __riscv_vmflt_vf_f32m8_b4(_p, lower, vl);
            vbool4_t _higher = __riscv_vmfgt_vf_f32m8_b4(_p, upper, vl);
            vbool4_t _apply = __riscv_vmnor_mm_b4(_lower, _higher, vl);
            _p = __riscv_vfmerge_vfm_f32m8(_p, .0f, _lower, vl);

            vfloat32m8_t _p0 = __riscv_vfadd_vf_f32m8_m(_apply, __riscv_vfmul_vf_f32m8_m(_apply, _p, alpha, vl), beta, vl);
            _p = __riscv_vfmul_vv_f32m8_mu(_apply, _p, _p, _p0, vl);

            __riscv_vse32_v_f32m8(ptr, _p, vl);
            ptr += vl;
            n -= vl;
        }
    }
}