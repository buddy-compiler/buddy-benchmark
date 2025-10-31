#include "rvv_mathfun.h"

void selu(float* data, int channels, int size, float alpha, float lambda)
{
    float alphaxlambda = alpha * lambda;
    for (int q = 0; q < channels; q++)
    {
        float* ptr = data + q * size;
        int n = size;
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e32m8(n);
            vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
            vbool4_t _lower = __riscv_vmflt_vf_f32m8_b4(_p, 0.f, vl);
            vbool4_t _higher = __riscv_vmnot_m_b4(_lower, vl);

            _p = __riscv_vfmul_vf_f32m8_mu(_higher, _p, _p, lambda, vl);
            vfloat32m8_t _nps = exp_ps(_p, vl);
            _nps = __riscv_vfsub_vf_f32m8_mu(_lower, _p, _nps, 1.f, vl);
            _nps = __riscv_vfmul_vf_f32m8_mu(_lower, _p, _nps, alphaxlambda, vl);

            __riscv_vse32_v_f32m8(ptr, _nps, vl);
            ptr += vl;
            n -= vl;
        }
    }
}