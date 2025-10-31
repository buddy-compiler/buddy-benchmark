#include "rvv_mathfun.h"

void gelu(float* data, int channels, int size, bool fast_gelu)
{
    if (fast_gelu)
    {
        for (int q = 0; q < channels; q++)
        {
            float* ptr = data + q * size;
            int n = size;
            while (n > 0)
            {
                size_t vl = __riscv_vsetvl_e32m4(n);

                vfloat32m4_t _p = __riscv_vle32_v_f32m4(ptr, vl);

                vfloat32m4_t _arg = __riscv_vfmul_vf_f32m4(
                                        __riscv_vfmul_vv_f32m4(__riscv_vfmul_vv_f32m4(_p, _p, vl), _p, vl), 0.044715f, vl);

                _arg = __riscv_vfadd_vv_f32m4(_p, _arg, vl);
                _arg = __riscv_vfmul_vf_f32m4(_arg, 0.79788452f, vl);
                vfloat32m4_t _tanharg = tanh_ps(_arg, vl);
                _p = __riscv_vfmul_vf_f32m4(
                         __riscv_vfmul_vv_f32m4(_p, __riscv_vfadd_vf_f32m4(_tanharg, 1.f, vl), vl), .5f, vl);

                __riscv_vse32_v_f32m4(ptr, _p, vl);
                n -= vl;
                ptr += vl;
            }
        }
    }
    else
    {
        for (int q = 0; q < channels; q++)
        {
            float* ptr = data + q * size;

            int n = size;
            while (n > 0)
            {
                size_t vl = __riscv_vsetvl_e32m8(n);
                auto _p = __riscv_vle32_v_f32m8(ptr, vl);
                auto _perfc = __riscv_vfmul_vf_f32m8(_p, -.70710678f, vl);
                _p = __riscv_vfmul_vf_f32m8(_p, .5f, vl);

                _perfc = erfc_ps(_perfc, vl);

                _p = __riscv_vfmul_vv_f32m8(_p, _perfc, vl);
                __riscv_vse32_v_f32m8(ptr, _p, vl);

                n -= vl;
                ptr += vl;
            }
        }
    }
}