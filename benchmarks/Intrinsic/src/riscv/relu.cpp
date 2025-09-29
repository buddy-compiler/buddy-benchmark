#include <riscv_vector.h>

void relu(float* data, int channels, int size, float slope = 0.f)
{
    for (int q = 0; q < channels; q++)
    {
        float* ptr = data + q * size;
        if (slope == 0.f) {
            int n = size;
            while (n > 0) {
                size_t vl = __riscv_vsetvl_e32m8(n);
                vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
                _p = __riscv_vfmax_vf_f32m8(_p, 0.f, vl);
                __riscv_vse32_v_f32m8(ptr, _p, vl);

                ptr += vl;
                n -= vl;
            }
        }
        else
        {
            int n = size;
            while (n > 0)
            {
                size_t vl = __riscv_vsetvl_e32m8(n);
                vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
                _p = __riscv_vfmul_vf_f32m8_mu(__riscv_vmflt_vf_f32m8_b4(_p, .0f, vl), _p, _p, slope, vl);
                __riscv_vse32_v_f32m8(ptr, _p, vl);
                ptr += vl;
                n -= vl;
            }
        }
    }
}