#include "rvv_mathfun.h"

void sigmoid(float* data, int channels, int size)
{
    for (int q = 0; q < channels; q++)
    {
        float* ptr = data + q * size;

        int n = size;
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e32m8(n);

            vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
            _p = sigmoid_ps(_p, vl);
            __riscv_vse32_v_f32m8(ptr, _p, vl);

            ptr += vl;
            n -= vl;
        }
    }
}
