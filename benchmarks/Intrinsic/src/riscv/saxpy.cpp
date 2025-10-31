#include <riscv_vector.h>

void SAXPY(int n, const float *x, float *y, const float a) {
  for (size_t vl; n > 0; n -= vl, x += vl, y += vl) {
    vl = __riscv_vsetvl_e32m1(n);
    vfloat32m1_t vx = __riscv_vle32_v_f32m1(x, vl);
    vfloat32m1_t vy = __riscv_vle32_v_f32m1(y, vl);
    __riscv_vse32_v_f32m1(y, __riscv_vfmacc_vf_f32m1(vy, a, vx, vl), vl);
  }
}
