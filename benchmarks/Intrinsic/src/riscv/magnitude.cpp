#include <riscv_vector.h>

void magnitude32f(const float *x, const float *y, float *dst, int len) {
    size_t vl;
    for (; len > 0; len -= (int)vl, x += vl, y += vl, dst += vl)
    {
        vl = __riscv_vsetvl_e32m8(len);

        auto vx = __riscv_vle32_v_f32m8(x, vl);
        auto vy = __riscv_vle32_v_f32m8(y, vl);

        auto fqrtX = __riscv_vfmadd(vx, vx, __riscv_vfmul(vy, vy, vl), vl);
        auto fqrtX2 = __riscv_vfmul(fqrtX, 0.5, vl);
        auto fqrtY = __riscv_vfrsqrt7(fqrtX, vl);
        for (size_t i = 0; i < 2; i++)
        {
            auto t = __riscv_vfmul(fqrtY, fqrtY, vl);
            t = __riscv_vfmul(t, fqrtX2, vl);
            t = __riscv_vfrsub(t, 1.5, vl);
            fqrtY = __riscv_vfmul(t, fqrtY, vl);
        }
        asm volatile("" ::: "memory");
        auto classified = __riscv_vfclass(fqrtX, vl);
        auto mask = __riscv_vmseq(__riscv_vand(classified, 0b10111000, vl), 0, vl);
        auto vmag = __riscv_vfmul_mu(mask, fqrtX, fqrtX, fqrtY, vl);

        __riscv_vse32_v_f32m8(dst, vmag, vl);
    }
}
