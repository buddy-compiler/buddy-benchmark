#include <riscv_vector.h>

void merge8u(const unsigned char** src, unsigned char* dst, int len, int cn ) {
    int vl = 0;
    if (cn == 1)
    {
        const unsigned char* src0 = src[0];
        for (int i = 0; i < len; i += vl)
        {
            vl = __riscv_vsetvl_e8m8(len - i);
            __riscv_vse8_v_u8m8(dst + i, __riscv_vle8_v_u8m8(src0 + i, vl), vl);
        }
    }
    else if (cn == 2)
    {
        const unsigned char *src0 = src[0], *src1 = src[1];
        for (int i = 0; i < len; i += vl)
        {
            vl = __riscv_vsetvl_e8m4(len - i);
            vuint8m4x2_t seg = __riscv_vcreate_v_u8m4x2(
                __riscv_vle8_v_u8m4(src0 + i, vl),
                __riscv_vle8_v_u8m4(src1 + i, vl)
            );
            __riscv_vsseg2e8_v_u8m4x2(dst + i * cn, seg, vl);
        }
    }
    else if (cn == 3)
    {
        const unsigned char *src0 = src[0], *src1 = src[1], *src2 = src[2];
        for (int i = 0; i < len; i += vl)
        {
            vl = __riscv_vsetvl_e8m2(len - i);
            vuint8m2x3_t seg = __riscv_vcreate_v_u8m2x3(
                __riscv_vle8_v_u8m2(src0 + i, vl),
                __riscv_vle8_v_u8m2(src1 + i, vl),
                __riscv_vle8_v_u8m2(src2 + i, vl)
            );
            __riscv_vsseg3e8_v_u8m2x3(dst + i * cn, seg, vl);
        }
    }
    else if (cn == 4)
    {
        const unsigned char *src0 = src[0], *src1 = src[1], *src2 = src[2], *src3 = src[3];
        for (int i = 0; i < len; i += vl)
        {
            vl = __riscv_vsetvl_e8m2(len - i);
            vuint8m2x4_t seg = __riscv_vcreate_v_u8m2x4(
                __riscv_vle8_v_u8m2(src0 + i, vl),
                __riscv_vle8_v_u8m2(src1 + i, vl),
                __riscv_vle8_v_u8m2(src2 + i, vl),
                __riscv_vle8_v_u8m2(src3 + i, vl)
            );
            __riscv_vsseg4e8_v_u8m2x4(dst + i * cn, seg, vl);
        }
    }
    else
    {
        int k = 0;
        for (; k <= cn - 4; k += 4)
        {
            const unsigned char *src0 = src[k], *src1 = src[k + 1], *src2 = src[k + 2], *src3 = src[k + 3];
            for (int i = 0; i < len; i += vl)
            {
                vl = __riscv_vsetvl_e8m2(len - i);
                vuint8m2x4_t seg = __riscv_vcreate_v_u8m2x4(
                    __riscv_vle8_v_u8m2(src0 + i, vl),
                    __riscv_vle8_v_u8m2(src1 + i, vl),
                    __riscv_vle8_v_u8m2(src2 + i, vl),
                    __riscv_vle8_v_u8m2(src3 + i, vl)
                );
                __riscv_vssseg4e8_v_u8m2x4(dst + k + i * cn, cn, seg, vl);
            }
        }
        for (; k < cn; ++k)
        {
            const unsigned char* srcK = src[k];
            for (int i = 0; i < len; i += vl)
            {
                vl = __riscv_vsetvl_e8m2(len - i);
                vuint8m2_t seg = __riscv_vle8_v_u8m2(srcK + i, vl);
                __riscv_vsse8_v_u8m2(dst + k + i * cn, cn, seg, vl);
            }
        }
    }
}
