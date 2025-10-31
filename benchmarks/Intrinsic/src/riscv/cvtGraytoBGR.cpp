#include <riscv_vector.h>
#include <cstdint>
#include <algorithm>
int cvtGraytoBGR(const unsigned char* src_data, size_t src_step,
                 unsigned char* dst_data, size_t dst_step,
                 int width, int height,
                 int depth, int scn)
{
    if (depth != 0)
        return 1;
    if (scn != 3 && scn != 4)
        return 1;

    for (size_t i = 0u; i < (size_t)height; ++i)
    {
        const unsigned char * src = src_data + i * src_step;
        unsigned char * dst = dst_data + i * dst_step;

        int vl = 0;

        if (scn == 3) {
            for (int j = 0; j < width; j += vl)
            {
                vl = __riscv_vsetvl_e8m2(width - j);
                vuint8m2_t v_gray = __riscv_vle8_v_u8m2(src + j, vl);
                vuint8m2x3_t v_bgr = __riscv_vcreate_v_u8m2x3(v_gray, v_gray, v_gray);
                __riscv_vsseg3e8_v_u8m2x3(dst + j * 3, v_bgr, vl);
            }
        }
        else if (scn == 4) {
            vuint8m2_t v_alpha = __riscv_vmv_v_x_u8m2(255, vl);
            for (int j = 0; j < width; j += vl)
            {
                vl = __riscv_vsetvl_e8m2(width - j);
                vuint8m2_t v_gray = __riscv_vle8_v_u8m2(src + j, vl);
                vuint8m2x4_t v_bgrx = __riscv_vcreate_v_u8m2x4(v_gray, v_gray, v_gray, v_alpha);
                __riscv_vsseg4e8_v_u8m2x4(dst + j * 4, v_bgrx, vl);
            }
        }
    }

    return 0;
}
