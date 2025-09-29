#include <riscv_vector.h>
#include <cstdint>
#include <cstddef>
#include <algorithm>
int cvtBGRtoBGR5x5(const unsigned char * src_data, size_t src_step,
                   unsigned char * dst_data, size_t dst_step,
                   int width, int height,
                   int scn, bool swapBlue, int greenBits)
{
    if (!src_data || !dst_data || width <= 0 || height <= 0)
        return 1;

    if (scn != 3 && scn != 4)
        return 1;

    if (greenBits != 5 && greenBits != 6)
        return 1;

    dst_step /= sizeof(uint16_t);

    for (int i = 0; i < height; i++)
    {
        const unsigned char * src = src_data + i * src_step;
        uint16_t * dst = reinterpret_cast<uint16_t*>(dst_data + i * dst_step * sizeof(uint16_t));
        int vl;
        for (int j = 0; j < width; j += vl)
        {
            vl = __riscv_vsetvl_e8m2(width - j);
            vuint16m4_t vec_srcB, vec_srcG, vec_srcR, vec_srcA = __riscv_vmv_v_x_u16m4(0, vl);
            if (scn == 3)
            {
                auto x = __riscv_vlseg3e8_v_u8m2x3(src + j * 3, vl);
                vec_srcB = __riscv_vwcvtu_x(__riscv_vget_v_u8m2x3_u8m2(x, 0), vl);
                vec_srcG = __riscv_vwcvtu_x(__riscv_vget_v_u8m2x3_u8m2(x, 1), vl);
                vec_srcR = __riscv_vwcvtu_x(__riscv_vget_v_u8m2x3_u8m2(x, 2), vl);
            }
            else
            {
                auto x = __riscv_vlseg4e8_v_u8m2x4(src + j * 4, vl);
                vec_srcB = __riscv_vwcvtu_x(__riscv_vget_v_u8m2x4_u8m2(x, 0), vl);
                vec_srcG = __riscv_vwcvtu_x(__riscv_vget_v_u8m2x4_u8m2(x, 1), vl);
                vec_srcR = __riscv_vwcvtu_x(__riscv_vget_v_u8m2x4_u8m2(x, 2), vl);
                vec_srcA = __riscv_vwcvtu_x(__riscv_vget_v_u8m2x4_u8m2(x, 3), vl);
            }
            if (swapBlue)
            {
                auto t = vec_srcB;
                vec_srcB = vec_srcR, vec_srcR = t;
            }

            auto vec_dst = __riscv_vsrl(vec_srcB, 3, vl);
            if (greenBits == 6)
            {
                vec_dst = __riscv_vor(__riscv_vor(vec_dst, __riscv_vsll(__riscv_vand(vec_srcG, ~3, vl), 3, vl), vl), __riscv_vsll(__riscv_vand(vec_srcR, ~7, vl), 8, vl), vl);
            }
            else
            {
                vec_dst = __riscv_vor(__riscv_vor(vec_dst, __riscv_vsll(__riscv_vand(vec_srcG, ~7, vl), 2, vl), vl), __riscv_vsll(__riscv_vand(vec_srcR, ~7, vl), 7, vl), vl);
                vec_dst = __riscv_vor_mu(__riscv_vmsne(vec_srcA, 0, vl), vec_dst, vec_dst, 0x8000, vl);
            }
            __riscv_vse16(dst + j, vec_dst, vl);
        }
    }

    return 0;
}
