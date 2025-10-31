#include <riscv_vector.h>
#include <cstdint>
#include <algorithm>
template<typename T>
inline T saturate_cast(int32_t val)
{
    if (val > 255) return 255;
    if (val < 0) return 0;
    return static_cast<T>(val);
}
void cvtBGRtoYUV(const unsigned char* src_data, size_t src_step,
                 unsigned char* dst_data, size_t dst_step,
                 int width, int height, int depth, int scn)
{

    static constexpr int B2Y = 1868, G2Y = 9617, R2Y = 4899;
    static constexpr int B2U = 8061, R2V = 14369;
    static constexpr int YCB = 9241, YCR = 11682;

    for (int i = 0; i < height; i++)
    {
        const unsigned char* src = src_data + i * src_step;
        unsigned char* dst = dst_data + i * dst_step;
        int j = 0;
        int vl;
        for (; j < width; j += vl)
        {
            vl = __riscv_vsetvl_e8m1(width - j);
            vuint8m1_t vec_srcB, vec_srcG, vec_srcR;
            if (scn == 3)
            {
                auto x = __riscv_vlseg3e8_v_u8m1x3(src + j * scn, vl);
                vec_srcB = __riscv_vget_v_u8m1x3_u8m1(x, 0);
                vec_srcG = __riscv_vget_v_u8m1x3_u8m1(x, 1);
                vec_srcR = __riscv_vget_v_u8m1x3_u8m1(x, 2);
            }
            else
            {
                auto x = __riscv_vlseg4e8_v_u8m1x4(src + j * scn, vl);
                vec_srcB = __riscv_vget_v_u8m1x4_u8m1(x, 0);
                vec_srcG = __riscv_vget_v_u8m1x4_u8m1(x, 1);
                vec_srcR = __riscv_vget_v_u8m1x4_u8m1(x, 2);
            }
            auto vec_srcB_32 = __riscv_vreinterpret_v_u32m4_i32m4(__riscv_vzext_vf4(vec_srcB, vl));
            auto vec_srcG_32 = __riscv_vreinterpret_v_u32m4_i32m4(__riscv_vzext_vf4(vec_srcG, vl));
            auto vec_srcR_32 = __riscv_vreinterpret_v_u32m4_i32m4(__riscv_vzext_vf4(vec_srcR, vl));

            auto vec_Y = __riscv_vsra(__riscv_vmadd(vec_srcB_32, B2Y,
                                      __riscv_vmadd(vec_srcG_32, G2Y,
                                      __riscv_vmul(vec_srcR_32, R2Y, vl), vl), vl), 14, vl);

            auto delta_cb = __riscv_vmv_v_x_i32m4((128 << 14), vl);
            auto vec_Cb = __riscv_vmadd(__riscv_vsub(vec_srcB_32, vec_Y, vl), YCB, delta_cb, vl);
            auto delta_cr = __riscv_vmv_v_x_i32m4((128 << 14), vl);
            auto vec_Cr = __riscv_vmadd(__riscv_vsub(vec_srcR_32, vec_Y, vl), YCR, delta_cr, vl);

            vuint8m1x3_t x{};
            x = __riscv_vset_v_u8m1_u8m1x3(x, 0, __riscv_vnclipu(__riscv_vnclipu(__riscv_vreinterpret_v_i32m4_u32m4(__riscv_vmax(vec_Y, 0, vl)), 0, __RISCV_VXRM_RNU, vl), 0, __RISCV_VXRM_RNU, vl));
            x = __riscv_vset_v_u8m1_u8m1x3(x, 1, __riscv_vnclipu(__riscv_vnclipu(__riscv_vreinterpret_v_i32m4_u32m4(__riscv_vmax(vec_Cr, 0, vl)), 14, __RISCV_VXRM_RNU, vl), 0, __RISCV_VXRM_RNU, vl));
            x = __riscv_vset_v_u8m1_u8m1x3(x, 2, __riscv_vnclipu(__riscv_vnclipu(__riscv_vreinterpret_v_i32m4_u32m4(__riscv_vmax(vec_Cb, 0, vl)), 14, __RISCV_VXRM_RNU, vl), 0, __RISCV_VXRM_RNU, vl));
            __riscv_vsseg3e8(dst + j * 3, x, vl);

        }
    }
}
