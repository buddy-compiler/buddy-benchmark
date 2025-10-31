#include <riscv_vector.h>
#include <cstdint>
#include <algorithm>
enum
{
    SHIFT = 15,
    SHIFT_DELTA = 1 << (SHIFT - 1),

    R2Y_BT601   = 9798,
    G2Y_BT601   = 19235,
    B2Y_BT601   = 3735,

    R2Y_BT709   = 3483,
    G2Y_BT709   = 11718,
    B2Y_BT709   = 1183,
};
template<typename T>
inline T saturate_cast(int val)
{
    return static_cast<T>(std::max(0, std::min(val, 255)));
}
int cvtBGRtoGray(const unsigned char* src_data, size_t src_step,
                 unsigned char* dst_data, size_t dst_step,
                 int width, int height,
                 int depth, int scn)
{
    if (depth != 0)
      return 1;
    if (scn != 3 && scn != 4) return 1;
    const uint32_t R2Y = R2Y_BT601;
    const uint32_t G2Y = G2Y_BT601;
    const uint32_t B2Y = B2Y_BT601;

    for (size_t i = 0u; i < (size_t)height; ++i)
    {
        const unsigned char * src = src_data + i * src_step;
        unsigned char * dst = dst_data + i * dst_step;

        int vl = 0;

        if (scn == 3) {
            for (int j = 0; j < width; j += vl)
            {
                vl = __riscv_vsetvl_e8m2(width - j);
                vuint8m2x3_t v_bgr = __riscv_vlseg3e8_v_u8m2x3(src + j * 3, vl);
                vuint8m2_t v_b = __riscv_vget_v_u8m2x3_u8m2(v_bgr, 0);
                vuint8m2_t v_g = __riscv_vget_v_u8m2x3_u8m2(v_bgr, 1);
                vuint8m2_t v_r = __riscv_vget_v_u8m2x3_u8m2(v_bgr, 2);
                vuint16m4_t v_b16 = __riscv_vwcvtu_x_x_v_u16m4(v_b, vl);
                vuint16m4_t v_g16 = __riscv_vwcvtu_x_x_v_u16m4(v_g, vl);
                vuint16m4_t v_r16 = __riscv_vwcvtu_x_x_v_u16m4(v_r, vl);
                vuint32m8_t v_b32 = __riscv_vwcvtu_x_x_v_u32m8(v_b16, vl);
                vuint32m8_t v_g32 = __riscv_vwcvtu_x_x_v_u32m8(v_g16, vl);
                vuint32m8_t v_r32 = __riscv_vwcvtu_x_x_v_u32m8(v_r16, vl);
                vuint32m8_t v_gray32 = __riscv_vmul_vx_u32m8(v_b32, B2Y, vl);
                v_gray32 = __riscv_vmadd_vx_u32m8(v_g32, G2Y, v_gray32, vl);
                v_gray32 = __riscv_vmadd_vx_u32m8(v_r32, R2Y, v_gray32, vl);
                v_gray32 = __riscv_vadd_vx_u32m8(v_gray32, SHIFT_DELTA, vl);
                vuint16m4_t v_gray16 = __riscv_vnsrl_wx_u16m4(v_gray32, SHIFT, vl);
                vuint8m2_t v_gray = __riscv_vnclipu_wx_u8m2(v_gray16, 0, __RISCV_VXRM_RNU, vl);
                __riscv_vse8_v_u8m2(dst + j, v_gray, vl);
            }
        }
        else if (scn == 4) {
            for (int j = 0; j < width; j += vl)
            {
                vl = __riscv_vsetvl_e8m2(width - j);
                vuint8m2x4_t v_bgrx = __riscv_vlseg4e8_v_u8m2x4(src + j * 4, vl);
                vuint8m2_t v_b = __riscv_vget_v_u8m2x4_u8m2(v_bgrx, 0);
                vuint8m2_t v_g = __riscv_vget_v_u8m2x4_u8m2(v_bgrx, 1);
                vuint8m2_t v_r = __riscv_vget_v_u8m2x4_u8m2(v_bgrx, 2);

                vuint16m4_t v_b16 = __riscv_vwcvtu_x_x_v_u16m4(v_b, vl);
                vuint16m4_t v_g16 = __riscv_vwcvtu_x_x_v_u16m4(v_g, vl);
                vuint16m4_t v_r16 = __riscv_vwcvtu_x_x_v_u16m4(v_r, vl);
                vuint32m8_t v_b32 = __riscv_vwcvtu_x_x_v_u32m8(v_b16, vl);
                vuint32m8_t v_g32 = __riscv_vwcvtu_x_x_v_u32m8(v_g16, vl);
                vuint32m8_t v_r32 = __riscv_vwcvtu_x_x_v_u32m8(v_r16, vl);
                vuint32m8_t v_gray32 = __riscv_vmul_vx_u32m8(v_b32, B2Y, vl);
                v_gray32 = __riscv_vmadd_vx_u32m8(v_g32, G2Y, v_gray32, vl);
                v_gray32 = __riscv_vmadd_vx_u32m8(v_r32, R2Y, v_gray32, vl);
                v_gray32 = __riscv_vadd_vx_u32m8(v_gray32, SHIFT_DELTA, vl);
                vuint16m4_t v_gray16 = __riscv_vnsrl_wx_u16m4(v_gray32, SHIFT, vl);
                vuint8m2_t v_gray = __riscv_vnclipu_wx_u8m2(v_gray16, 0, __RISCV_VXRM_RNU, vl);
                __riscv_vse8_v_u8m2(dst + j, v_gray, vl);
            }
        }
    }

    return 0;
}
