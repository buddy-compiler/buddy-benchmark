#include <arm_neon.h>
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

inline uint8x8_t convertToGray(const uint16x8_t & v_r,
                               const uint16x8_t & v_g,
                               const uint16x8_t & v_b,
                               const uint16x4_t & v_r2y,
                               const uint16x4_t & v_g2y,
                               const uint16x4_t & v_b2y)
{
    uint32x4_t v_dst0 = vmull_u16(vget_low_u16(v_g), v_g2y);
    uint32x4_t v_dst1 = vmull_u16(vget_high_u16(v_g), v_g2y);
    v_dst0 = vmlal_u16(v_dst0, vget_low_u16(v_r), v_r2y);
    v_dst1 = vmlal_u16(v_dst1, vget_high_u16(v_r), v_r2y);
    v_dst0 = vmlal_u16(v_dst0, vget_low_u16(v_b), v_b2y);
    v_dst1 = vmlal_u16(v_dst1, vget_high_u16(v_b), v_b2y);
    uint8x8_t v_gray = vqmovn_u16(vcombine_u16(vrshrn_n_u32(v_dst0, SHIFT),
                                               vrshrn_n_u32(v_dst1, SHIFT)));

    return v_gray;
}

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
    uint16x4_t v_r2y = vdup_n_u16(R2Y),
               v_g2y = vdup_n_u16(G2Y),
               v_b2y = vdup_n_u16(B2Y);
    size_t roiw16 = width >= 15 ? width - 15 : 0;
    size_t roiw8 = width >= 7 ? width - 7 : 0;

    for (size_t i = 0u; i < (size_t)height; ++i)
    {
        const unsigned char * src = src_data + i * src_step;
        unsigned char * dst = dst_data + i * dst_step;
        size_t sj = 0u, dj = 0u;

        if (scn == 3) {

            for (; dj < roiw16; sj += 48, dj += 16)
            {
                uint8x16x3_t v_src0 = vld3q_u8(src + sj);
                uint16x8_t v_b = vmovl_u8(vget_low_u8(v_src0.val[0])),
                        v_g = vmovl_u8(vget_low_u8(v_src0.val[1])),
                        v_r = vmovl_u8(vget_low_u8(v_src0.val[2]));
                uint8x8_t v_gray0 = convertToGray(v_r, v_g, v_b, v_r2y, v_g2y, v_b2y);

                v_b = vmovl_u8(vget_high_u8(v_src0.val[0])),
                v_g = vmovl_u8(vget_high_u8(v_src0.val[1])),
                v_r = vmovl_u8(vget_high_u8(v_src0.val[2]));
                uint8x8_t v_gray1 = convertToGray(v_r, v_g, v_b, v_r2y, v_g2y, v_b2y);

                vst1q_u8(dst + dj, vcombine_u8(v_gray0, v_gray1));
            }

            if (dj < roiw8)
            {
                uint8x8x3_t v_src = vld3_u8(src + sj);
                uint16x8_t v_b = vmovl_u8(v_src.val[0]),
                           v_g = vmovl_u8(v_src.val[1]),
                           v_r = vmovl_u8(v_src.val[2]);

                uint8x8_t v_gray = convertToGray(v_r, v_g, v_b, v_r2y, v_g2y, v_b2y);

                vst1_u8(dst + dj, v_gray);
                sj += 24; dj += 8;
            }

            for (; dj < (size_t)width; sj += 3, dj++)
            {
                uint32_t val = src[sj] * B2Y + src[sj + 1] * G2Y + src[sj + 2] * R2Y;
                dst[dj] = saturate_cast<unsigned char>((val + SHIFT_DELTA) >> SHIFT);
            }
        }
        else if (scn == 4) {

            for (; dj < roiw16; sj += 64, dj += 16)
            {
                uint8x16x4_t v_src0 = vld4q_u8(src + sj);
                uint16x8_t v_b = vmovl_u8(vget_low_u8(v_src0.val[0])),
                        v_g = vmovl_u8(vget_low_u8(v_src0.val[1])),
                        v_r = vmovl_u8(vget_low_u8(v_src0.val[2]));
                uint8x8_t v_gray0 = convertToGray(v_r, v_g, v_b, v_r2y, v_g2y, v_b2y);

                v_b = vmovl_u8(vget_high_u8(v_src0.val[0])),
                v_g = vmovl_u8(vget_high_u8(v_src0.val[1])),
                v_r = vmovl_u8(vget_high_u8(v_src0.val[2]));
                uint8x8_t v_gray1 = convertToGray(v_r, v_g, v_b, v_r2y, v_g2y, v_b2y);

                vst1q_u8(dst + dj, vcombine_u8(v_gray0, v_gray1));
            }
            if (dj < roiw8)
            {
                uint8x8x4_t v_src = vld4_u8(src + sj);
                uint16x8_t v_b = vmovl_u8(v_src.val[0]),
                        v_g = vmovl_u8(v_src.val[1]),
                        v_r = vmovl_u8(v_src.val[2]);
                uint8x8_t v_gray = convertToGray(v_r, v_g, v_b, v_r2y, v_g2y, v_b2y);

                vst1_u8(dst + dj, v_gray);
                sj += 32; dj += 8;
            }

            for (; dj < (size_t)width; sj += 4, dj++)
            {
                uint32_t val = src[sj] * B2Y + src[sj + 1] * G2Y + src[sj + 2] * R2Y;
                dst[dj] = saturate_cast<unsigned char>((val + SHIFT_DELTA) >> SHIFT);
            }
        }
    }

    return 0;
}
