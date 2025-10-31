#include <arm_neon.h>
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

    const int16_t coeff_YR = 4899;
    const int16_t coeff_YG = 9617;
    const int16_t coeff_YB = 1868;
    const int16_t coeff_CrG = 6860;
    const int16_t coeff_CrB = 1332;
    const int16_t coeff_CbR = 2765;
    const int16_t coeff_CbG = 5427;

    const int16x4_t v_YR = vdup_n_s16(coeff_YR);
    const int16x4_t v_YG = vdup_n_s16(coeff_YG);
    const int16x4_t v_YB = vdup_n_s16(coeff_YB);
    const int16x4_t v_CrG = vdup_n_s16(coeff_CrG);
    const int16x4_t v_CrB = vdup_n_s16(coeff_CrB);
    const int16x4_t v_CbR = vdup_n_s16(coeff_CbR);
    const int16x4_t v_CbG = vdup_n_s16(coeff_CbG);

    for (int i = 0; i < height; ++i)
    {
        const unsigned char* src = src_data + i * src_step;
        unsigned char* dst = dst_data + i * dst_step;
        int j = 0;
        size_t roiw8 = width >= 7 ? width - 7 : 0;
        for (; j < roiw8; j += 8)
        {
            uint16x8_t v_b, v_g, v_r;

            if (scn == 3) {
                uint8x8x3_t v_bgr = vld3_u8(src + j * 3);
                v_b = vmovl_u8(v_bgr.val[0]);
                v_g = vmovl_u8(v_bgr.val[1]);
                v_r = vmovl_u8(v_bgr.val[2]);
            } else {

                uint8x8x4_t v_bgrx = vld4_u8(src + j * 4);
                v_b = vmovl_u8(v_bgrx.val[0]);
                v_g = vmovl_u8(v_bgrx.val[1]);
                v_r = vmovl_u8(v_bgrx.val[2]);
            }
            int32x4_t v_Y_low = vmull_s16(vget_low_s16(vreinterpretq_s16_u16(v_g)), v_YG);
            int32x4_t v_Y_high = vmull_s16(vget_high_s16(vreinterpretq_s16_u16(v_g)), v_YG);

            v_Y_low = vmlal_s16(v_Y_low, vget_low_s16(vreinterpretq_s16_u16(v_b)), v_YB);
            v_Y_high = vmlal_s16(v_Y_high, vget_high_s16(vreinterpretq_s16_u16(v_b)), v_YB);

            v_Y_low = vmlal_s16(v_Y_low, vget_low_s16(vreinterpretq_s16_u16(v_r)), v_YR);
            v_Y_high = vmlal_s16(v_Y_high, vget_high_s16(vreinterpretq_s16_u16(v_r)), v_YR);
            int32x4_t v_Cr_low = vshll_n_s16(vget_low_s16(vreinterpretq_s16_u16(v_r)), 13);
            int32x4_t v_Cr_high = vshll_n_s16(vget_high_s16(vreinterpretq_s16_u16(v_r)), 13);

            v_Cr_low = vmlsl_s16(v_Cr_low, vget_low_s16(vreinterpretq_s16_u16(v_g)), v_CrG);
            v_Cr_high = vmlsl_s16(v_Cr_high, vget_high_s16(vreinterpretq_s16_u16(v_g)), v_CrG);

            v_Cr_low = vmlsl_s16(v_Cr_low, vget_low_s16(vreinterpretq_s16_u16(v_b)), v_CrB);
            v_Cr_high = vmlsl_s16(v_Cr_high, vget_high_s16(vreinterpretq_s16_u16(v_b)), v_CrB);
            int32x4_t v_Cb_low = vshll_n_s16(vget_low_s16(vreinterpretq_s16_u16(v_b)), 13);
            int32x4_t v_Cb_high = vshll_n_s16(vget_high_s16(vreinterpretq_s16_u16(v_b)), 13);

            v_Cb_low = vmlsl_s16(v_Cb_low, vget_low_s16(vreinterpretq_s16_u16(v_r)), v_CbR);
            v_Cb_high = vmlsl_s16(v_Cb_high, vget_high_s16(vreinterpretq_s16_u16(v_r)), v_CbR);

            v_Cb_low = vmlsl_s16(v_Cb_low, vget_low_s16(vreinterpretq_s16_u16(v_g)), v_CbG);
            v_Cb_high = vmlsl_s16(v_Cb_high, vget_high_s16(vreinterpretq_s16_u16(v_g)), v_CbG);
            int16x4_t v_Y_low_s16 = vrshrn_n_s32(v_Y_low, 14);
            int16x4_t v_Y_high_s16 = vrshrn_n_s32(v_Y_high, 14);
            int16x4_t v_Cr_low_s16 = vrshrn_n_s32(v_Cr_low, 14);
            int16x4_t v_Cr_high_s16 = vrshrn_n_s32(v_Cr_high, 14);
            int16x4_t v_Cb_low_s16 = vrshrn_n_s32(v_Cb_low, 14);
            int16x4_t v_Cb_high_s16 = vrshrn_n_s32(v_Cb_high, 14);
            int16x8_t v_128 = vmovq_n_s16(128);
            int16x8_t v_Cr = vaddq_s16(v_128, vcombine_s16(v_Cr_low_s16, v_Cr_high_s16));
            int16x8_t v_Cb = vaddq_s16(v_128, vcombine_s16(v_Cb_low_s16, v_Cb_high_s16));
            uint8x8x3_t v_ycrcb;
            v_ycrcb.val[0] = vqmovn_u16(vreinterpretq_u16_s16(vcombine_s16(v_Y_low_s16, v_Y_high_s16)));
            v_ycrcb.val[1] = vqmovun_s16(v_Cr);
            v_ycrcb.val[2] = vqmovun_s16(v_Cb);

            vst3_u8(dst + j * 3, v_ycrcb);
        }
        for (; j < width; ++j)
        {
            int32_t b = src[j * scn];
            int32_t g = src[j * scn + 1];
            int32_t r = src[j * scn + 2];
            int32_t Y = (r * coeff_YR + g * coeff_YG + b * coeff_YB + (1 << 13)) >> 14;
            int32_t Cr = 128 + ((r * 8192 - g * coeff_CrG - b * coeff_CrB + (1 << 13)) >> 14);
            int32_t Cb = 128 + ((r * (-coeff_CbR) - g * coeff_CbG + b * 8192 + (1 << 13)) >> 14);

            dst[j * 3] = saturate_cast<unsigned char>(Y);
            dst[j * 3 + 1] = saturate_cast<unsigned char>(Cr);
            dst[j * 3 + 2] = saturate_cast<unsigned char>(Cb);
        }
    }
}
