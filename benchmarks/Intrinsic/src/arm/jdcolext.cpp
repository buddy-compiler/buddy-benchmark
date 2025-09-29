#include <arm_neon.h>

#include <stdint.h>

typedef unsigned int JDIMENSION;
typedef unsigned char JSAMPLE;
typedef JSAMPLE *JSAMPROW;
typedef JSAMPROW *JSAMPARRAY;
typedef JSAMPARRAY *JSAMPIMAGE;
#define ALIGN(alignment)  __attribute__((aligned(alignment)))
#define RGB_RED  0
#define RGB_GREEN  1
#define RGB_BLUE  2
#define RGB_PIXELSIZE 3

#define F_0_344  11277
#define F_0_714  23401
#define F_1_402  22971
#define F_1_772  29033

ALIGN(16) static const int16_t jsimd_ycc_rgb_convert_neon_consts[] = {
  -F_0_344, F_0_714, F_1_402, F_1_772
};

void ycc_rgb_convert(JDIMENSION output_width, JSAMPIMAGE input_buf,
                           JDIMENSION input_row, JSAMPARRAY output_buf,
                           int num_rows)
{
  JSAMPROW outptr;

  JSAMPROW inptr0, inptr1, inptr2;

  const int16x4_t consts = vld1_s16(jsimd_ycc_rgb_convert_neon_consts);
  const int16x8_t neg_128 = vdupq_n_s16(-128);

  while (--num_rows >= 0) {
    inptr0 = input_buf[0][input_row];
    inptr1 = input_buf[1][input_row];
    inptr2 = input_buf[2][input_row];
    input_row++;
    outptr = *output_buf++;
    int cols_remaining = output_width;
    for (; cols_remaining >= 16; cols_remaining -= 16) {
      uint8x16_t y  = vld1q_u8(inptr0);
      uint8x16_t cb = vld1q_u8(inptr1);
      uint8x16_t cr = vld1q_u8(inptr2);

      int16x8_t cr_128_l =
        vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(neg_128),
                                       vget_low_u8(cr)));
      int16x8_t cr_128_h =
        vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(neg_128),
                                       vget_high_u8(cr)));
      int16x8_t cb_128_l =
        vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(neg_128),
                                       vget_low_u8(cb)));
      int16x8_t cb_128_h =
        vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(neg_128),
                                       vget_high_u8(cb)));

      int32x4_t g_sub_y_ll = vmull_lane_s16(vget_low_s16(cb_128_l), consts, 0);
      int32x4_t g_sub_y_lh = vmull_lane_s16(vget_high_s16(cb_128_l),
                                            consts, 0);
      int32x4_t g_sub_y_hl = vmull_lane_s16(vget_low_s16(cb_128_h), consts, 0);
      int32x4_t g_sub_y_hh = vmull_lane_s16(vget_high_s16(cb_128_h),
                                            consts, 0);
      g_sub_y_ll = vmlsl_lane_s16(g_sub_y_ll, vget_low_s16(cr_128_l),
                                  consts, 1);
      g_sub_y_lh = vmlsl_lane_s16(g_sub_y_lh, vget_high_s16(cr_128_l),
                                  consts, 1);
      g_sub_y_hl = vmlsl_lane_s16(g_sub_y_hl, vget_low_s16(cr_128_h),
                                  consts, 1);
      g_sub_y_hh = vmlsl_lane_s16(g_sub_y_hh, vget_high_s16(cr_128_h),
                                  consts, 1);

      int16x8_t g_sub_y_l = vcombine_s16(vrshrn_n_s32(g_sub_y_ll, 15),
                                         vrshrn_n_s32(g_sub_y_lh, 15));
      int16x8_t g_sub_y_h = vcombine_s16(vrshrn_n_s32(g_sub_y_hl, 15),
                                         vrshrn_n_s32(g_sub_y_hh, 15));

      int16x8_t r_sub_y_l = vqrdmulhq_lane_s16(vshlq_n_s16(cr_128_l, 1),
                                               consts, 2);
      int16x8_t r_sub_y_h = vqrdmulhq_lane_s16(vshlq_n_s16(cr_128_h, 1),
                                               consts, 2);

      int16x8_t b_sub_y_l = vqrdmulhq_lane_s16(vshlq_n_s16(cb_128_l, 1),
                                               consts, 3);
      int16x8_t b_sub_y_h = vqrdmulhq_lane_s16(vshlq_n_s16(cb_128_h, 1),
                                               consts, 3);

      int16x8_t r_l =
        vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(r_sub_y_l),
                                       vget_low_u8(y)));
      int16x8_t r_h =
        vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(r_sub_y_h),
                                       vget_high_u8(y)));
      int16x8_t b_l =
        vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(b_sub_y_l),
                                       vget_low_u8(y)));
      int16x8_t b_h =
        vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(b_sub_y_h),
                                       vget_high_u8(y)));
      int16x8_t g_l =
        vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(g_sub_y_l),
                                       vget_low_u8(y)));
      int16x8_t g_h =
        vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(g_sub_y_h),
                                       vget_high_u8(y)));

      uint8x16x3_t rgb;

      rgb.val[RGB_RED] = vcombine_u8(vqmovun_s16(r_l), vqmovun_s16(r_h));
      rgb.val[RGB_GREEN] = vcombine_u8(vqmovun_s16(g_l), vqmovun_s16(g_h));
      rgb.val[RGB_BLUE] = vcombine_u8(vqmovun_s16(b_l), vqmovun_s16(b_h));

      vst3q_u8(outptr, rgb);

      inptr0 += 16;
      inptr1 += 16;
      inptr2 += 16;
      outptr += (RGB_PIXELSIZE * 16);
    }

    if (cols_remaining >= 8) {
      uint8x8_t y  = vld1_u8(inptr0);
      uint8x8_t cb = vld1_u8(inptr1);
      uint8x8_t cr = vld1_u8(inptr2);

      int16x8_t cr_128 =
        vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(neg_128), cr));
      int16x8_t cb_128 =
        vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(neg_128), cb));

      int32x4_t g_sub_y_l = vmull_lane_s16(vget_low_s16(cb_128), consts, 0);
      int32x4_t g_sub_y_h = vmull_lane_s16(vget_high_s16(cb_128), consts, 0);
      g_sub_y_l = vmlsl_lane_s16(g_sub_y_l, vget_low_s16(cr_128), consts, 1);
      g_sub_y_h = vmlsl_lane_s16(g_sub_y_h, vget_high_s16(cr_128), consts, 1);

      int16x8_t g_sub_y = vcombine_s16(vrshrn_n_s32(g_sub_y_l, 15),
                                       vrshrn_n_s32(g_sub_y_h, 15));

      int16x8_t r_sub_y = vqrdmulhq_lane_s16(vshlq_n_s16(cr_128, 1),
                                             consts, 2);

      int16x8_t b_sub_y = vqrdmulhq_lane_s16(vshlq_n_s16(cb_128, 1),
                                             consts, 3);

      int16x8_t r =
        vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(r_sub_y), y));
      int16x8_t b =
        vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(b_sub_y), y));
      int16x8_t g =
        vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(g_sub_y), y));

      uint8x8x3_t rgb;

      rgb.val[RGB_RED] = vqmovun_s16(r);
      rgb.val[RGB_GREEN] = vqmovun_s16(g);
      rgb.val[RGB_BLUE] = vqmovun_s16(b);

      vst3_u8(outptr, rgb);

      inptr0 += 8;
      inptr1 += 8;
      inptr2 += 8;
      outptr += (RGB_PIXELSIZE * 8);
      cols_remaining -= 8;
    }

    if (cols_remaining > 0) {
      uint8x8_t y  = vld1_u8(inptr0);
      uint8x8_t cb = vld1_u8(inptr1);
      uint8x8_t cr = vld1_u8(inptr2);

      int16x8_t cr_128 =
        vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(neg_128), cr));
      int16x8_t cb_128 =
        vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(neg_128), cb));

      int32x4_t g_sub_y_l = vmull_lane_s16(vget_low_s16(cb_128), consts, 0);
      int32x4_t g_sub_y_h = vmull_lane_s16(vget_high_s16(cb_128), consts, 0);
      g_sub_y_l = vmlsl_lane_s16(g_sub_y_l, vget_low_s16(cr_128), consts, 1);
      g_sub_y_h = vmlsl_lane_s16(g_sub_y_h, vget_high_s16(cr_128), consts, 1);

      int16x8_t g_sub_y = vcombine_s16(vrshrn_n_s32(g_sub_y_l, 15),
                                       vrshrn_n_s32(g_sub_y_h, 15));

      int16x8_t r_sub_y = vqrdmulhq_lane_s16(vshlq_n_s16(cr_128, 1),
                                             consts, 2);

      int16x8_t b_sub_y = vqrdmulhq_lane_s16(vshlq_n_s16(cb_128, 1),
                                             consts, 3);

      int16x8_t r =
        vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(r_sub_y), y));
      int16x8_t b =
        vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(b_sub_y), y));
      int16x8_t g =
        vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(g_sub_y), y));

      uint8x8x3_t rgb;

      rgb.val[RGB_RED] = vqmovun_s16(r);
      rgb.val[RGB_GREEN] = vqmovun_s16(g);
      rgb.val[RGB_BLUE] = vqmovun_s16(b);

      switch (cols_remaining) {
      case 7:
        vst3_lane_u8(outptr + 6 * RGB_PIXELSIZE, rgb, 6);

      case 6:
        vst3_lane_u8(outptr + 5 * RGB_PIXELSIZE, rgb, 5);

      case 5:
        vst3_lane_u8(outptr + 4 * RGB_PIXELSIZE, rgb, 4);

      case 4:
        vst3_lane_u8(outptr + 3 * RGB_PIXELSIZE, rgb, 3);

      case 3:
        vst3_lane_u8(outptr + 2 * RGB_PIXELSIZE, rgb, 2);

      case 2:
        vst3_lane_u8(outptr + RGB_PIXELSIZE, rgb, 1);

      case 1:
        vst3_lane_u8(outptr, rgb, 0);

      default:
        break;
      }
    }
  }
}
