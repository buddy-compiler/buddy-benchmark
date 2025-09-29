
#include <arm_neon.h>

#include <stdint.h>
#include <cstring>
typedef unsigned int JDIMENSION;
typedef unsigned char JSAMPLE;
typedef JSAMPLE *JSAMPROW;
typedef JSAMPROW *JSAMPARRAY;
typedef JSAMPARRAY *JSAMPIMAGE;
#define RGB_RED  0
#define RGB_GREEN  1
#define RGB_BLUE  2
#define RGB_PIXELSIZE 3
#define ALIGN(alignment)  __attribute__((aligned(alignment)))

#define F_0_344  11277
#define F_0_714  23401
#define F_1_402  22971
#define F_1_772  29033

ALIGN(16) static const int16_t jsimd_ycc_rgb_convert_neon_consts[] = {
  -F_0_344, F_0_714, F_1_402, F_1_772
};

void h2v1_merged_upsample(JDIMENSION output_width, JSAMPIMAGE input_buf,
                                JDIMENSION in_row_group_ctr,
                                JSAMPARRAY output_buf)
{
  JSAMPROW outptr;

  JSAMPROW inptr0, inptr1, inptr2;

  const int16x4_t consts = vld1_s16(jsimd_ycc_rgb_convert_neon_consts);
  const int16x8_t neg_128 = vdupq_n_s16(-128);

  inptr0 = input_buf[0][in_row_group_ctr];
  inptr1 = input_buf[1][in_row_group_ctr];
  inptr2 = input_buf[2][in_row_group_ctr];
  outptr = output_buf[0];

  int cols_remaining = output_width;
  for (; cols_remaining >= 16; cols_remaining -= 16) {
    uint8x8x2_t y = vld2_u8(inptr0);
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

    int16x8_t r_sub_y = vqrdmulhq_lane_s16(vshlq_n_s16(cr_128, 1), consts, 2);

    int16x8_t b_sub_y = vqrdmulhq_lane_s16(vshlq_n_s16(cb_128, 1), consts, 3);
    int16x8_t g_even =
      vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(g_sub_y),
                                     y.val[0]));
    int16x8_t r_even =
      vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(r_sub_y),
                                     y.val[0]));
    int16x8_t b_even =
      vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(b_sub_y),
                                     y.val[0]));
    int16x8_t g_odd =
      vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(g_sub_y),
                                     y.val[1]));
    int16x8_t r_odd =
      vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(r_sub_y),
                                     y.val[1]));
    int16x8_t b_odd =
      vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(b_sub_y),
                                     y.val[1]));
    uint8x8x2_t r = vzip_u8(vqmovun_s16(r_even), vqmovun_s16(r_odd));
    uint8x8x2_t g = vzip_u8(vqmovun_s16(g_even), vqmovun_s16(g_odd));
    uint8x8x2_t b = vzip_u8(vqmovun_s16(b_even), vqmovun_s16(b_odd));

    uint8x16x3_t rgb;
    rgb.val[RGB_RED] = vcombine_u8(r.val[0], r.val[1]);
    rgb.val[RGB_GREEN] = vcombine_u8(g.val[0], g.val[1]);
    rgb.val[RGB_BLUE] = vcombine_u8(b.val[0], b.val[1]);

    vst3q_u8(outptr, rgb);

    inptr0 += 16;
    inptr1 += 8;
    inptr2 += 8;
    outptr += (RGB_PIXELSIZE * 16);
  }

  if (cols_remaining > 0) {

    uint8x8x2_t y = vld2_u8(inptr0);
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

    int16x8_t r_sub_y = vqrdmulhq_lane_s16(vshlq_n_s16(cr_128, 1), consts, 2);

    int16x8_t b_sub_y = vqrdmulhq_lane_s16(vshlq_n_s16(cb_128, 1), consts, 3);

    int16x8_t g_even =
      vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(g_sub_y),
                                     y.val[0]));
    int16x8_t r_even =
      vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(r_sub_y),
                                     y.val[0]));
    int16x8_t b_even =
      vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(b_sub_y),
                                     y.val[0]));
    int16x8_t g_odd =
      vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(g_sub_y),
                                     y.val[1]));
    int16x8_t r_odd =
      vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(r_sub_y),
                                     y.val[1]));
    int16x8_t b_odd =
      vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(b_sub_y),
                                     y.val[1]));

    uint8x8x2_t r = vzip_u8(vqmovun_s16(r_even), vqmovun_s16(r_odd));
    uint8x8x2_t g = vzip_u8(vqmovun_s16(g_even), vqmovun_s16(g_odd));
    uint8x8x2_t b = vzip_u8(vqmovun_s16(b_even), vqmovun_s16(b_odd));

    uint8x8x3_t rgb_h;
    rgb_h.val[RGB_RED] = r.val[1];
    rgb_h.val[RGB_GREEN] = g.val[1];
    rgb_h.val[RGB_BLUE] = b.val[1];
    uint8x8x3_t rgb_l;
    rgb_l.val[RGB_RED] = r.val[0];
    rgb_l.val[RGB_GREEN] = g.val[0];
    rgb_l.val[RGB_BLUE] = b.val[0];

    switch (cols_remaining) {
    case 15:
      vst3_lane_u8(outptr + 14 * RGB_PIXELSIZE, rgb_h, 6);

    case 14:
      vst3_lane_u8(outptr + 13 * RGB_PIXELSIZE, rgb_h, 5);

    case 13:
      vst3_lane_u8(outptr + 12 * RGB_PIXELSIZE, rgb_h, 4);

    case 12:
      vst3_lane_u8(outptr + 11 * RGB_PIXELSIZE, rgb_h, 3);

    case 11:
      vst3_lane_u8(outptr + 10 * RGB_PIXELSIZE, rgb_h, 2);

    case 10:
      vst3_lane_u8(outptr + 9 * RGB_PIXELSIZE, rgb_h, 1);

    case 9:
      vst3_lane_u8(outptr + 8 * RGB_PIXELSIZE, rgb_h, 0);

    case 8:
      vst3_u8(outptr, rgb_l);
      break;
    case 7:
      vst3_lane_u8(outptr + 6 * RGB_PIXELSIZE, rgb_l, 6);

    case 6:
      vst3_lane_u8(outptr + 5 * RGB_PIXELSIZE, rgb_l, 5);

    case 5:
      vst3_lane_u8(outptr + 4 * RGB_PIXELSIZE, rgb_l, 4);

    case 4:
      vst3_lane_u8(outptr + 3 * RGB_PIXELSIZE, rgb_l, 3);

    case 3:
      vst3_lane_u8(outptr + 2 * RGB_PIXELSIZE, rgb_l, 2);

    case 2:
      vst3_lane_u8(outptr + RGB_PIXELSIZE, rgb_l, 1);

    case 1:
      vst3_lane_u8(outptr, rgb_l, 0);

    default:
      break;
    }
  }
}

void h2v2_merged_upsample(JDIMENSION output_width, JSAMPIMAGE input_buf,
                                JDIMENSION in_row_group_ctr,
                                JSAMPARRAY output_buf)
{
  JSAMPROW outptr0, outptr1;

  JSAMPROW inptr0_0, inptr0_1, inptr1, inptr2;

  const int16x4_t consts = vld1_s16(jsimd_ycc_rgb_convert_neon_consts);
  const int16x8_t neg_128 = vdupq_n_s16(-128);

  inptr0_0 = input_buf[0][in_row_group_ctr * 2];
  inptr0_1 = input_buf[0][in_row_group_ctr * 2 + 1];
  inptr1 = input_buf[1][in_row_group_ctr];
  inptr2 = input_buf[2][in_row_group_ctr];
  outptr0 = output_buf[0];
  outptr1 = output_buf[1];

  int cols_remaining = output_width;
  for (; cols_remaining >= 16; cols_remaining -= 16) {

    uint8x8x2_t y0 = vld2_u8(inptr0_0);
    uint8x8x2_t y1 = vld2_u8(inptr0_1);
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

    int16x8_t r_sub_y = vqrdmulhq_lane_s16(vshlq_n_s16(cr_128, 1), consts, 2);

    int16x8_t b_sub_y = vqrdmulhq_lane_s16(vshlq_n_s16(cb_128, 1), consts, 3);

    int16x8_t g0_even =
      vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(g_sub_y),
                                     y0.val[0]));
    int16x8_t r0_even =
      vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(r_sub_y),
                                     y0.val[0]));
    int16x8_t b0_even =
      vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(b_sub_y),
                                     y0.val[0]));
    int16x8_t g0_odd =
      vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(g_sub_y),
                                     y0.val[1]));
    int16x8_t r0_odd =
      vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(r_sub_y),
                                     y0.val[1]));
    int16x8_t b0_odd =
      vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(b_sub_y),
                                     y0.val[1]));
    int16x8_t g1_even =
      vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(g_sub_y),
                                     y1.val[0]));
    int16x8_t r1_even =
      vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(r_sub_y),
                                     y1.val[0]));
    int16x8_t b1_even =
      vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(b_sub_y),
                                     y1.val[0]));
    int16x8_t g1_odd =
      vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(g_sub_y),
                                     y1.val[1]));
    int16x8_t r1_odd =
      vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(r_sub_y),
                                     y1.val[1]));
    int16x8_t b1_odd =
      vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(b_sub_y),
                                     y1.val[1]));

    uint8x8x2_t r0 = vzip_u8(vqmovun_s16(r0_even), vqmovun_s16(r0_odd));
    uint8x8x2_t r1 = vzip_u8(vqmovun_s16(r1_even), vqmovun_s16(r1_odd));
    uint8x8x2_t g0 = vzip_u8(vqmovun_s16(g0_even), vqmovun_s16(g0_odd));
    uint8x8x2_t g1 = vzip_u8(vqmovun_s16(g1_even), vqmovun_s16(g1_odd));
    uint8x8x2_t b0 = vzip_u8(vqmovun_s16(b0_even), vqmovun_s16(b0_odd));
    uint8x8x2_t b1 = vzip_u8(vqmovun_s16(b1_even), vqmovun_s16(b1_odd));

    uint8x16x3_t rgb0, rgb1;
    rgb0.val[RGB_RED] = vcombine_u8(r0.val[0], r0.val[1]);
    rgb1.val[RGB_RED] = vcombine_u8(r1.val[0], r1.val[1]);
    rgb0.val[RGB_GREEN] = vcombine_u8(g0.val[0], g0.val[1]);
    rgb1.val[RGB_GREEN] = vcombine_u8(g1.val[0], g1.val[1]);
    rgb0.val[RGB_BLUE] = vcombine_u8(b0.val[0], b0.val[1]);
    rgb1.val[RGB_BLUE] = vcombine_u8(b1.val[0], b1.val[1]);

    vst3q_u8(outptr0, rgb0);
    vst3q_u8(outptr1, rgb1);

    inptr0_0 += 16;
    inptr0_1 += 16;
    inptr1 += 8;
    inptr2 += 8;
    outptr0 += (RGB_PIXELSIZE * 16);
    outptr1 += (RGB_PIXELSIZE * 16);
  }

  if (cols_remaining > 0) {

    uint8x8x2_t y0 = vld2_u8(inptr0_0);
    uint8x8x2_t y1 = vld2_u8(inptr0_1);
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

    int16x8_t r_sub_y = vqrdmulhq_lane_s16(vshlq_n_s16(cr_128, 1), consts, 2);

    int16x8_t b_sub_y = vqrdmulhq_lane_s16(vshlq_n_s16(cb_128, 1), consts, 3);

    int16x8_t g0_even =
      vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(g_sub_y),
                                     y0.val[0]));
    int16x8_t r0_even =
      vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(r_sub_y),
                                     y0.val[0]));
    int16x8_t b0_even =
      vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(b_sub_y),
                                     y0.val[0]));
    int16x8_t g0_odd =
      vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(g_sub_y),
                                     y0.val[1]));
    int16x8_t r0_odd =
      vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(r_sub_y),
                                     y0.val[1]));
    int16x8_t b0_odd =
      vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(b_sub_y),
                                     y0.val[1]));
    int16x8_t g1_even =
      vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(g_sub_y),
                                     y1.val[0]));
    int16x8_t r1_even =
      vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(r_sub_y),
                                     y1.val[0]));
    int16x8_t b1_even =
      vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(b_sub_y),
                                     y1.val[0]));
    int16x8_t g1_odd =
      vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(g_sub_y),
                                     y1.val[1]));
    int16x8_t r1_odd =
      vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(r_sub_y),
                                     y1.val[1]));
    int16x8_t b1_odd =
      vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(b_sub_y),
                                     y1.val[1]));

    uint8x8x2_t r0 = vzip_u8(vqmovun_s16(r0_even), vqmovun_s16(r0_odd));
    uint8x8x2_t r1 = vzip_u8(vqmovun_s16(r1_even), vqmovun_s16(r1_odd));
    uint8x8x2_t g0 = vzip_u8(vqmovun_s16(g0_even), vqmovun_s16(g0_odd));
    uint8x8x2_t g1 = vzip_u8(vqmovun_s16(g1_even), vqmovun_s16(g1_odd));
    uint8x8x2_t b0 = vzip_u8(vqmovun_s16(b0_even), vqmovun_s16(b0_odd));
    uint8x8x2_t b1 = vzip_u8(vqmovun_s16(b1_even), vqmovun_s16(b1_odd));

    uint8x8x3_t rgb0_h, rgb1_h;
    rgb0_h.val[RGB_RED] = r0.val[1];
    rgb1_h.val[RGB_RED] = r1.val[1];
    rgb0_h.val[RGB_GREEN] = g0.val[1];
    rgb1_h.val[RGB_GREEN] = g1.val[1];
    rgb0_h.val[RGB_BLUE] = b0.val[1];
    rgb1_h.val[RGB_BLUE] = b1.val[1];

    uint8x8x3_t rgb0_l, rgb1_l;
    rgb0_l.val[RGB_RED] = r0.val[0];
    rgb1_l.val[RGB_RED] = r1.val[0];
    rgb0_l.val[RGB_GREEN] = g0.val[0];
    rgb1_l.val[RGB_GREEN] = g1.val[0];
    rgb0_l.val[RGB_BLUE] = b0.val[0];
    rgb1_l.val[RGB_BLUE] = b1.val[0];

    switch (cols_remaining) {
    case 15:
      vst3_lane_u8(outptr0 + 14 * RGB_PIXELSIZE, rgb0_h, 6);
      vst3_lane_u8(outptr1 + 14 * RGB_PIXELSIZE, rgb1_h, 6);

    case 14:
      vst3_lane_u8(outptr0 + 13 * RGB_PIXELSIZE, rgb0_h, 5);
      vst3_lane_u8(outptr1 + 13 * RGB_PIXELSIZE, rgb1_h, 5);

    case 13:
      vst3_lane_u8(outptr0 + 12 * RGB_PIXELSIZE, rgb0_h, 4);
      vst3_lane_u8(outptr1 + 12 * RGB_PIXELSIZE, rgb1_h, 4);

    case 12:
      vst3_lane_u8(outptr0 + 11 * RGB_PIXELSIZE, rgb0_h, 3);
      vst3_lane_u8(outptr1 + 11 * RGB_PIXELSIZE, rgb1_h, 3);

    case 11:
      vst3_lane_u8(outptr0 + 10 * RGB_PIXELSIZE, rgb0_h, 2);
      vst3_lane_u8(outptr1 + 10 * RGB_PIXELSIZE, rgb1_h, 2);

    case 10:
      vst3_lane_u8(outptr0 + 9 * RGB_PIXELSIZE, rgb0_h, 1);
      vst3_lane_u8(outptr1 + 9 * RGB_PIXELSIZE, rgb1_h, 1);

    case 9:
      vst3_lane_u8(outptr0 + 8 * RGB_PIXELSIZE, rgb0_h, 0);
      vst3_lane_u8(outptr1 + 8 * RGB_PIXELSIZE, rgb1_h, 0);

    case 8:
      vst3_u8(outptr0, rgb0_l);
      vst3_u8(outptr1, rgb1_l);
      break;
    case 7:
      vst3_lane_u8(outptr0 + 6 * RGB_PIXELSIZE, rgb0_l, 6);
      vst3_lane_u8(outptr1 + 6 * RGB_PIXELSIZE, rgb1_l, 6);

    case 6:
      vst3_lane_u8(outptr0 + 5 * RGB_PIXELSIZE, rgb0_l, 5);
      vst3_lane_u8(outptr1 + 5 * RGB_PIXELSIZE, rgb1_l, 5);

    case 5:
      vst3_lane_u8(outptr0 + 4 * RGB_PIXELSIZE, rgb0_l, 4);
      vst3_lane_u8(outptr1 + 4 * RGB_PIXELSIZE, rgb1_l, 4);

    case 4:
      vst3_lane_u8(outptr0 + 3 * RGB_PIXELSIZE, rgb0_l, 3);
      vst3_lane_u8(outptr1 + 3 * RGB_PIXELSIZE, rgb1_l, 3);

    case 3:
      vst3_lane_u8(outptr0 + 2 * RGB_PIXELSIZE, rgb0_l, 2);
      vst3_lane_u8(outptr1 + 2 * RGB_PIXELSIZE, rgb1_l, 2);

    case 2:
      vst3_lane_u8(outptr0 + 1 * RGB_PIXELSIZE, rgb0_l, 1);
      vst3_lane_u8(outptr1 + 1 * RGB_PIXELSIZE, rgb1_l, 1);

    case 1:
      vst3_lane_u8(outptr0, rgb0_l, 0);
      vst3_lane_u8(outptr1, rgb1_l, 0);

    default:
      break;
    }
  }
}
