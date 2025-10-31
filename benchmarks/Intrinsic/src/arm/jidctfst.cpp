#include <arm_neon.h>

#include <stdint.h>

typedef unsigned int JDIMENSION;
typedef unsigned char JSAMPLE;
typedef JSAMPLE *JSAMPROW;
typedef JSAMPROW *JSAMPARRAY;
typedef JSAMPARRAY *JSAMPIMAGE;

#define ALIGN(alignment)  __attribute__((aligned(alignment)))

typedef short DCTELEM;
#define DCTSIZE             8
#define DCTSIZE2            64
#define CENTERJSAMPLE    128

typedef short JCOEF;
typedef JCOEF *JCOEFPTR;
typedef short IFAST_MULT_TYPE;

#define PASS1_BITS  2

#define F_0_082  2688
#define F_0_414  13568
#define F_0_847  27776
#define F_0_613  20096

ALIGN(16) static const int16_t jsimd_idct_ifast_neon_consts[] = {
  F_0_082, F_0_414, F_0_847, F_0_613
};

void idct_ifast(void *dct_table, JCOEFPTR coef_block,
                JSAMPARRAY output_buf, JDIMENSION output_col, JSAMPROW sample_range_limit)
{
  IFAST_MULT_TYPE *quantptr = (IFAST_MULT_TYPE*)dct_table;

  int16x8_t row0 = vld1q_s16(coef_block + 0 * DCTSIZE);
  int16x8_t row1 = vld1q_s16(coef_block + 1 * DCTSIZE);
  int16x8_t row2 = vld1q_s16(coef_block + 2 * DCTSIZE);
  int16x8_t row3 = vld1q_s16(coef_block + 3 * DCTSIZE);
  int16x8_t row4 = vld1q_s16(coef_block + 4 * DCTSIZE);
  int16x8_t row5 = vld1q_s16(coef_block + 5 * DCTSIZE);
  int16x8_t row6 = vld1q_s16(coef_block + 6 * DCTSIZE);
  int16x8_t row7 = vld1q_s16(coef_block + 7 * DCTSIZE);

  int16x8_t quant_row0 = vld1q_s16(quantptr + 0 * DCTSIZE);

  row0 = vmulq_s16(row0, quant_row0);

  int16x8_t bitmap = vorrq_s16(row1, row2);
  bitmap = vorrq_s16(bitmap, row3);
  bitmap = vorrq_s16(bitmap, row4);
  bitmap = vorrq_s16(bitmap, row5);
  bitmap = vorrq_s16(bitmap, row6);
  bitmap = vorrq_s16(bitmap, row7);

  int64_t left_ac_bitmap = vgetq_lane_s64(vreinterpretq_s64_s16(bitmap), 0);
  int64_t right_ac_bitmap = vgetq_lane_s64(vreinterpretq_s64_s16(bitmap), 1);

  const int16x4_t consts = vld1_s16(jsimd_idct_ifast_neon_consts);

  if (left_ac_bitmap == 0 && right_ac_bitmap == 0) {

    int16x8_t dcval = row0;
    row1 = dcval;
    row2 = dcval;
    row3 = dcval;
    row4 = dcval;
    row5 = dcval;
    row6 = dcval;
    row7 = dcval;
  } else if (left_ac_bitmap == 0) {

    int16x4_t dcval = vget_low_s16(row0);

    int16x4_t quant_row1 = vld1_s16(quantptr + 1 * DCTSIZE + 4);
    int16x4_t quant_row2 = vld1_s16(quantptr + 2 * DCTSIZE + 4);
    int16x4_t quant_row3 = vld1_s16(quantptr + 3 * DCTSIZE + 4);
    int16x4_t quant_row4 = vld1_s16(quantptr + 4 * DCTSIZE + 4);
    int16x4_t quant_row5 = vld1_s16(quantptr + 5 * DCTSIZE + 4);
    int16x4_t quant_row6 = vld1_s16(quantptr + 6 * DCTSIZE + 4);
    int16x4_t quant_row7 = vld1_s16(quantptr + 7 * DCTSIZE + 4);

    int16x4_t tmp0 = vget_high_s16(row0);
    int16x4_t tmp1 = vmul_s16(vget_high_s16(row2), quant_row2);
    int16x4_t tmp2 = vmul_s16(vget_high_s16(row4), quant_row4);
    int16x4_t tmp3 = vmul_s16(vget_high_s16(row6), quant_row6);

    int16x4_t tmp10 = vadd_s16(tmp0, tmp2);
    int16x4_t tmp11 = vsub_s16(tmp0, tmp2);

    int16x4_t tmp13 = vadd_s16(tmp1, tmp3);
    int16x4_t tmp1_sub_tmp3 = vsub_s16(tmp1, tmp3);
    int16x4_t tmp12 = vqdmulh_lane_s16(tmp1_sub_tmp3, consts, 1);
    tmp12 = vadd_s16(tmp12, tmp1_sub_tmp3);
    tmp12 = vsub_s16(tmp12, tmp13);

    tmp0 = vadd_s16(tmp10, tmp13);
    tmp3 = vsub_s16(tmp10, tmp13);
    tmp1 = vadd_s16(tmp11, tmp12);
    tmp2 = vsub_s16(tmp11, tmp12);

    int16x4_t tmp4 = vmul_s16(vget_high_s16(row1), quant_row1);
    int16x4_t tmp5 = vmul_s16(vget_high_s16(row3), quant_row3);
    int16x4_t tmp6 = vmul_s16(vget_high_s16(row5), quant_row5);
    int16x4_t tmp7 = vmul_s16(vget_high_s16(row7), quant_row7);

    int16x4_t z13 = vadd_s16(tmp6, tmp5);
    int16x4_t neg_z10 = vsub_s16(tmp5, tmp6);
    int16x4_t z11 = vadd_s16(tmp4, tmp7);
    int16x4_t z12 = vsub_s16(tmp4, tmp7);

    tmp7 = vadd_s16(z11, z13);
    int16x4_t z11_sub_z13 = vsub_s16(z11, z13);
    tmp11 = vqdmulh_lane_s16(z11_sub_z13, consts, 1);
    tmp11 = vadd_s16(tmp11, z11_sub_z13);

    int16x4_t z10_add_z12 = vsub_s16(z12, neg_z10);
    int16x4_t z5 = vqdmulh_lane_s16(z10_add_z12, consts, 2);
    z5 = vadd_s16(z5, z10_add_z12);
    tmp10 = vqdmulh_lane_s16(z12, consts, 0);
    tmp10 = vadd_s16(tmp10, z12);
    tmp10 = vsub_s16(tmp10, z5);
    tmp12 = vqdmulh_lane_s16(neg_z10, consts, 3);
    tmp12 = vadd_s16(tmp12, vadd_s16(neg_z10, neg_z10));
    tmp12 = vadd_s16(tmp12, z5);

    tmp6 = vsub_s16(tmp12, tmp7);
    tmp5 = vsub_s16(tmp11, tmp6);
    tmp4 = vadd_s16(tmp10, tmp5);

    row0 = vcombine_s16(dcval, vadd_s16(tmp0, tmp7));
    row7 = vcombine_s16(dcval, vsub_s16(tmp0, tmp7));
    row1 = vcombine_s16(dcval, vadd_s16(tmp1, tmp6));
    row6 = vcombine_s16(dcval, vsub_s16(tmp1, tmp6));
    row2 = vcombine_s16(dcval, vadd_s16(tmp2, tmp5));
    row5 = vcombine_s16(dcval, vsub_s16(tmp2, tmp5));
    row4 = vcombine_s16(dcval, vadd_s16(tmp3, tmp4));
    row3 = vcombine_s16(dcval, vsub_s16(tmp3, tmp4));
  } else if (right_ac_bitmap == 0) {

    int16x4_t dcval = vget_high_s16(row0);

    int16x4_t quant_row1 = vld1_s16(quantptr + 1 * DCTSIZE);
    int16x4_t quant_row2 = vld1_s16(quantptr + 2 * DCTSIZE);
    int16x4_t quant_row3 = vld1_s16(quantptr + 3 * DCTSIZE);
    int16x4_t quant_row4 = vld1_s16(quantptr + 4 * DCTSIZE);
    int16x4_t quant_row5 = vld1_s16(quantptr + 5 * DCTSIZE);
    int16x4_t quant_row6 = vld1_s16(quantptr + 6 * DCTSIZE);
    int16x4_t quant_row7 = vld1_s16(quantptr + 7 * DCTSIZE);

    int16x4_t tmp0 = vget_low_s16(row0);
    int16x4_t tmp1 = vmul_s16(vget_low_s16(row2), quant_row2);
    int16x4_t tmp2 = vmul_s16(vget_low_s16(row4), quant_row4);
    int16x4_t tmp3 = vmul_s16(vget_low_s16(row6), quant_row6);

    int16x4_t tmp10 = vadd_s16(tmp0, tmp2);
    int16x4_t tmp11 = vsub_s16(tmp0, tmp2);

    int16x4_t tmp13 = vadd_s16(tmp1, tmp3);
    int16x4_t tmp1_sub_tmp3 = vsub_s16(tmp1, tmp3);
    int16x4_t tmp12 = vqdmulh_lane_s16(tmp1_sub_tmp3, consts, 1);
    tmp12 = vadd_s16(tmp12, tmp1_sub_tmp3);
    tmp12 = vsub_s16(tmp12, tmp13);

    tmp0 = vadd_s16(tmp10, tmp13);
    tmp3 = vsub_s16(tmp10, tmp13);
    tmp1 = vadd_s16(tmp11, tmp12);
    tmp2 = vsub_s16(tmp11, tmp12);

    int16x4_t tmp4 = vmul_s16(vget_low_s16(row1), quant_row1);
    int16x4_t tmp5 = vmul_s16(vget_low_s16(row3), quant_row3);
    int16x4_t tmp6 = vmul_s16(vget_low_s16(row5), quant_row5);
    int16x4_t tmp7 = vmul_s16(vget_low_s16(row7), quant_row7);

    int16x4_t z13 = vadd_s16(tmp6, tmp5);
    int16x4_t neg_z10 = vsub_s16(tmp5, tmp6);
    int16x4_t z11 = vadd_s16(tmp4, tmp7);
    int16x4_t z12 = vsub_s16(tmp4, tmp7);

    tmp7 = vadd_s16(z11, z13);
    int16x4_t z11_sub_z13 = vsub_s16(z11, z13);
    tmp11 = vqdmulh_lane_s16(z11_sub_z13, consts, 1);
    tmp11 = vadd_s16(tmp11, z11_sub_z13);

    int16x4_t z10_add_z12 = vsub_s16(z12, neg_z10);
    int16x4_t z5 = vqdmulh_lane_s16(z10_add_z12, consts, 2);
    z5 = vadd_s16(z5, z10_add_z12);
    tmp10 = vqdmulh_lane_s16(z12, consts, 0);
    tmp10 = vadd_s16(tmp10, z12);
    tmp10 = vsub_s16(tmp10, z5);
    tmp12 = vqdmulh_lane_s16(neg_z10, consts, 3);
    tmp12 = vadd_s16(tmp12, vadd_s16(neg_z10, neg_z10));
    tmp12 = vadd_s16(tmp12, z5);

    tmp6 = vsub_s16(tmp12, tmp7);
    tmp5 = vsub_s16(tmp11, tmp6);
    tmp4 = vadd_s16(tmp10, tmp5);

    row0 = vcombine_s16(vadd_s16(tmp0, tmp7), dcval);
    row7 = vcombine_s16(vsub_s16(tmp0, tmp7), dcval);
    row1 = vcombine_s16(vadd_s16(tmp1, tmp6), dcval);
    row6 = vcombine_s16(vsub_s16(tmp1, tmp6), dcval);
    row2 = vcombine_s16(vadd_s16(tmp2, tmp5), dcval);
    row5 = vcombine_s16(vsub_s16(tmp2, tmp5), dcval);
    row4 = vcombine_s16(vadd_s16(tmp3, tmp4), dcval);
    row3 = vcombine_s16(vsub_s16(tmp3, tmp4), dcval);
  } else {

    int16x8_t quant_row1 = vld1q_s16(quantptr + 1 * DCTSIZE);
    int16x8_t quant_row2 = vld1q_s16(quantptr + 2 * DCTSIZE);
    int16x8_t quant_row3 = vld1q_s16(quantptr + 3 * DCTSIZE);
    int16x8_t quant_row4 = vld1q_s16(quantptr + 4 * DCTSIZE);
    int16x8_t quant_row5 = vld1q_s16(quantptr + 5 * DCTSIZE);
    int16x8_t quant_row6 = vld1q_s16(quantptr + 6 * DCTSIZE);
    int16x8_t quant_row7 = vld1q_s16(quantptr + 7 * DCTSIZE);

    int16x8_t tmp0 = row0;
    int16x8_t tmp1 = vmulq_s16(row2, quant_row2);
    int16x8_t tmp2 = vmulq_s16(row4, quant_row4);
    int16x8_t tmp3 = vmulq_s16(row6, quant_row6);

    int16x8_t tmp10 = vaddq_s16(tmp0, tmp2);
    int16x8_t tmp11 = vsubq_s16(tmp0, tmp2);

    int16x8_t tmp13 = vaddq_s16(tmp1, tmp3);
    int16x8_t tmp1_sub_tmp3 = vsubq_s16(tmp1, tmp3);
    int16x8_t tmp12 = vqdmulhq_lane_s16(tmp1_sub_tmp3, consts, 1);
    tmp12 = vaddq_s16(tmp12, tmp1_sub_tmp3);
    tmp12 = vsubq_s16(tmp12, tmp13);

    tmp0 = vaddq_s16(tmp10, tmp13);
    tmp3 = vsubq_s16(tmp10, tmp13);
    tmp1 = vaddq_s16(tmp11, tmp12);
    tmp2 = vsubq_s16(tmp11, tmp12);

    int16x8_t tmp4 = vmulq_s16(row1, quant_row1);
    int16x8_t tmp5 = vmulq_s16(row3, quant_row3);
    int16x8_t tmp6 = vmulq_s16(row5, quant_row5);
    int16x8_t tmp7 = vmulq_s16(row7, quant_row7);

    int16x8_t z13 = vaddq_s16(tmp6, tmp5);
    int16x8_t neg_z10 = vsubq_s16(tmp5, tmp6);
    int16x8_t z11 = vaddq_s16(tmp4, tmp7);
    int16x8_t z12 = vsubq_s16(tmp4, tmp7);

    tmp7 = vaddq_s16(z11, z13);
    int16x8_t z11_sub_z13 = vsubq_s16(z11, z13);
    tmp11 = vqdmulhq_lane_s16(z11_sub_z13, consts, 1);
    tmp11 = vaddq_s16(tmp11, z11_sub_z13);

    int16x8_t z10_add_z12 = vsubq_s16(z12, neg_z10);
    int16x8_t z5 = vqdmulhq_lane_s16(z10_add_z12, consts, 2);
    z5 = vaddq_s16(z5, z10_add_z12);
    tmp10 = vqdmulhq_lane_s16(z12, consts, 0);
    tmp10 = vaddq_s16(tmp10, z12);
    tmp10 = vsubq_s16(tmp10, z5);
    tmp12 = vqdmulhq_lane_s16(neg_z10, consts, 3);
    tmp12 = vaddq_s16(tmp12, vaddq_s16(neg_z10, neg_z10));
    tmp12 = vaddq_s16(tmp12, z5);

    tmp6 = vsubq_s16(tmp12, tmp7);
    tmp5 = vsubq_s16(tmp11, tmp6);
    tmp4 = vaddq_s16(tmp10, tmp5);

    row0 = vaddq_s16(tmp0, tmp7);
    row7 = vsubq_s16(tmp0, tmp7);
    row1 = vaddq_s16(tmp1, tmp6);
    row6 = vsubq_s16(tmp1, tmp6);
    row2 = vaddq_s16(tmp2, tmp5);
    row5 = vsubq_s16(tmp2, tmp5);
    row4 = vaddq_s16(tmp3, tmp4);
    row3 = vsubq_s16(tmp3, tmp4);
  }

  int16x8x2_t rows_01 = vtrnq_s16(row0, row1);
  int16x8x2_t rows_23 = vtrnq_s16(row2, row3);
  int16x8x2_t rows_45 = vtrnq_s16(row4, row5);
  int16x8x2_t rows_67 = vtrnq_s16(row6, row7);

  int32x4x2_t rows_0145_l = vtrnq_s32(vreinterpretq_s32_s16(rows_01.val[0]),
                                      vreinterpretq_s32_s16(rows_45.val[0]));
  int32x4x2_t rows_0145_h = vtrnq_s32(vreinterpretq_s32_s16(rows_01.val[1]),
                                      vreinterpretq_s32_s16(rows_45.val[1]));
  int32x4x2_t rows_2367_l = vtrnq_s32(vreinterpretq_s32_s16(rows_23.val[0]),
                                      vreinterpretq_s32_s16(rows_67.val[0]));
  int32x4x2_t rows_2367_h = vtrnq_s32(vreinterpretq_s32_s16(rows_23.val[1]),
                                      vreinterpretq_s32_s16(rows_67.val[1]));

  int32x4x2_t cols_04 = vzipq_s32(rows_0145_l.val[0], rows_2367_l.val[0]);
  int32x4x2_t cols_15 = vzipq_s32(rows_0145_h.val[0], rows_2367_h.val[0]);
  int32x4x2_t cols_26 = vzipq_s32(rows_0145_l.val[1], rows_2367_l.val[1]);
  int32x4x2_t cols_37 = vzipq_s32(rows_0145_h.val[1], rows_2367_h.val[1]);

  int16x8_t col0 = vreinterpretq_s16_s32(cols_04.val[0]);
  int16x8_t col1 = vreinterpretq_s16_s32(cols_15.val[0]);
  int16x8_t col2 = vreinterpretq_s16_s32(cols_26.val[0]);
  int16x8_t col3 = vreinterpretq_s16_s32(cols_37.val[0]);
  int16x8_t col4 = vreinterpretq_s16_s32(cols_04.val[1]);
  int16x8_t col5 = vreinterpretq_s16_s32(cols_15.val[1]);
  int16x8_t col6 = vreinterpretq_s16_s32(cols_26.val[1]);
  int16x8_t col7 = vreinterpretq_s16_s32(cols_37.val[1]);

  int16x8_t tmp10 = vaddq_s16(col0, col4);
  int16x8_t tmp11 = vsubq_s16(col0, col4);

  int16x8_t tmp13 = vaddq_s16(col2, col6);
  int16x8_t col2_sub_col6 = vsubq_s16(col2, col6);
  int16x8_t tmp12 = vqdmulhq_lane_s16(col2_sub_col6, consts, 1);
  tmp12 = vaddq_s16(tmp12, col2_sub_col6);
  tmp12 = vsubq_s16(tmp12, tmp13);

  int16x8_t tmp0 = vaddq_s16(tmp10, tmp13);
  int16x8_t tmp3 = vsubq_s16(tmp10, tmp13);
  int16x8_t tmp1 = vaddq_s16(tmp11, tmp12);
  int16x8_t tmp2 = vsubq_s16(tmp11, tmp12);

  int16x8_t z13 = vaddq_s16(col5, col3);
  int16x8_t neg_z10 = vsubq_s16(col3, col5);
  int16x8_t z11 = vaddq_s16(col1, col7);
  int16x8_t z12 = vsubq_s16(col1, col7);

  int16x8_t tmp7 = vaddq_s16(z11, z13);
  int16x8_t z11_sub_z13 = vsubq_s16(z11, z13);
  tmp11 = vqdmulhq_lane_s16(z11_sub_z13, consts, 1);
  tmp11 = vaddq_s16(tmp11, z11_sub_z13);

  int16x8_t z10_add_z12 = vsubq_s16(z12, neg_z10);
  int16x8_t z5 = vqdmulhq_lane_s16(z10_add_z12, consts, 2);
  z5 = vaddq_s16(z5, z10_add_z12);
  tmp10 = vqdmulhq_lane_s16(z12, consts, 0);
  tmp10 = vaddq_s16(tmp10, z12);
  tmp10 = vsubq_s16(tmp10, z5);
  tmp12 = vqdmulhq_lane_s16(neg_z10, consts, 3);
  tmp12 = vaddq_s16(tmp12, vaddq_s16(neg_z10, neg_z10));
  tmp12 = vaddq_s16(tmp12, z5);

  int16x8_t tmp6 = vsubq_s16(tmp12, tmp7);
  int16x8_t tmp5 = vsubq_s16(tmp11, tmp6);
  int16x8_t tmp4 = vaddq_s16(tmp10, tmp5);

  col0 = vaddq_s16(tmp0, tmp7);
  col7 = vsubq_s16(tmp0, tmp7);
  col1 = vaddq_s16(tmp1, tmp6);
  col6 = vsubq_s16(tmp1, tmp6);
  col2 = vaddq_s16(tmp2, tmp5);
  col5 = vsubq_s16(tmp2, tmp5);
  col4 = vaddq_s16(tmp3, tmp4);
  col3 = vsubq_s16(tmp3, tmp4);

  int8x16_t cols_01_s8 = vcombine_s8(vqshrn_n_s16(col0, PASS1_BITS + 3),
                                     vqshrn_n_s16(col1, PASS1_BITS + 3));
  int8x16_t cols_45_s8 = vcombine_s8(vqshrn_n_s16(col4, PASS1_BITS + 3),
                                     vqshrn_n_s16(col5, PASS1_BITS + 3));
  int8x16_t cols_23_s8 = vcombine_s8(vqshrn_n_s16(col2, PASS1_BITS + 3),
                                     vqshrn_n_s16(col3, PASS1_BITS + 3));
  int8x16_t cols_67_s8 = vcombine_s8(vqshrn_n_s16(col6, PASS1_BITS + 3),
                                     vqshrn_n_s16(col7, PASS1_BITS + 3));

  uint8x16_t cols_01 =
    vreinterpretq_u8_s8
      (vaddq_s8(cols_01_s8, vreinterpretq_s8_u8(vdupq_n_u8(CENTERJSAMPLE))));
  uint8x16_t cols_45 =
    vreinterpretq_u8_s8
      (vaddq_s8(cols_45_s8, vreinterpretq_s8_u8(vdupq_n_u8(CENTERJSAMPLE))));
  uint8x16_t cols_23 =
    vreinterpretq_u8_s8
      (vaddq_s8(cols_23_s8, vreinterpretq_s8_u8(vdupq_n_u8(CENTERJSAMPLE))));
  uint8x16_t cols_67 =
    vreinterpretq_u8_s8
      (vaddq_s8(cols_67_s8, vreinterpretq_s8_u8(vdupq_n_u8(CENTERJSAMPLE))));

  uint32x4x2_t cols_0415 = vzipq_u32(vreinterpretq_u32_u8(cols_01),
                                     vreinterpretq_u32_u8(cols_45));
  uint32x4x2_t cols_2637 = vzipq_u32(vreinterpretq_u32_u8(cols_23),
                                     vreinterpretq_u32_u8(cols_67));

  uint8x16x2_t cols_0145 = vtrnq_u8(vreinterpretq_u8_u32(cols_0415.val[0]),
                                    vreinterpretq_u8_u32(cols_0415.val[1]));
  uint8x16x2_t cols_2367 = vtrnq_u8(vreinterpretq_u8_u32(cols_2637.val[0]),
                                    vreinterpretq_u8_u32(cols_2637.val[1]));
  uint16x8x2_t rows_0426 = vtrnq_u16(vreinterpretq_u16_u8(cols_0145.val[0]),
                                     vreinterpretq_u16_u8(cols_2367.val[0]));
  uint16x8x2_t rows_1537 = vtrnq_u16(vreinterpretq_u16_u8(cols_0145.val[1]),
                                     vreinterpretq_u16_u8(cols_2367.val[1]));

  uint8x16_t rows_04 = vreinterpretq_u8_u16(rows_0426.val[0]);
  uint8x16_t rows_15 = vreinterpretq_u8_u16(rows_1537.val[0]);
  uint8x16_t rows_26 = vreinterpretq_u8_u16(rows_0426.val[1]);
  uint8x16_t rows_37 = vreinterpretq_u8_u16(rows_1537.val[1]);

  JSAMPROW outptr0 = output_buf[0] + output_col;
  JSAMPROW outptr1 = output_buf[1] + output_col;
  JSAMPROW outptr2 = output_buf[2] + output_col;
  JSAMPROW outptr3 = output_buf[3] + output_col;
  JSAMPROW outptr4 = output_buf[4] + output_col;
  JSAMPROW outptr5 = output_buf[5] + output_col;
  JSAMPROW outptr6 = output_buf[6] + output_col;
  JSAMPROW outptr7 = output_buf[7] + output_col;

  vst1q_lane_u64((uint64_t *)outptr0, vreinterpretq_u64_u8(rows_04), 0);
  vst1q_lane_u64((uint64_t *)outptr1, vreinterpretq_u64_u8(rows_15), 0);
  vst1q_lane_u64((uint64_t *)outptr2, vreinterpretq_u64_u8(rows_26), 0);
  vst1q_lane_u64((uint64_t *)outptr3, vreinterpretq_u64_u8(rows_37), 0);
  vst1q_lane_u64((uint64_t *)outptr4, vreinterpretq_u64_u8(rows_04), 1);
  vst1q_lane_u64((uint64_t *)outptr5, vreinterpretq_u64_u8(rows_15), 1);
  vst1q_lane_u64((uint64_t *)outptr6, vreinterpretq_u64_u8(rows_26), 1);
  vst1q_lane_u64((uint64_t *)outptr7, vreinterpretq_u64_u8(rows_37), 1);
}
