#include <arm_neon.h>

#include <stdint.h>
#include <cstring>
typedef unsigned int JDIMENSION;
typedef unsigned char JSAMPLE;
typedef JSAMPLE *JSAMPROW;
typedef JSAMPROW *JSAMPARRAY;
typedef JSAMPARRAY *JSAMPIMAGE;
typedef short JCOEF;
typedef JCOEF *JCOEFPTR;

#define DCTSIZE             8
#define CENTERJSAMPLE    128
#define PASS1_BITS  2

#define ALIGN(alignment)  __attribute__((aligned(alignment)))

#define CONST_BITS  13
#define PASS1_BITS  2

#define DESCALE_P1  (CONST_BITS - PASS1_BITS)
#define DESCALE_P2  (CONST_BITS + PASS1_BITS + 3)

#define F_0_298  2446
#define F_0_390  3196
#define F_0_541  4433
#define F_0_765  6270
#define F_0_899  7373
#define F_1_175  9633
#define F_1_501  12299
#define F_1_847  15137
#define F_1_961  16069
#define F_2_053  16819
#define F_2_562  20995
#define F_3_072  25172

#define F_1_175_MINUS_1_961  (F_1_175 - F_1_961)
#define F_1_175_MINUS_0_390  (F_1_175 - F_0_390)
#define F_0_541_MINUS_1_847  (F_0_541 - F_1_847)
#define F_3_072_MINUS_2_562  (F_3_072 - F_2_562)
#define F_0_298_MINUS_0_899  (F_0_298 - F_0_899)
#define F_1_501_MINUS_0_899  (F_1_501 - F_0_899)
#define F_2_053_MINUS_2_562  (F_2_053 - F_2_562)
#define F_0_541_PLUS_0_765   (F_0_541 + F_0_765)

ALIGN(16) static const int16_t idct_islow_neon_consts[] = {
  F_0_899,             F_0_541,
  F_2_562,             F_0_298_MINUS_0_899,
  F_1_501_MINUS_0_899, F_2_053_MINUS_2_562,
  F_0_541_PLUS_0_765,  F_1_175,
  F_1_175_MINUS_0_390, F_0_541_MINUS_1_847,
  F_3_072_MINUS_2_562, F_1_175_MINUS_1_961,
  0, 0, 0, 0
};

static inline void idct_islow_pass1_regular(int16x4_t row0,
                                            int16x4_t row1,
                                            int16x4_t row2,
                                            int16x4_t row3,
                                            int16x4_t row4,
                                            int16x4_t row5,
                                            int16x4_t row6,
                                            int16x4_t row7,
                                            int16x4_t quant_row0,
                                            int16x4_t quant_row1,
                                            int16x4_t quant_row2,
                                            int16x4_t quant_row3,
                                            int16x4_t quant_row4,
                                            int16x4_t quant_row5,
                                            int16x4_t quant_row6,
                                            int16x4_t quant_row7,
                                            int16_t *workspace_1,
                                            int16_t *workspace_2);

static inline void idct_islow_pass1_sparse(int16x4_t row0,
                                          int16x4_t row1,
                                          int16x4_t row2,
                                          int16x4_t row3,
                                          int16x4_t quant_row0,
                                          int16x4_t quant_row1,
                                          int16x4_t quant_row2,
                                          int16x4_t quant_row3,
                                          int16_t *workspace_1,
                                                 int16_t *workspace_2);

static inline void idct_islow_pass2_regular(int16_t *workspace,
                                            JSAMPARRAY output_buf,
                                            JDIMENSION output_col,
                                            unsigned buf_offset);

static inline void idct_islow_pass2_sparse(int16_t *workspace,
                                            JSAMPARRAY output_buf,
                                            JDIMENSION output_col,
                                            unsigned buf_offset);

void idct_islow(short *quantptr, JCOEFPTR coef_block,
                      JSAMPARRAY output_buf, JDIMENSION output_col, JSAMPROW sample_range_limit)
{

  int16_t workspace_l[8 * DCTSIZE / 2];
  int16_t workspace_r[8 * DCTSIZE / 2];

  int16x4_t row0 = vld1_s16(coef_block + 0 * DCTSIZE);
  int16x4_t row1 = vld1_s16(coef_block + 1 * DCTSIZE);
  int16x4_t row2 = vld1_s16(coef_block + 2 * DCTSIZE);
  int16x4_t row3 = vld1_s16(coef_block + 3 * DCTSIZE);
  int16x4_t row4 = vld1_s16(coef_block + 4 * DCTSIZE);
  int16x4_t row5 = vld1_s16(coef_block + 5 * DCTSIZE);
  int16x4_t row6 = vld1_s16(coef_block + 6 * DCTSIZE);
  int16x4_t row7 = vld1_s16(coef_block + 7 * DCTSIZE);

  int16x4_t quant_row0 = vld1_s16(quantptr + 0 * DCTSIZE);
  int16x4_t quant_row1 = vld1_s16(quantptr + 1 * DCTSIZE);
  int16x4_t quant_row2 = vld1_s16(quantptr + 2 * DCTSIZE);
  int16x4_t quant_row3 = vld1_s16(quantptr + 3 * DCTSIZE);
  int16x4_t quant_row4 = vld1_s16(quantptr + 4 * DCTSIZE);
  int16x4_t quant_row5 = vld1_s16(quantptr + 5 * DCTSIZE);
  int16x4_t quant_row6 = vld1_s16(quantptr + 6 * DCTSIZE);
  int16x4_t quant_row7 = vld1_s16(quantptr + 7 * DCTSIZE);

  int16x4_t bitmap = vorr_s16(row7, row6);
  bitmap = vorr_s16(bitmap, row5);
  bitmap = vorr_s16(bitmap, row4);
  int64_t bitmap_rows_4567 = vget_lane_s64(vreinterpret_s64_s16(bitmap), 0);

  if (bitmap_rows_4567 == 0) {
    bitmap = vorr_s16(bitmap, row3);
    bitmap = vorr_s16(bitmap, row2);
    bitmap = vorr_s16(bitmap, row1);
    int64_t left_ac_bitmap = vget_lane_s64(vreinterpret_s64_s16(bitmap), 0);

    if (left_ac_bitmap == 0) {
      int16x4_t dcval = vshl_n_s16(vmul_s16(row0, quant_row0), PASS1_BITS);
      int16x4x4_t quadrant = { { dcval, dcval, dcval, dcval } };

      vst4_s16(workspace_l, quadrant);
      vst4_s16(workspace_r, quadrant);
    } else {
      idct_islow_pass1_sparse(row0, row1, row2, row3, quant_row0,
                                    quant_row1, quant_row2, quant_row3,
                                    workspace_l, workspace_r);
    }
  } else {
    idct_islow_pass1_regular(row0, row1, row2, row3, row4, row5,
                                   row6, row7, quant_row0, quant_row1,
                                   quant_row2, quant_row3, quant_row4,
                                   quant_row5, quant_row6, quant_row7,
                                   workspace_l, workspace_r);
  }

  row0 = vld1_s16(coef_block + 0 * DCTSIZE + 4);
  row1 = vld1_s16(coef_block + 1 * DCTSIZE + 4);
  row2 = vld1_s16(coef_block + 2 * DCTSIZE + 4);
  row3 = vld1_s16(coef_block + 3 * DCTSIZE + 4);
  row4 = vld1_s16(coef_block + 4 * DCTSIZE + 4);
  row5 = vld1_s16(coef_block + 5 * DCTSIZE + 4);
  row6 = vld1_s16(coef_block + 6 * DCTSIZE + 4);
  row7 = vld1_s16(coef_block + 7 * DCTSIZE + 4);

  quant_row0 = vld1_s16(quantptr + 0 * DCTSIZE + 4);
  quant_row1 = vld1_s16(quantptr + 1 * DCTSIZE + 4);
  quant_row2 = vld1_s16(quantptr + 2 * DCTSIZE + 4);
  quant_row3 = vld1_s16(quantptr + 3 * DCTSIZE + 4);
  quant_row4 = vld1_s16(quantptr + 4 * DCTSIZE + 4);
  quant_row5 = vld1_s16(quantptr + 5 * DCTSIZE + 4);
  quant_row6 = vld1_s16(quantptr + 6 * DCTSIZE + 4);
  quant_row7 = vld1_s16(quantptr + 7 * DCTSIZE + 4);

  bitmap = vorr_s16(row7, row6);
  bitmap = vorr_s16(bitmap, row5);
  bitmap = vorr_s16(bitmap, row4);
  bitmap_rows_4567 = vget_lane_s64(vreinterpret_s64_s16(bitmap), 0);
  bitmap = vorr_s16(bitmap, row3);
  bitmap = vorr_s16(bitmap, row2);
  bitmap = vorr_s16(bitmap, row1);
  int64_t right_ac_bitmap = vget_lane_s64(vreinterpret_s64_s16(bitmap), 0);

  int64_t right_ac_dc_bitmap = 1;

  if (right_ac_bitmap == 0) {
    bitmap = vorr_s16(bitmap, row0);
    right_ac_dc_bitmap = vget_lane_s64(vreinterpret_s64_s16(bitmap), 0);

    if (right_ac_dc_bitmap != 0) {
      int16x4_t dcval = vshl_n_s16(vmul_s16(row0, quant_row0), PASS1_BITS);
      int16x4x4_t quadrant = { { dcval, dcval, dcval, dcval } };

      vst4_s16(workspace_l + 4 * DCTSIZE / 2, quadrant);
      vst4_s16(workspace_r + 4 * DCTSIZE / 2, quadrant);
    }
  } else {
    if (bitmap_rows_4567 == 0) {
      idct_islow_pass1_sparse(row0, row1, row2, row3, quant_row0,
                                    quant_row1, quant_row2, quant_row3,
                                    workspace_l + 4 * DCTSIZE / 2,
                                    workspace_r + 4 * DCTSIZE / 2);
    } else {
      idct_islow_pass1_regular(row0, row1, row2, row3, row4, row5,
                                     row6, row7, quant_row0, quant_row1,
                                     quant_row2, quant_row3, quant_row4,
                                     quant_row5, quant_row6, quant_row7,
                                     workspace_l + 4 * DCTSIZE / 2,
                                     workspace_r + 4 * DCTSIZE / 2);
    }
  }

  if (right_ac_dc_bitmap == 0) {
    idct_islow_pass2_sparse(workspace_l, output_buf, output_col, 0);
    idct_islow_pass2_sparse(workspace_r, output_buf, output_col, 4);
  } else {
    idct_islow_pass2_regular(workspace_l, output_buf, output_col, 0);
    idct_islow_pass2_regular(workspace_r, output_buf, output_col, 4);
  }
}

static inline void idct_islow_pass1_regular(int16x4_t row0,
                                                  int16x4_t row1,
                                                  int16x4_t row2,
                                                  int16x4_t row3,
                                                  int16x4_t row4,
                                                  int16x4_t row5,
                                                  int16x4_t row6,
                                                  int16x4_t row7,
                                                  int16x4_t quant_row0,
                                                  int16x4_t quant_row1,
                                                  int16x4_t quant_row2,
                                                  int16x4_t quant_row3,
                                                  int16x4_t quant_row4,
                                                  int16x4_t quant_row5,
                                                  int16x4_t quant_row6,
                                                  int16x4_t quant_row7,
                                                  int16_t *workspace_1,
                                                  int16_t *workspace_2)
{

  const int16x4x3_t consts = vld1_s16_x3(idct_islow_neon_consts);

  int16x4_t z2_s16 = vmul_s16(row2, quant_row2);
  int16x4_t z3_s16 = vmul_s16(row6, quant_row6);

  int32x4_t tmp2 = vmull_lane_s16(z2_s16, consts.val[0], 1);
  int32x4_t tmp3 = vmull_lane_s16(z2_s16, consts.val[1], 2);
  tmp2 = vmlal_lane_s16(tmp2, z3_s16, consts.val[2], 1);
  tmp3 = vmlal_lane_s16(tmp3, z3_s16, consts.val[0], 1);

  z2_s16 = vmul_s16(row0, quant_row0);
  z3_s16 = vmul_s16(row4, quant_row4);

  int32x4_t tmp0 = vshll_n_s16(vadd_s16(z2_s16, z3_s16), CONST_BITS);
  int32x4_t tmp1 = vshll_n_s16(vsub_s16(z2_s16, z3_s16), CONST_BITS);

  int32x4_t tmp10 = vaddq_s32(tmp0, tmp3);
  int32x4_t tmp13 = vsubq_s32(tmp0, tmp3);
  int32x4_t tmp11 = vaddq_s32(tmp1, tmp2);
  int32x4_t tmp12 = vsubq_s32(tmp1, tmp2);

  int16x4_t tmp0_s16 = vmul_s16(row7, quant_row7);
  int16x4_t tmp1_s16 = vmul_s16(row5, quant_row5);
  int16x4_t tmp2_s16 = vmul_s16(row3, quant_row3);
  int16x4_t tmp3_s16 = vmul_s16(row1, quant_row1);

  z3_s16 = vadd_s16(tmp0_s16, tmp2_s16);
  int16x4_t z4_s16 = vadd_s16(tmp1_s16, tmp3_s16);

  int32x4_t z3 = vmull_lane_s16(z3_s16, consts.val[2], 3);
  int32x4_t z4 = vmull_lane_s16(z3_s16, consts.val[1], 3);
  z3 = vmlal_lane_s16(z3, z4_s16, consts.val[1], 3);
  z4 = vmlal_lane_s16(z4, z4_s16, consts.val[2], 0);

  tmp0 = vmull_lane_s16(tmp0_s16, consts.val[0], 3);
  tmp1 = vmull_lane_s16(tmp1_s16, consts.val[1], 1);
  tmp2 = vmull_lane_s16(tmp2_s16, consts.val[2], 2);
  tmp3 = vmull_lane_s16(tmp3_s16, consts.val[1], 0);

  tmp0 = vmlsl_lane_s16(tmp0, tmp3_s16, consts.val[0], 0);
  tmp1 = vmlsl_lane_s16(tmp1, tmp2_s16, consts.val[0], 2);
  tmp2 = vmlsl_lane_s16(tmp2, tmp1_s16, consts.val[0], 2);
  tmp3 = vmlsl_lane_s16(tmp3, tmp0_s16, consts.val[0], 0);

  tmp0 = vaddq_s32(tmp0, z3);
  tmp1 = vaddq_s32(tmp1, z4);
  tmp2 = vaddq_s32(tmp2, z3);
  tmp3 = vaddq_s32(tmp3, z4);

  int16x4x4_t rows_0123 = { {
    vrshrn_n_s32(vaddq_s32(tmp10, tmp3), DESCALE_P1),
    vrshrn_n_s32(vaddq_s32(tmp11, tmp2), DESCALE_P1),
    vrshrn_n_s32(vaddq_s32(tmp12, tmp1), DESCALE_P1),
    vrshrn_n_s32(vaddq_s32(tmp13, tmp0), DESCALE_P1)
  } };
  int16x4x4_t rows_4567 = { {
    vrshrn_n_s32(vsubq_s32(tmp13, tmp0), DESCALE_P1),
    vrshrn_n_s32(vsubq_s32(tmp12, tmp1), DESCALE_P1),
    vrshrn_n_s32(vsubq_s32(tmp11, tmp2), DESCALE_P1),
    vrshrn_n_s32(vsubq_s32(tmp10, tmp3), DESCALE_P1)
  } };

  vst4_s16(workspace_1, rows_0123);
  vst4_s16(workspace_2, rows_4567);
}

static inline void idct_islow_pass1_sparse(int16x4_t row0,
                                                 int16x4_t row1,
                                                 int16x4_t row2,
                                                 int16x4_t row3,
                                                 int16x4_t quant_row0,
                                                 int16x4_t quant_row1,
                                                 int16x4_t quant_row2,
                                                 int16x4_t quant_row3,
                                                 int16_t *workspace_1,
                                                 int16_t *workspace_2)
{

  const int16x4x3_t consts = vld1_s16_x3(idct_islow_neon_consts);
  int16x4_t z2_s16 = vmul_s16(row2, quant_row2);

  int32x4_t tmp2 = vmull_lane_s16(z2_s16, consts.val[0], 1);
  int32x4_t tmp3 = vmull_lane_s16(z2_s16, consts.val[1], 2);

  z2_s16 = vmul_s16(row0, quant_row0);
  int32x4_t tmp0 = vshll_n_s16(z2_s16, CONST_BITS);
  int32x4_t tmp1 = vshll_n_s16(z2_s16, CONST_BITS);

  int32x4_t tmp10 = vaddq_s32(tmp0, tmp3);
  int32x4_t tmp13 = vsubq_s32(tmp0, tmp3);
  int32x4_t tmp11 = vaddq_s32(tmp1, tmp2);
  int32x4_t tmp12 = vsubq_s32(tmp1, tmp2);

  int16x4_t tmp2_s16 = vmul_s16(row3, quant_row3);
  int16x4_t tmp3_s16 = vmul_s16(row1, quant_row1);

  int16x4_t z3_s16 = tmp2_s16;
  int16x4_t z4_s16 = tmp3_s16;

  int32x4_t z3 = vmull_lane_s16(z3_s16, consts.val[2], 3);
  int32x4_t z4 = vmull_lane_s16(z3_s16, consts.val[1], 3);
  z3 = vmlal_lane_s16(z3, z4_s16, consts.val[1], 3);
  z4 = vmlal_lane_s16(z4, z4_s16, consts.val[2], 0);

  tmp0 = vmlsl_lane_s16(z3, tmp3_s16, consts.val[0], 0);
  tmp1 = vmlsl_lane_s16(z4, tmp2_s16, consts.val[0], 2);
  tmp2 = vmlal_lane_s16(z3, tmp2_s16, consts.val[2], 2);
  tmp3 = vmlal_lane_s16(z4, tmp3_s16, consts.val[1], 0);

  int16x4x4_t rows_0123 = { {
    vrshrn_n_s32(vaddq_s32(tmp10, tmp3), DESCALE_P1),
    vrshrn_n_s32(vaddq_s32(tmp11, tmp2), DESCALE_P1),
    vrshrn_n_s32(vaddq_s32(tmp12, tmp1), DESCALE_P1),
    vrshrn_n_s32(vaddq_s32(tmp13, tmp0), DESCALE_P1)
  } };
  int16x4x4_t rows_4567 = { {
    vrshrn_n_s32(vsubq_s32(tmp13, tmp0), DESCALE_P1),
    vrshrn_n_s32(vsubq_s32(tmp12, tmp1), DESCALE_P1),
    vrshrn_n_s32(vsubq_s32(tmp11, tmp2), DESCALE_P1),
    vrshrn_n_s32(vsubq_s32(tmp10, tmp3), DESCALE_P1)
  } };

  vst4_s16(workspace_1, rows_0123);
  vst4_s16(workspace_2, rows_4567);
}

static inline void idct_islow_pass2_regular(int16_t *workspace,
                                                  JSAMPARRAY output_buf,
                                                  JDIMENSION output_col,
                                                  unsigned buf_offset)
{

  const int16x4x3_t consts = vld1_s16_x3(idct_islow_neon_consts);

  int16x4_t z2_s16 = vld1_s16(workspace + 2 * DCTSIZE / 2);
  int16x4_t z3_s16 = vld1_s16(workspace + 6 * DCTSIZE / 2);

  int32x4_t tmp2 = vmull_lane_s16(z2_s16, consts.val[0], 1);
  int32x4_t tmp3 = vmull_lane_s16(z2_s16, consts.val[1], 2);
  tmp2 = vmlal_lane_s16(tmp2, z3_s16, consts.val[2], 1);
  tmp3 = vmlal_lane_s16(tmp3, z3_s16, consts.val[0], 1);

  z2_s16 = vld1_s16(workspace + 0 * DCTSIZE / 2);
  z3_s16 = vld1_s16(workspace + 4 * DCTSIZE / 2);

  int32x4_t tmp0 = vshll_n_s16(vadd_s16(z2_s16, z3_s16), CONST_BITS);
  int32x4_t tmp1 = vshll_n_s16(vsub_s16(z2_s16, z3_s16), CONST_BITS);

  int32x4_t tmp10 = vaddq_s32(tmp0, tmp3);
  int32x4_t tmp13 = vsubq_s32(tmp0, tmp3);
  int32x4_t tmp11 = vaddq_s32(tmp1, tmp2);
  int32x4_t tmp12 = vsubq_s32(tmp1, tmp2);

  int16x4_t tmp0_s16 = vld1_s16(workspace + 7 * DCTSIZE / 2);
  int16x4_t tmp1_s16 = vld1_s16(workspace + 5 * DCTSIZE / 2);
  int16x4_t tmp2_s16 = vld1_s16(workspace + 3 * DCTSIZE / 2);
  int16x4_t tmp3_s16 = vld1_s16(workspace + 1 * DCTSIZE / 2);

  z3_s16 = vadd_s16(tmp0_s16, tmp2_s16);
  int16x4_t z4_s16 = vadd_s16(tmp1_s16, tmp3_s16);

  int32x4_t z3 = vmull_lane_s16(z3_s16, consts.val[2], 3);
  int32x4_t z4 = vmull_lane_s16(z3_s16, consts.val[1], 3);
  z3 = vmlal_lane_s16(z3, z4_s16, consts.val[1], 3);
  z4 = vmlal_lane_s16(z4, z4_s16, consts.val[2], 0);

  tmp0 = vmull_lane_s16(tmp0_s16, consts.val[0], 3);
  tmp1 = vmull_lane_s16(tmp1_s16, consts.val[1], 1);
  tmp2 = vmull_lane_s16(tmp2_s16, consts.val[2], 2);
  tmp3 = vmull_lane_s16(tmp3_s16, consts.val[1], 0);

  tmp0 = vmlsl_lane_s16(tmp0, tmp3_s16, consts.val[0], 0);
  tmp1 = vmlsl_lane_s16(tmp1, tmp2_s16, consts.val[0], 2);
  tmp2 = vmlsl_lane_s16(tmp2, tmp1_s16, consts.val[0], 2);
  tmp3 = vmlsl_lane_s16(tmp3, tmp0_s16, consts.val[0], 0);

  tmp0 = vaddq_s32(tmp0, z3);
  tmp1 = vaddq_s32(tmp1, z4);
  tmp2 = vaddq_s32(tmp2, z3);
  tmp3 = vaddq_s32(tmp3, z4);

  int16x8_t cols_02_s16 = vcombine_s16(vaddhn_s32(tmp10, tmp3),
                                       vaddhn_s32(tmp12, tmp1));
  int16x8_t cols_13_s16 = vcombine_s16(vaddhn_s32(tmp11, tmp2),
                                       vaddhn_s32(tmp13, tmp0));
  int16x8_t cols_46_s16 = vcombine_s16(vsubhn_s32(tmp13, tmp0),
                                       vsubhn_s32(tmp11, tmp2));
  int16x8_t cols_57_s16 = vcombine_s16(vsubhn_s32(tmp12, tmp1),
                                       vsubhn_s32(tmp10, tmp3));

  int8x8_t cols_02_s8 = vqrshrn_n_s16(cols_02_s16, DESCALE_P2 - 16);
  int8x8_t cols_13_s8 = vqrshrn_n_s16(cols_13_s16, DESCALE_P2 - 16);
  int8x8_t cols_46_s8 = vqrshrn_n_s16(cols_46_s16, DESCALE_P2 - 16);
  int8x8_t cols_57_s8 = vqrshrn_n_s16(cols_57_s16, DESCALE_P2 - 16);

  uint8x8_t cols_02_u8 = vadd_u8(vreinterpret_u8_s8(cols_02_s8),
                                 vdup_n_u8(CENTERJSAMPLE));
  uint8x8_t cols_13_u8 = vadd_u8(vreinterpret_u8_s8(cols_13_s8),
                                 vdup_n_u8(CENTERJSAMPLE));
  uint8x8_t cols_46_u8 = vadd_u8(vreinterpret_u8_s8(cols_46_s8),
                                 vdup_n_u8(CENTERJSAMPLE));
  uint8x8_t cols_57_u8 = vadd_u8(vreinterpret_u8_s8(cols_57_s8),
                                 vdup_n_u8(CENTERJSAMPLE));

  uint8x8x2_t cols_01_23 = vzip_u8(cols_02_u8, cols_13_u8);
  uint8x8x2_t cols_45_67 = vzip_u8(cols_46_u8, cols_57_u8);
  uint16x4x4_t cols_01_23_45_67 = { {
    vreinterpret_u16_u8(cols_01_23.val[0]),
    vreinterpret_u16_u8(cols_01_23.val[1]),
    vreinterpret_u16_u8(cols_45_67.val[0]),
    vreinterpret_u16_u8(cols_45_67.val[1])
  } };

  JSAMPROW outptr0 = output_buf[buf_offset + 0] + output_col;
  JSAMPROW outptr1 = output_buf[buf_offset + 1] + output_col;
  JSAMPROW outptr2 = output_buf[buf_offset + 2] + output_col;
  JSAMPROW outptr3 = output_buf[buf_offset + 3] + output_col;

  vst4_lane_u16((uint16_t *)outptr0, cols_01_23_45_67, 0);
  vst4_lane_u16((uint16_t *)outptr1, cols_01_23_45_67, 1);
  vst4_lane_u16((uint16_t *)outptr2, cols_01_23_45_67, 2);
  vst4_lane_u16((uint16_t *)outptr3, cols_01_23_45_67, 3);
}

static inline void idct_islow_pass2_sparse(int16_t *workspace,
                                                 JSAMPARRAY output_buf,
                                                 JDIMENSION output_col,
                                                 unsigned buf_offset)
{

  const int16x4x3_t consts = vld1_s16_x3(idct_islow_neon_consts);

  int16x4_t z2_s16 = vld1_s16(workspace + 2 * DCTSIZE / 2);

  int32x4_t tmp2 = vmull_lane_s16(z2_s16, consts.val[0], 1);
  int32x4_t tmp3 = vmull_lane_s16(z2_s16, consts.val[1], 2);

  z2_s16 = vld1_s16(workspace + 0 * DCTSIZE / 2);
  int32x4_t tmp0 = vshll_n_s16(z2_s16, CONST_BITS);
  int32x4_t tmp1 = vshll_n_s16(z2_s16, CONST_BITS);

  int32x4_t tmp10 = vaddq_s32(tmp0, tmp3);
  int32x4_t tmp13 = vsubq_s32(tmp0, tmp3);
  int32x4_t tmp11 = vaddq_s32(tmp1, tmp2);
  int32x4_t tmp12 = vsubq_s32(tmp1, tmp2);

  int16x4_t tmp2_s16 = vld1_s16(workspace + 3 * DCTSIZE / 2);
  int16x4_t tmp3_s16 = vld1_s16(workspace + 1 * DCTSIZE / 2);

  int16x4_t z3_s16 = tmp2_s16;
  int16x4_t z4_s16 = tmp3_s16;

  int32x4_t z3 = vmull_lane_s16(z3_s16, consts.val[2], 3);
  z3 = vmlal_lane_s16(z3, z4_s16, consts.val[1], 3);
  int32x4_t z4 = vmull_lane_s16(z3_s16, consts.val[1], 3);
  z4 = vmlal_lane_s16(z4, z4_s16, consts.val[2], 0);

  tmp0 = vmlsl_lane_s16(z3, tmp3_s16, consts.val[0], 0);
  tmp1 = vmlsl_lane_s16(z4, tmp2_s16, consts.val[0], 2);
  tmp2 = vmlal_lane_s16(z3, tmp2_s16, consts.val[2], 2);
  tmp3 = vmlal_lane_s16(z4, tmp3_s16, consts.val[1], 0);

  int16x8_t cols_02_s16 = vcombine_s16(vaddhn_s32(tmp10, tmp3),
                                       vaddhn_s32(tmp12, tmp1));
  int16x8_t cols_13_s16 = vcombine_s16(vaddhn_s32(tmp11, tmp2),
                                       vaddhn_s32(tmp13, tmp0));
  int16x8_t cols_46_s16 = vcombine_s16(vsubhn_s32(tmp13, tmp0),
                                       vsubhn_s32(tmp11, tmp2));
  int16x8_t cols_57_s16 = vcombine_s16(vsubhn_s32(tmp12, tmp1),
                                       vsubhn_s32(tmp10, tmp3));

  int8x8_t cols_02_s8 = vqrshrn_n_s16(cols_02_s16, DESCALE_P2 - 16);
  int8x8_t cols_13_s8 = vqrshrn_n_s16(cols_13_s16, DESCALE_P2 - 16);
  int8x8_t cols_46_s8 = vqrshrn_n_s16(cols_46_s16, DESCALE_P2 - 16);
  int8x8_t cols_57_s8 = vqrshrn_n_s16(cols_57_s16, DESCALE_P2 - 16);

  uint8x8_t cols_02_u8 = vadd_u8(vreinterpret_u8_s8(cols_02_s8),
                                 vdup_n_u8(CENTERJSAMPLE));
  uint8x8_t cols_13_u8 = vadd_u8(vreinterpret_u8_s8(cols_13_s8),
                                 vdup_n_u8(CENTERJSAMPLE));
  uint8x8_t cols_46_u8 = vadd_u8(vreinterpret_u8_s8(cols_46_s8),
                                 vdup_n_u8(CENTERJSAMPLE));
  uint8x8_t cols_57_u8 = vadd_u8(vreinterpret_u8_s8(cols_57_s8),
                                 vdup_n_u8(CENTERJSAMPLE));

  uint8x8x2_t cols_01_23 = vzip_u8(cols_02_u8, cols_13_u8);
  uint8x8x2_t cols_45_67 = vzip_u8(cols_46_u8, cols_57_u8);
  uint16x4x4_t cols_01_23_45_67 = { {
    vreinterpret_u16_u8(cols_01_23.val[0]),
    vreinterpret_u16_u8(cols_01_23.val[1]),
    vreinterpret_u16_u8(cols_45_67.val[0]),
    vreinterpret_u16_u8(cols_45_67.val[1])
  } };

  JSAMPROW outptr0 = output_buf[buf_offset + 0] + output_col;
  JSAMPROW outptr1 = output_buf[buf_offset + 1] + output_col;
  JSAMPROW outptr2 = output_buf[buf_offset + 2] + output_col;
  JSAMPROW outptr3 = output_buf[buf_offset + 3] + output_col;

  vst4_lane_u16((uint16_t *)outptr0, cols_01_23_45_67, 0);
  vst4_lane_u16((uint16_t *)outptr1, cols_01_23_45_67, 1);
  vst4_lane_u16((uint16_t *)outptr2, cols_01_23_45_67, 2);
  vst4_lane_u16((uint16_t *)outptr3, cols_01_23_45_67, 3);
}
