#include <riscv_vector.h>

#include <cstring>
#include <stdlib.h>
#include <stdint.h>

typedef unsigned int JDIMENSION;
typedef unsigned char JSAMPLE;
typedef JSAMPLE *JSAMPROW;
typedef JSAMPROW *JSAMPARRAY;
typedef JSAMPARRAY *JSAMPIMAGE;

#define RGB_RED  0
#define RGB_GREEN  1
#define RGB_BLUE  2
#define RGB_PIXELSIZE 3

#define DCTSIZE             8

static void expand_right_edge(JSAMPARRAY image_data, int num_rows, JDIMENSION input_cols,
                  JDIMENSION output_cols)
{
  JSAMPROW ptr;
  JSAMPLE pixval;
  int count;
  int row;
  int numcols = (int)(output_cols - input_cols);

  if (numcols > 0) {
    for (row = 0; row < num_rows; row++) {
      ptr = image_data[row] + input_cols;
      pixval = ptr[-1];
      for (count = numcols; count > 0; count--)
        *ptr++ = pixval;
    }
  }
}

void h2v1_downsample(JDIMENSION image_width,
                               int max_v_samp_factor,
                               JDIMENSION v_samp_factor,
                               JDIMENSION width_in_blocks,
                               JSAMPARRAY input_data,
                               JSAMPARRAY output_data)
{
  int outrow, outcol, i, zr_one = 0;
  JDIMENSION output_cols = width_in_blocks * DCTSIZE;
  JSAMPROW inptr, outptr;

  vuint8m2_t this_, adjct, out;

  vuint16m4_t this_w, adjct_w, bias_w, out_w;

  size_t vl = __riscv_vsetvl_e16m4(output_cols * 2);
  uint16_t *bias = NULL;
  if (NULL == (bias = (uint16_t *)malloc(vl * sizeof(uint16_t))))
  {
    return;
  }
  for (i = 0; i < vl; i++)
  {
    bias[i] = zr_one;
    zr_one ^= 1;
  }

  expand_right_edge(input_data, max_v_samp_factor, image_width,
                    output_cols * 2);

  for (outrow = 0; outrow < v_samp_factor; outrow++)
  {
    outptr = output_data[outrow];
    inptr = input_data[outrow];

    for (outcol = output_cols; outcol > 0;
         outcol -= vl, inptr += vl * 2, outptr += vl)
    {
      vl = __riscv_vsetvl_e16m4(outcol);

      vuint8m2x2_t pixels = __riscv_vlsseg2e8_v_u8m2x2(inptr, 2 * sizeof(JSAMPLE), vl);
      this_ = __riscv_vget_v_u8m2x2_u8m2(pixels, 0);
      adjct = __riscv_vget_v_u8m2x2_u8m2(pixels, 1);

      bias_w = __riscv_vle16_v_u16m4(bias, vl);

      out_w = __riscv_vwaddu_wv_u16m4(bias_w, this_, vl);
      out_w = __riscv_vwaddu_wv_u16m4(out_w, adjct, vl);

      out = __riscv_vnsrl_wx_u8m2(out_w, 1, vl);
      __riscv_vse8_v_u8m2(outptr, out, vl);
    }
  }

  free(bias);
}

void h2v2_downsample(JDIMENSION image_width, int max_v_samp_factor,
                               JDIMENSION v_samp_factor,
                               JDIMENSION width_in_blocks,
                               JSAMPARRAY input_data, JSAMPARRAY output_data)
{
  int inrow, outrow, outcol, i, one_two = 1;
  JDIMENSION output_cols = width_in_blocks * DCTSIZE;
  JSAMPROW inptr0, inptr1, outptr;

  vuint8m2_t this0, adjct0, this1, adjct1, out;

  vuint16m4_t this_w, adjct_w, bias_w, out_w;

  size_t vl = __riscv_vsetvl_e16m4(output_cols * 2);
  uint16_t *bias = NULL;
  if (NULL == (bias = (uint16_t *)malloc(vl * sizeof(uint16_t))))
  {

    return;
  }
  for (i = 0; i < vl; i++)
  {
    bias[i] = one_two;
    one_two ^= 3;
  }

  expand_right_edge(input_data, max_v_samp_factor, image_width,
                    output_cols * 2);

  for (inrow = 0, outrow = 0; outrow < v_samp_factor;
       inrow += 2, outrow++)
  {
    inptr0 = input_data[inrow];
    inptr1 = input_data[inrow + 1];
    outptr = output_data[outrow];

    for (outcol = output_cols; outcol > 0;
         outcol -= vl, inptr0 += vl * 2, inptr1 += vl * 2, outptr += vl)
    {
      vl = __riscv_vsetvl_e16m4(outcol);

      this0 = __riscv_vlse8_v_u8m2(inptr0, 2 * sizeof(JSAMPLE), vl);
      adjct0 = __riscv_vlse8_v_u8m2(inptr0 + 1, 2 * sizeof(JSAMPLE), vl);
      this1 = __riscv_vlse8_v_u8m2(inptr1, 2 * sizeof(JSAMPLE), vl);
      adjct1 = __riscv_vlse8_v_u8m2(inptr1 + 1, 2 * sizeof(JSAMPLE), vl);

      adjct_w = __riscv_vzext_vf2_u16m4(adjct0, vl);
      bias_w = __riscv_vle16_v_u16m4(bias, vl);

      out_w = __riscv_vwaddu_wv_u16m4(adjct_w, this0, vl);
      out_w = __riscv_vadd_vv_u16m4(out_w, bias_w, vl);

      out_w = __riscv_vwaddu_wv_u16m4(out_w, this1, vl);
      out_w = __riscv_vwaddu_wv_u16m4(out_w, adjct1, vl);

      out = __riscv_vnsrl_wx_u8m2(out_w, 2, vl);
      __riscv_vse8_v_u8m2(outptr, out, vl);
    }
  }

  free(bias);
}
