#include <arm_neon.h>

#include <stdint.h>
#include <cstring>
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

#define F_0_298  19595
#define F_0_587  38470
#define F_0_113  7471
#define F_0_168  11059
#define F_0_331  21709
#define F_0_500  32768
#define F_0_418  27439
#define F_0_081  5329

void rgb_gray_convert(JDIMENSION image_width, JSAMPARRAY input_buf,
                            JSAMPIMAGE output_buf, JDIMENSION output_row,
                            int num_rows)
{
  JSAMPROW inptr;
  JSAMPROW outptr;

  ALIGN(16) uint8_t tmp_buf[16 * RGB_PIXELSIZE];

  while (--num_rows >= 0) {
    inptr = *input_buf++;
    outptr = output_buf[0][output_row];
    output_row++;

    int cols_remaining = image_width;
    for (; cols_remaining > 0; cols_remaining -= 16) {
      if (cols_remaining < 16) {
        memcpy(tmp_buf, inptr, cols_remaining * RGB_PIXELSIZE);
        inptr = tmp_buf;
      }

      uint8x16x3_t input_pixels = vld3q_u8(inptr);
      uint16x8_t r_l = vmovl_u8(vget_low_u8(input_pixels.val[RGB_RED]));
      uint16x8_t r_h = vmovl_u8(vget_high_u8(input_pixels.val[RGB_RED]));
      uint16x8_t g_l = vmovl_u8(vget_low_u8(input_pixels.val[RGB_GREEN]));
      uint16x8_t g_h = vmovl_u8(vget_high_u8(input_pixels.val[RGB_GREEN]));
      uint16x8_t b_l = vmovl_u8(vget_low_u8(input_pixels.val[RGB_BLUE]));
      uint16x8_t b_h = vmovl_u8(vget_high_u8(input_pixels.val[RGB_BLUE]));

      uint32x4_t y_ll = vmull_n_u16(vget_low_u16(r_l), F_0_298);
      uint32x4_t y_lh = vmull_n_u16(vget_high_u16(r_l), F_0_298);
      uint32x4_t y_hl = vmull_n_u16(vget_low_u16(r_h), F_0_298);
      uint32x4_t y_hh = vmull_n_u16(vget_high_u16(r_h), F_0_298);
      y_ll = vmlal_n_u16(y_ll, vget_low_u16(g_l), F_0_587);
      y_lh = vmlal_n_u16(y_lh, vget_high_u16(g_l), F_0_587);
      y_hl = vmlal_n_u16(y_hl, vget_low_u16(g_h), F_0_587);
      y_hh = vmlal_n_u16(y_hh, vget_high_u16(g_h), F_0_587);
      y_ll = vmlal_n_u16(y_ll, vget_low_u16(b_l), F_0_113);
      y_lh = vmlal_n_u16(y_lh, vget_high_u16(b_l), F_0_113);
      y_hl = vmlal_n_u16(y_hl, vget_low_u16(b_h), F_0_113);
      y_hh = vmlal_n_u16(y_hh, vget_high_u16(b_h), F_0_113);

      uint16x8_t y_l = vcombine_u16(vrshrn_n_u32(y_ll, 16),
                                    vrshrn_n_u32(y_lh, 16));
      uint16x8_t y_h = vcombine_u16(vrshrn_n_u32(y_hl, 16),
                                    vrshrn_n_u32(y_hh, 16));

      vst1q_u8(outptr, vcombine_u8(vmovn_u16(y_l), vmovn_u16(y_h)));

      inptr += (16 * RGB_PIXELSIZE);
      outptr += 16;
    }
  }
}
