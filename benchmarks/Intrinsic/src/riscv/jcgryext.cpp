#include <riscv_vector.h>

typedef unsigned int JDIMENSION;
typedef unsigned char JSAMPLE;
typedef JSAMPLE *JSAMPROW;
typedef JSAMPROW *JSAMPARRAY;
typedef JSAMPARRAY *JSAMPIMAGE;

#define F_0_114  7471
#define F_0_299  19595
#define F_0_587  38470
#define SCALEBITS  16

#define RGB_RED  0
#define RGB_GREEN  1
#define RGB_BLUE  2
#define RGB_PIXELSIZE 3

void rgb_gray_convert(JDIMENSION img_width, JSAMPARRAY input_buf,
                                JSAMPIMAGE output_buf,
                                JDIMENSION output_row, int num_rows)
{
    JSAMPROW inptr, outptr;
    vuint16m4_t r, g, b, y;
    vuint32m8_t tmp;
    vuint8m2_t dest, src;
    size_t pitch = img_width * RGB_PIXELSIZE, num_cols, i;

    while (--num_rows >= 0)
    {
        inptr = *input_buf++;
        outptr = output_buf[0][output_row];
        output_row++;

        num_cols = pitch;
        while (num_cols > 0)
        {

            size_t vl = __riscv_vsetvl_e16m4(num_cols / RGB_PIXELSIZE);
            size_t cols = vl * RGB_PIXELSIZE;
            ptrdiff_t bstride = RGB_PIXELSIZE * sizeof(JSAMPLE);

            src = __riscv_vlse8_v_u8m2(inptr + RGB_RED, bstride, vl);
            r = __riscv_vzext_vf2_u16m4(src, vl);
            src = __riscv_vlse8_v_u8m2(inptr + RGB_GREEN, bstride, vl);
            g = __riscv_vzext_vf2_u16m4(src, vl);
            src = __riscv_vlse8_v_u8m2(inptr + RGB_BLUE, bstride, vl);
            b = __riscv_vzext_vf2_u16m4(src, vl);

            tmp = __riscv_vwmulu_vx_u32m8(r, F_0_299, vl);
            tmp = __riscv_vwmaccu_vx_u32m8(tmp, F_0_587, g, vl);
            tmp = __riscv_vwmaccu_vx_u32m8(tmp, F_0_114, b, vl);

            y = __riscv_vnclipu_wx_u16m4(tmp, SCALEBITS, __RISCV_VXRM_RNU, vl);
            dest = __riscv_vncvt_x_x_w_u8m2(y, vl);
            __riscv_vse8_v_u8m2(outptr, dest, vl);

            inptr += cols;
            outptr += vl;
            num_cols -= cols;
        }
    }
}