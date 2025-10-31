#include <riscv_vector.h>

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

#define F_0_081  5329
#define F_0_114  7471
#define F_0_168  11059
#define F_0_250  16384
#define F_0_299  19595
#define F_0_331  21709
#define F_0_418  27439
#define F_0_500  32768
#define F_0_587  38470

#define SCALEBITS  16
#define CENTERJSAMPLE   128
#define ONE_HALF  (1 << (SCALEBITS - 1))
#define SCALED_CENTERJSAMPLE (CENTERJSAMPLE << SCALEBITS)

void rgb_ycc_convert(JDIMENSION img_width, JSAMPARRAY input_buf,
                               JSAMPIMAGE output_buf,
                               JDIMENSION output_row, int num_rows)
{
    JSAMPROW inptr, outptr0, outptr1, outptr2;
    vuint16m4_t r, g, b, y;
    vuint32m8_t tmp0, tmp1;
    vuint8m2_t dest, src;
    size_t pitch = img_width * RGB_PIXELSIZE, num_cols, i;

    while (--num_rows >= 0)
    {
        inptr = *input_buf++;
        outptr0 = output_buf[0][output_row];
        outptr1 = output_buf[1][output_row];
        outptr2 = output_buf[2][output_row];
        output_row++;

        num_cols = pitch;
        while (num_cols > 0)
        {

            size_t vl =__riscv_vsetvl_e16m4(num_cols / RGB_PIXELSIZE);
            size_t cols = vl * RGB_PIXELSIZE;
            ptrdiff_t bstride = RGB_PIXELSIZE * sizeof(JSAMPLE);

            src =__riscv_vlse8_v_u8m2(inptr + RGB_RED, bstride, vl);
            r = __riscv_vzext_vf2_u16m4(src, vl);
            src = __riscv_vlse8_v_u8m2(inptr + RGB_GREEN, bstride, vl);
            g = __riscv_vzext_vf2_u16m4(src, vl);
            src = __riscv_vlse8_v_u8m2(inptr + RGB_BLUE, bstride, vl);
            b = __riscv_vzext_vf2_u16m4(src, vl);

            tmp0 = __riscv_vwmulu_vx_u32m8(r, F_0_299, vl);
            tmp0 = __riscv_vwmaccu_vx_u32m8(tmp0, F_0_587, g, vl);
            tmp0 = __riscv_vwmaccu_vx_u32m8(tmp0, F_0_114, b, vl);

            y = __riscv_vnclipu_wx_u16m4(tmp0, SCALEBITS, __RISCV_VXRM_RNU, vl);

            dest = __riscv_vncvt_x_x_w_u8m2(y, vl);
            __riscv_vse8_v_u8m2(outptr0, dest, vl);

            tmp0 = __riscv_vwmulu_vx_u32m8(b, F_0_500, vl);
            tmp0 = __riscv_vadd_vx_u32m8(tmp0, SCALED_CENTERJSAMPLE, vl);
            tmp1 = __riscv_vwmulu_vx_u32m8(g, F_0_331, vl);
            tmp0 = __riscv_vsub_vv_u32m8(tmp0, tmp1, vl);
            tmp1 = __riscv_vwmulu_vx_u32m8(r, F_0_168, vl);
            tmp0 = __riscv_vsub_vv_u32m8(tmp0, tmp1, vl);

            tmp0 = __riscv_vadd_vx_u32m8(tmp0, ONE_HALF - 1, vl);
            y = __riscv_vnsrl_wx_u16m4(tmp0, SCALEBITS, vl);

            dest = __riscv_vncvt_x_x_w_u8m2(y, vl);
            __riscv_vse8_v_u8m2(outptr1, dest, vl);

            tmp0 = __riscv_vwmulu_vx_u32m8(r, F_0_500, vl);
            tmp0 = __riscv_vadd_vx_u32m8(tmp0, SCALED_CENTERJSAMPLE, vl);
            tmp1 = __riscv_vwmulu_vx_u32m8(g, F_0_418, vl);
            tmp0 = __riscv_vsub_vv_u32m8(tmp0, tmp1, vl);
            tmp1 = __riscv_vwmulu_vx_u32m8(b, F_0_081, vl);
            tmp0 = __riscv_vsub_vv_u32m8(tmp0, tmp1, vl);

            tmp0 = __riscv_vadd_vx_u32m8(tmp0, ONE_HALF - 1, vl);
            y = __riscv_vnsrl_wx_u16m4(tmp0, SCALEBITS, vl);

            dest = __riscv_vncvt_x_x_w_u8m2(y, vl);
            __riscv_vse8_v_u8m2(outptr2, dest, vl);

            inptr += cols;
            outptr0 += vl;
            outptr1 += vl;
            outptr2 += vl;
            num_cols -= cols;
        }
    }
}