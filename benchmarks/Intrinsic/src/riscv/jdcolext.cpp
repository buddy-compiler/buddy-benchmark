#include <riscv_vector.h>

typedef unsigned int JDIMENSION;
typedef unsigned char JSAMPLE;
typedef JSAMPLE *JSAMPROW;
typedef JSAMPROW *JSAMPARRAY;
typedef JSAMPARRAY *JSAMPIMAGE;

#define RGB_RED  0
#define RGB_GREEN  1
#define RGB_BLUE  2
#define RGB_PIXELSIZE 3

#define MAXJSAMPLE       255
#define CENTERJSAMPLE    128
#define F_0_344  22554
#define F_0_714  46802
#define F_0_772  50594
#define F_0_228  (65536 - F_0_772)
#define F_0_402  26345
#define F_0_285 (65536 - F_0_714)

#define SCALEBITS  16
#define ONE_HALF  (1 << (SCALEBITS - 1))

void ycc_rgb_convert(JDIMENSION out_width, JSAMPIMAGE input_buf,
                               JDIMENSION input_row, JSAMPARRAY output_buf,
                               int num_rows)
{
    JSAMPROW outptr, inptr0, inptr1, inptr2;
    vuint16m4_t y, cb, cr;
    vint32m8_t tmp0, tmp1;
    vint16m4_t sy, scb, scr, r, tmp;
    vbool4_t mask;
    vuint8m2_t src;
    vint8m2_t dest;
    size_t pitch = out_width * RGB_PIXELSIZE, num_cols,
           cols, i, vl;
    vl = __riscv_vsetvl_e16m4(pitch);
    ptrdiff_t bstride;

    while (--num_rows >= 0)
    {
        inptr0 = input_buf[0][input_row];
        inptr1 = input_buf[1][input_row];
        inptr2 = input_buf[2][input_row];
        input_row++;
        outptr = *output_buf++;

        num_cols = pitch;
        while (num_cols > 0)
        {

            vl = __riscv_vsetvl_e16m4(num_cols / RGB_PIXELSIZE);
            cols = vl * RGB_PIXELSIZE;
            bstride = RGB_PIXELSIZE * sizeof(JSAMPLE);

            src = __riscv_vle8_v_u8m2(inptr0, vl);
            y = __riscv_vzext_vf2_u16m4(src, vl);
            src = __riscv_vle8_v_u8m2(inptr1, vl);
            cb = __riscv_vzext_vf2_u16m4(src, vl);
            src = __riscv_vle8_v_u8m2(inptr2, vl);
            cr = __riscv_vzext_vf2_u16m4(src, vl);

            sy = __riscv_vreinterpret_v_u16m4_i16m4(y);
            scb = __riscv_vreinterpret_v_u16m4_i16m4(cb);
            scr = __riscv_vreinterpret_v_u16m4_i16m4(cr);

            scb = __riscv_vsub_vx_i16m4(scb, CENTERJSAMPLE, vl);
            scr = __riscv_vsub_vx_i16m4(scr, CENTERJSAMPLE, vl);

            r = __riscv_vadd_vv_i16m4(sy, scr, vl);
            tmp0 = __riscv_vwmul_vx_i32m8(scr, F_0_402, vl);
            tmp0 = __riscv_vadd_vx_i32m8(tmp0, ONE_HALF, vl);
            tmp = __riscv_vnclip_wx_i16m4(tmp0, SCALEBITS, __RISCV_VXRM_RDN, vl);
            r = __riscv_vadd_vv_i16m4(r, tmp, vl);

            r = __riscv_vmax_vx_i16m4(r, 0, vl);
            r = __riscv_vmin_vx_i16m4(r, MAXJSAMPLE, vl);

            dest = __riscv_vnsra_wx_i8m2(r, 0, vl);
            __riscv_vsse8_v_i8m2((signed char*)(outptr + RGB_RED), bstride, dest, vl);

            tmp0 = __riscv_vwmul_vx_i32m8(scb, -F_0_344, vl);
            tmp0 = __riscv_vwmacc_vx_i32m8(tmp0, F_0_285, scr, vl);
            r = __riscv_vnclip_wx_i16m4(tmp0, SCALEBITS, __RISCV_VXRM_RNU, vl);
            r = __riscv_vadd_vv_i16m4(r, sy, vl);
            r = __riscv_vsub_vv_i16m4(r, scr, vl);

            r = __riscv_vmax_vx_i16m4(r, 0, vl);
            r = __riscv_vmin_vx_i16m4(r, MAXJSAMPLE, vl);

            dest = __riscv_vnsra_wx_i8m2(r, 0, vl);
            __riscv_vsse8_v_i8m2((signed char*)(outptr + RGB_GREEN), bstride, dest, vl);

            r = __riscv_vadd_vv_i16m4(sy, scb, vl);
            r = __riscv_vadd_vv_i16m4(r, scb, vl);
            tmp0 = __riscv_vwmul_vx_i32m8(scb, -F_0_228, vl);
            tmp = __riscv_vnclip_wx_i16m4(tmp0, SCALEBITS, __RISCV_VXRM_RNU, vl);
            r = __riscv_vadd_vv_i16m4(r, tmp, vl);

            r = __riscv_vmax_vx_i16m4(r, 0, vl);
            r = __riscv_vmin_vx_i16m4(r, MAXJSAMPLE, vl);

            dest = __riscv_vnsra_wx_i8m2(r, 0, vl);
            __riscv_vsse8_v_i8m2((signed char*)(outptr + RGB_BLUE), bstride, dest, vl);

            outptr += cols;
            inptr0 += vl;   inptr1 += vl;   inptr2 += vl;
            num_cols -= cols;
        }
    }
}