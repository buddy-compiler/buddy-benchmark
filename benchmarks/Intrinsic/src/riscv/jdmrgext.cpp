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

void h2v1_merged_upsample(JDIMENSION output_width,
                                    JSAMPIMAGE input_buf,
                                    JDIMENSION in_row_group_ctr,
                                    JSAMPARRAY output_buf)
{
    JSAMPROW outptr;

    JSAMPROW inptr0, inptr1, inptr2;
    int pitch = output_width * RGB_PIXELSIZE, width, yloop;

    vuint8m1_t src;
    vint8m1_t dest;

    size_t vl, vl_odd, cols;
    vl = __riscv_vsetvl_e16m2((output_width + 1) / 2);
    ptrdiff_t bstride;
    vuint16m2_t y0, y1, cb, cr;
    vint32m4_t tmp0;
    vint16m2_t sy0, sy1, scb, scr,
               r_sub_y, g_sub_y, b_sub_y,
               r, g, b;
    vbool8_t mask;

    inptr0 = input_buf[0][in_row_group_ctr];
    inptr1 = input_buf[1][in_row_group_ctr];
    inptr2 = input_buf[2][in_row_group_ctr];
    outptr = output_buf[0];

    for (width = output_width; width > 0;
         inptr0 += 2 * vl, inptr1 += vl, inptr2 += vl,
         outptr += cols, width -= (vl + vl_odd)) {

        vl = __riscv_vsetvl_e16m2((width + 1) >> 1);
        vl_odd = ((vl == (width + 1) >> 1) && (width & 1))? (vl - 1): vl;
        cols = (vl + vl_odd) * RGB_PIXELSIZE;
        bstride = RGB_PIXELSIZE * sizeof(JSAMPLE);

        src = __riscv_vlse8_v_u8m1(inptr0, 2 * sizeof(JSAMPLE), vl);
        y0 = __riscv_vwaddu_vx_u16m2(src, 0, vl);

        src = __riscv_vlse8_v_u8m1(inptr0 + 1, 2 * sizeof(JSAMPLE), vl_odd);
        y1 = __riscv_vwaddu_vx_u16m2(src, 0, vl_odd);
        src = __riscv_vle8_v_u8m1(inptr1, vl);
        cb = __riscv_vwaddu_vx_u16m2(src, 0, vl);
        src = __riscv_vle8_v_u8m1(inptr2, vl);
        cr = __riscv_vwaddu_vx_u16m2(src, 0, vl);

        sy0 = __riscv_vreinterpret_v_u16m2_i16m2(y0);
        sy1 = __riscv_vreinterpret_v_u16m2_i16m2(y1);
        scb = __riscv_vreinterpret_v_u16m2_i16m2(cb);
        scr = __riscv_vreinterpret_v_u16m2_i16m2(cr);

        scb = __riscv_vsub_vx_i16m2(scb, CENTERJSAMPLE, vl);
        scr = __riscv_vsub_vx_i16m2(scr, CENTERJSAMPLE, vl);

        tmp0 = __riscv_vwmul_vx_i32m4(scr, F_0_402, vl);
        r_sub_y = __riscv_vnclip_wx_i16m2(tmp0, SCALEBITS, __RISCV_VXRM_RNU, vl);
        r_sub_y = __riscv_vadd_vv_i16m2(r_sub_y, scr, vl);

        tmp0 = __riscv_vwmul_vx_i32m4(scb, -F_0_344, vl);
        tmp0 = __riscv_vwmacc_vx_i32m4(tmp0, F_0_285, scr, vl);
        g_sub_y = __riscv_vnclip_wx_i16m2(tmp0, SCALEBITS, __RISCV_VXRM_RNU, vl);
        g_sub_y = __riscv_vsub_vv_i16m2(g_sub_y, scr, vl);

        tmp0 = __riscv_vwmul_vx_i32m4(scb, -F_0_228, vl);
        b_sub_y = __riscv_vnclip_wx_i16m2(tmp0, SCALEBITS, __RISCV_VXRM_RNU, vl);
        b_sub_y = __riscv_vadd_vv_i16m2(b_sub_y, scb, vl);
        b_sub_y = __riscv_vadd_vv_i16m2(b_sub_y, scb, vl);

        r = __riscv_vadd_vv_i16m2(r_sub_y, sy0, vl);

        r = __riscv_vmax_vx_i16m2(r, 0, vl);
        r = __riscv_vmin_vx_i16m2(r, MAXJSAMPLE, vl);

        g = __riscv_vadd_vv_i16m2(g_sub_y, sy0, vl);

        g = __riscv_vmax_vx_i16m2(g, 0, vl);
        g = __riscv_vmin_vx_i16m2(g, MAXJSAMPLE, vl);

        b = __riscv_vadd_vv_i16m2(b_sub_y, sy0, vl);

        b = __riscv_vmax_vx_i16m2(b, 0, vl);
        b = __riscv_vmin_vx_i16m2(b, MAXJSAMPLE, vl);

        dest = __riscv_vnsra_wx_i8m1(r, 0, vl);
        __riscv_vsse8_v_i8m1((signed char*)outptr + RGB_RED, 2 * bstride, dest, vl);
        dest = __riscv_vnsra_wx_i8m1(g, 0, vl);
        __riscv_vsse8_v_i8m1((signed char*)outptr + RGB_GREEN, 2 * bstride, dest, vl);
        dest = __riscv_vnsra_wx_i8m1(b, 0, vl);
        __riscv_vsse8_v_i8m1((signed char*)outptr + RGB_BLUE, 2 * bstride, dest, vl);

        r = __riscv_vadd_vv_i16m2(r_sub_y, sy1, vl_odd);

        r = __riscv_vmax_vx_i16m2(r, 0, vl);
        r = __riscv_vmin_vx_i16m2(r, MAXJSAMPLE, vl);

        g = __riscv_vadd_vv_i16m2(g_sub_y, sy1, vl_odd);

        g = __riscv_vmax_vx_i16m2(g, 0, vl);
        g = __riscv_vmin_vx_i16m2(g, MAXJSAMPLE, vl);

        b = __riscv_vadd_vv_i16m2(b_sub_y, sy1, vl_odd);

        b = __riscv_vmax_vx_i16m2(b, 0, vl);
        b = __riscv_vmin_vx_i16m2(b, MAXJSAMPLE, vl);

        dest = __riscv_vnsra_wx_i8m1(r, 0, vl_odd);
        __riscv_vsse8_v_i8m1((signed char*)outptr + RGB_PIXELSIZE + RGB_RED, 2 * bstride, dest, vl_odd);
        dest = __riscv_vnsra_wx_i8m1(g, 0, vl_odd);
        __riscv_vsse8_v_i8m1((signed char*)outptr + RGB_PIXELSIZE + RGB_GREEN, 2 * bstride, dest, vl_odd);
        dest = __riscv_vnsra_wx_i8m1(b, 0, vl_odd);
        __riscv_vsse8_v_i8m1((signed char*)outptr + RGB_PIXELSIZE + RGB_BLUE, 2 * bstride, dest, vl_odd);
    }
}

void h2v2_merged_upsample(JDIMENSION output_width,
                                    JSAMPIMAGE input_buf,
                                    JDIMENSION in_row_group_ctr,
                                    JSAMPARRAY output_buf)
{
    JSAMPROW inptr, outptr;

    inptr = input_buf[0][in_row_group_ctr];
    outptr = output_buf[0];

    input_buf[0][in_row_group_ctr] = input_buf[0][in_row_group_ctr * 2];
    h2v1_merged_upsample(output_width, input_buf, in_row_group_ctr,
                                   output_buf);

    input_buf[0][in_row_group_ctr] = input_buf[0][in_row_group_ctr * 2 + 1];
    output_buf[0] = output_buf[1];
    h2v1_merged_upsample(output_width, input_buf, in_row_group_ctr,
                                   output_buf);

    input_buf[0][in_row_group_ctr] = inptr;
    output_buf[0] = outptr;
}