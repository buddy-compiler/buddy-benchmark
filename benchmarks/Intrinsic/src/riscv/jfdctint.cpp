#include <riscv_vector.h>

typedef unsigned char JSAMPLE;

typedef JSAMPLE *JSAMPROW;
typedef JSAMPROW *JSAMPARRAY;
typedef JSAMPARRAY *JSAMPIMAGE;

#define DCTSIZE             8
typedef short DCTELEM;

#define CONST_BITS  13
#define PASS1_BITS  2

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

#define ROUND_ADD(n)    (int32_t)1 << ((n) - 1)

void fdct_islow(DCTELEM *data)
{
    vint16m1_t row0, row1, row2, row3, row4, row5, row6, row7,
               col0, col1, col2, col3, col4, col5, col6, col7,
               tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7,
               tmp10, tmp11, tmp12, tmp13,
               z1, z2, z3, z4, z5, z11, z13,
               out0, out1, out2, out3, out4, out5, out6, out7;
    vint32m2_t p1, p2, p3, p4, p5, t4, t5, t6, t7, temp;
    size_t vl = __riscv_vsetvl_e16m1(DCTSIZE),
           col_stride = DCTSIZE * sizeof(DCTELEM);

    col0 = __riscv_vlse16_v_i16m1(data + 0, col_stride, vl);
    col1 = __riscv_vlse16_v_i16m1(data + 1, col_stride, vl);
    col2 = __riscv_vlse16_v_i16m1(data + 2, col_stride, vl);
    col3 = __riscv_vlse16_v_i16m1(data + 3, col_stride, vl);
    col4 = __riscv_vlse16_v_i16m1(data + 4, col_stride, vl);
    col5 = __riscv_vlse16_v_i16m1(data + 5, col_stride, vl);
    col6 = __riscv_vlse16_v_i16m1(data + 6, col_stride, vl);
    col7 = __riscv_vlse16_v_i16m1(data + 7, col_stride, vl);

    tmp0 = __riscv_vadd_vv_i16m1(col0, col7, vl);
    tmp7 = __riscv_vsub_vv_i16m1(col0, col7, vl);
    tmp1 = __riscv_vadd_vv_i16m1(col1, col6, vl);
    tmp6 = __riscv_vsub_vv_i16m1(col1, col6, vl);
    tmp2 = __riscv_vadd_vv_i16m1(col2, col5, vl);
    tmp5 = __riscv_vsub_vv_i16m1(col2, col5, vl);
    tmp3 = __riscv_vadd_vv_i16m1(col3, col4, vl);
    tmp4 = __riscv_vsub_vv_i16m1(col3, col4, vl);

    tmp10 = __riscv_vadd_vv_i16m1(tmp0, tmp3, vl);
    tmp13 = __riscv_vsub_vv_i16m1(tmp0, tmp3, vl);
    tmp11 = __riscv_vadd_vv_i16m1(tmp1, tmp2, vl);
    tmp12 = __riscv_vsub_vv_i16m1(tmp1, tmp2, vl);

    out0 = __riscv_vadd_vv_i16m1(tmp10, tmp11, vl);
    out0 = __riscv_vsll_vx_i16m1(out0, PASS1_BITS, vl);
    out4 = __riscv_vsub_vv_i16m1(tmp10, tmp11, vl);
    out4 = __riscv_vsll_vx_i16m1(out4, PASS1_BITS, vl);

    z1 = __riscv_vadd_vv_i16m1(tmp12, tmp13, vl);
    p1 = __riscv_vwmul_vx_i32m2(z1, F_0_541, vl);

    temp = __riscv_vwmul_vx_i32m2(tmp13, F_0_765, vl);
    temp = __riscv_vadd_vv_i32m2(p1, temp, vl);
    temp = __riscv_vadd_vx_i32m2(temp, ROUND_ADD(CONST_BITS - PASS1_BITS), vl);
    out2 = __riscv_vnsra_wx_i16m1(temp, CONST_BITS - PASS1_BITS, vl);

    temp = __riscv_vwmul_vx_i32m2(tmp12, F_1_847, vl);
    temp = __riscv_vsub_vv_i32m2(p1, temp, vl);
    temp = __riscv_vadd_vx_i32m2(temp, ROUND_ADD(CONST_BITS - PASS1_BITS), vl);
    out6 = __riscv_vnsra_wx_i16m1(temp, CONST_BITS - PASS1_BITS, vl);

    z1 = __riscv_vadd_vv_i16m1(tmp4, tmp7, vl);
    z2 = __riscv_vadd_vv_i16m1(tmp5, tmp6, vl);
    z3 = __riscv_vadd_vv_i16m1(tmp4, tmp6, vl);
    z4 = __riscv_vadd_vv_i16m1(tmp5, tmp7, vl);
    z5 = __riscv_vadd_vv_i16m1(z3, z4, vl);
    p5 = __riscv_vwmul_vx_i32m2(z5, F_1_175, vl);

    t4 = __riscv_vwmul_vx_i32m2(tmp4, F_0_298, vl);
    t5 = __riscv_vwmul_vx_i32m2(tmp5, F_2_053, vl);
    t6 = __riscv_vwmul_vx_i32m2(tmp6, F_3_072, vl);
    t7 = __riscv_vwmul_vx_i32m2(tmp7, F_1_501, vl);

    p1 = __riscv_vwmul_vx_i32m2(z1, -F_0_899, vl);
    p2 = __riscv_vwmul_vx_i32m2(z2, -F_2_562, vl);
    p3 = __riscv_vwmul_vx_i32m2(z3, -F_1_961, vl);
    p4 = __riscv_vwmul_vx_i32m2(z4, -F_0_390, vl);

    p3 = __riscv_vadd_vv_i32m2(p3, p5, vl);
    p4 = __riscv_vadd_vv_i32m2(p4, p5, vl);

    temp = __riscv_vadd_vv_i32m2(t4, p1, vl);
    temp = __riscv_vadd_vv_i32m2(temp, p3, vl);
    temp = __riscv_vadd_vx_i32m2(temp, ROUND_ADD(CONST_BITS - PASS1_BITS), vl);
    out7 = __riscv_vnsra_wx_i16m1(temp, CONST_BITS - PASS1_BITS, vl);

    temp = __riscv_vadd_vv_i32m2(t5, p2, vl);
    temp = __riscv_vadd_vv_i32m2(temp, p4, vl);
    temp = __riscv_vadd_vx_i32m2(temp, ROUND_ADD(CONST_BITS - PASS1_BITS), vl);
    out5 = __riscv_vnsra_wx_i16m1(temp, CONST_BITS - PASS1_BITS, vl);

    temp = __riscv_vadd_vv_i32m2(t6, p2, vl);
    temp = __riscv_vadd_vv_i32m2(temp, p3, vl);
    temp = __riscv_vadd_vx_i32m2(temp, ROUND_ADD(CONST_BITS - PASS1_BITS), vl);
    out3 = __riscv_vnsra_wx_i16m1(temp, CONST_BITS - PASS1_BITS, vl);

    temp = __riscv_vadd_vv_i32m2(t7, p1, vl);
    temp = __riscv_vadd_vv_i32m2(temp, p4, vl);
    temp = __riscv_vadd_vx_i32m2(temp, ROUND_ADD(CONST_BITS - PASS1_BITS), vl);
    out1 = __riscv_vnsra_wx_i16m1(temp, CONST_BITS - PASS1_BITS, vl);

    __riscv_vsse16_v_i16m1(data + 0, col_stride, out0, vl);
    __riscv_vsse16_v_i16m1(data + 1, col_stride, out1, vl);
    __riscv_vsse16_v_i16m1(data + 2, col_stride, out2, vl);
    __riscv_vsse16_v_i16m1(data + 3, col_stride, out3, vl);
    __riscv_vsse16_v_i16m1(data + 4, col_stride, out4, vl);
    __riscv_vsse16_v_i16m1(data + 5, col_stride, out5, vl);
    __riscv_vsse16_v_i16m1(data + 6, col_stride, out6, vl);
    __riscv_vsse16_v_i16m1(data + 7, col_stride, out7, vl);

    row0 = __riscv_vle16_v_i16m1(data + DCTSIZE * 0, vl);
    row1 = __riscv_vle16_v_i16m1(data + DCTSIZE * 1, vl);
    row2 = __riscv_vle16_v_i16m1(data + DCTSIZE * 2, vl);
    row3 = __riscv_vle16_v_i16m1(data + DCTSIZE * 3, vl);
    row4 = __riscv_vle16_v_i16m1(data + DCTSIZE * 4, vl);
    row5 = __riscv_vle16_v_i16m1(data + DCTSIZE * 5, vl);
    row6 = __riscv_vle16_v_i16m1(data + DCTSIZE * 6, vl);
    row7 = __riscv_vle16_v_i16m1(data + DCTSIZE * 7, vl);

    tmp0 = __riscv_vadd_vv_i16m1(row0, row7, vl);
    tmp7 = __riscv_vsub_vv_i16m1(row0, row7, vl);
    tmp1 = __riscv_vadd_vv_i16m1(row1, row6, vl);
    tmp6 = __riscv_vsub_vv_i16m1(row1, row6, vl);
    tmp2 = __riscv_vadd_vv_i16m1(row2, row5, vl);
    tmp5 = __riscv_vsub_vv_i16m1(row2, row5, vl);
    tmp3 = __riscv_vadd_vv_i16m1(row3, row4, vl);
    tmp4 = __riscv_vsub_vv_i16m1(row3, row4, vl);

    tmp10 = __riscv_vadd_vv_i16m1(tmp0, tmp3, vl);
    tmp13 = __riscv_vsub_vv_i16m1(tmp0, tmp3, vl);
    tmp11 = __riscv_vadd_vv_i16m1(tmp1, tmp2, vl);
    tmp12 = __riscv_vsub_vv_i16m1(tmp1, tmp2, vl);

    out0 = __riscv_vadd_vv_i16m1(tmp10, tmp11, vl);
    out0 = __riscv_vadd_vx_i16m1(out0, ROUND_ADD(PASS1_BITS), vl);
    out0 = __riscv_vsra_vx_i16m1(out0, PASS1_BITS, vl);
    out4 = __riscv_vsub_vv_i16m1(tmp10, tmp11, vl);
    out4 = __riscv_vadd_vx_i16m1(out4, ROUND_ADD(PASS1_BITS), vl);
    out4 = __riscv_vsra_vx_i16m1(out4, PASS1_BITS, vl);

    z1 = __riscv_vadd_vv_i16m1(tmp12, tmp13, vl);
    p1 = __riscv_vwmul_vx_i32m2(z1, F_0_541, vl);

    temp = __riscv_vwmul_vx_i32m2(tmp13, F_0_765, vl);
    temp = __riscv_vadd_vv_i32m2(p1, temp, vl);
    temp = __riscv_vadd_vx_i32m2(temp, ROUND_ADD(CONST_BITS + PASS1_BITS), vl);
    out2 = __riscv_vnsra_wx_i16m1(temp, CONST_BITS + PASS1_BITS, vl);

    temp = __riscv_vwmul_vx_i32m2(tmp12, F_1_847, vl);
    temp = __riscv_vsub_vv_i32m2(p1, temp, vl);
    temp = __riscv_vadd_vx_i32m2(temp, ROUND_ADD(CONST_BITS + PASS1_BITS), vl);
    out6 = __riscv_vnsra_wx_i16m1(temp, CONST_BITS + PASS1_BITS, vl);

    z1 = __riscv_vadd_vv_i16m1(tmp4, tmp7, vl);
    z2 = __riscv_vadd_vv_i16m1(tmp5, tmp6, vl);
    z3 = __riscv_vadd_vv_i16m1(tmp4, tmp6, vl);
    z4 = __riscv_vadd_vv_i16m1(tmp5, tmp7, vl);
    z5 = __riscv_vadd_vv_i16m1(z3, z4, vl);
    p5 = __riscv_vwmul_vx_i32m2(z5, F_1_175, vl);

    t4 = __riscv_vwmul_vx_i32m2(tmp4, F_0_298, vl);
    t5 = __riscv_vwmul_vx_i32m2(tmp5, F_2_053, vl);
    t6 = __riscv_vwmul_vx_i32m2(tmp6, F_3_072, vl);
    t7 = __riscv_vwmul_vx_i32m2(tmp7, F_1_501, vl);

    p1 = __riscv_vwmul_vx_i32m2(z1, -F_0_899, vl);
    p2 = __riscv_vwmul_vx_i32m2(z2, -F_2_562, vl);
    p3 = __riscv_vwmul_vx_i32m2(z3, -F_1_961, vl);
    p4 = __riscv_vwmul_vx_i32m2(z4, -F_0_390, vl);

    p3 = __riscv_vadd_vv_i32m2(p3, p5, vl);
    p4 = __riscv_vadd_vv_i32m2(p4, p5, vl);

    temp = __riscv_vadd_vv_i32m2(t4, p1, vl);
    temp = __riscv_vadd_vv_i32m2(temp, p3, vl);
    temp = __riscv_vadd_vx_i32m2(temp, ROUND_ADD(CONST_BITS + PASS1_BITS), vl);
    out7 = __riscv_vnsra_wx_i16m1(temp, CONST_BITS + PASS1_BITS, vl);

    temp = __riscv_vadd_vv_i32m2(t5, p2, vl);
    temp = __riscv_vadd_vv_i32m2(temp, p4, vl);
    temp = __riscv_vadd_vx_i32m2(temp, ROUND_ADD(CONST_BITS + PASS1_BITS), vl);
    out5 = __riscv_vnsra_wx_i16m1(temp, CONST_BITS + PASS1_BITS, vl);

    temp = __riscv_vadd_vv_i32m2(t6, p2, vl);
    temp = __riscv_vadd_vv_i32m2(temp, p3, vl);
    temp = __riscv_vadd_vx_i32m2(temp, ROUND_ADD(CONST_BITS + PASS1_BITS), vl);
    out3 = __riscv_vnsra_wx_i16m1(temp, CONST_BITS + PASS1_BITS, vl);

    temp = __riscv_vadd_vv_i32m2(t7, p1, vl);
    temp = __riscv_vadd_vv_i32m2(temp, p4, vl);
    temp = __riscv_vadd_vx_i32m2(temp, ROUND_ADD(CONST_BITS + PASS1_BITS), vl);
    out1 = __riscv_vnsra_wx_i16m1(temp, CONST_BITS + PASS1_BITS, vl);

    __riscv_vse16_v_i16m1(data + DCTSIZE * 0, out0, vl);
    __riscv_vse16_v_i16m1(data + DCTSIZE * 1, out1, vl);
    __riscv_vse16_v_i16m1(data + DCTSIZE * 2, out2, vl);
    __riscv_vse16_v_i16m1(data + DCTSIZE * 3, out3, vl);
    __riscv_vse16_v_i16m1(data + DCTSIZE * 4, out4, vl);
    __riscv_vse16_v_i16m1(data + DCTSIZE * 5, out5, vl);
    __riscv_vse16_v_i16m1(data + DCTSIZE * 6, out6, vl);
    __riscv_vse16_v_i16m1(data + DCTSIZE * 7, out7, vl);
}
