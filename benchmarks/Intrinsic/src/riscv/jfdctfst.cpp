#include <riscv_vector.h>

typedef unsigned int JDIMENSION;
typedef unsigned char JSAMPLE;

typedef JSAMPLE *JSAMPROW;
typedef JSAMPROW *JSAMPARRAY;
typedef JSAMPARRAY *JSAMPIMAGE;
#define F_0_382  98
#define F_0_541  139
#define F_0_707  181
#define F_1_306  334

#define CONST_BITS  8

#define DCTSIZE             8
typedef short DCTELEM;

#define DO_DCT() { \
 \
    tmp10 = __riscv_vadd_vv_i16m1(tmp0, tmp3, vl); \
    tmp13 = __riscv_vsub_vv_i16m1(tmp0, tmp3, vl); \
    tmp11 = __riscv_vadd_vv_i16m1(tmp1, tmp2, vl); \
    tmp12 = __riscv_vsub_vv_i16m1(tmp1, tmp2, vl); \
    \
    out0 = __riscv_vadd_vv_i16m1(tmp10, tmp11, vl); \
    out4 = __riscv_vsub_vv_i16m1(tmp10, tmp11, vl); \
    \
    z1 = __riscv_vadd_vv_i16m1(tmp12, tmp13, vl); \
    pdt = __riscv_vwmul_vx_i32m2(z1, F_0_707, vl); \
    z1 = __riscv_vnsra_wx_i16m1(pdt, CONST_BITS, vl); \
    out2 = __riscv_vadd_vv_i16m1(tmp13, z1, vl); \
    out6 = __riscv_vsub_vv_i16m1(tmp13, z1, vl); \
    \
 \
    tmp10 = __riscv_vadd_vv_i16m1(tmp4, tmp5, vl);\
    tmp11 = __riscv_vadd_vv_i16m1(tmp5, tmp6, vl); \
    tmp12 = __riscv_vadd_vv_i16m1(tmp6, tmp7, vl); \
    \
    z5 = __riscv_vsub_vv_i16m1(tmp10, tmp12, vl); \
    pdt = __riscv_vwmul_vx_i32m2(z5, F_0_382, vl); \
    z5 = __riscv_vnsra_wx_i16m1(pdt, CONST_BITS, vl); \
    pdt = __riscv_vwmul_vx_i32m2(tmp10, F_0_541, vl); \
    z2 = __riscv_vnsra_wx_i16m1(pdt, CONST_BITS, vl); \
    z2 = __riscv_vadd_vv_i16m1(z2, z5, vl); \
    pdt = __riscv_vwmul_vx_i32m2(tmp12, F_1_306, vl); \
    z4 = __riscv_vnsra_wx_i16m1(pdt, CONST_BITS, vl); \
    z4 = __riscv_vadd_vv_i16m1(z4, z5, vl); \
    pdt = __riscv_vwmul_vx_i32m2(tmp11, F_0_707, vl); \
    z3 = __riscv_vnsra_wx_i16m1(pdt, CONST_BITS, vl); \
    \
    z11 = __riscv_vadd_vv_i16m1(tmp7, z3, vl); \
    z13 = __riscv_vsub_vv_i16m1(tmp7, z3, vl); \
    \
    out5 = __riscv_vadd_vv_i16m1(z13, z2, vl); \
    out3 = __riscv_vsub_vv_i16m1(z13, z2, vl); \
    out1 = __riscv_vadd_vv_i16m1(z11, z4, vl); \
    out7 = __riscv_vsub_vv_i16m1(z11, z4, vl); \
}

void fdct_ifast(DCTELEM *data)
{
    vint16m1_t row0, row1, row2, row3, row4, row5, row6, row7,
               col0, col1, col2, col3, col4, col5, col6, col7,
               tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7,
               tmp10, tmp11, tmp12, tmp13,
               z1, z2, z3, z4, z5, z11, z13,
               out0, out1, out2, out3, out4, out5, out6, out7;
    vint32m2_t pdt;
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

    DO_DCT();

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

    DO_DCT();

    __riscv_vse16_v_i16m1(data + DCTSIZE * 0, out0, vl);
    __riscv_vse16_v_i16m1(data + DCTSIZE * 1, out1, vl);
    __riscv_vse16_v_i16m1(data + DCTSIZE * 2, out2, vl);
    __riscv_vse16_v_i16m1(data + DCTSIZE * 3, out3, vl);
    __riscv_vse16_v_i16m1(data + DCTSIZE * 4, out4, vl);
    __riscv_vse16_v_i16m1(data + DCTSIZE * 5, out5, vl);
    __riscv_vse16_v_i16m1(data + DCTSIZE * 6, out6, vl);
    __riscv_vse16_v_i16m1(data + DCTSIZE * 7, out7, vl);
}
