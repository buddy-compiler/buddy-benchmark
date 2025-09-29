#include <stdint.h>

typedef short DCTELEM;
typedef int32_t JLONG;

#define DCTSIZE 8

#define CONST_BITS  13
#define PASS1_BITS  2

#define FIX_0_298631336  ((JLONG)2446)
#define FIX_0_390180644  ((JLONG)3196)
#define FIX_0_541196100  ((JLONG)4433)
#define FIX_0_765366865  ((JLONG)6270)
#define FIX_0_899976223  ((JLONG)7373)
#define FIX_1_175875602  ((JLONG)9633)
#define FIX_1_501321110  ((JLONG)12299)
#define FIX_1_847759065  ((JLONG)15137)
#define FIX_1_961570560  ((JLONG)16069)
#define FIX_2_053119869  ((JLONG)16819)
#define FIX_2_562915447  ((JLONG)20995)
#define FIX_3_072711026  ((JLONG)25172)

#define LEFT_SHIFT(x, shft)   ((x) << (shft))
#define RIGHT_SHIFT(x, shft)  ((x) >> (shft))

#define DESCALE(x, n)  RIGHT_SHIFT((x) + (1L << ((n) - 1)), n)

#define MULTIPLY16C16(var, const)  (((int16_t)(var)) * ((int16_t)(const)))
#define MULTIPLY(var, const)  MULTIPLY16C16(var, const)

void fdct_islow(DCTELEM *data)
{
    JLONG tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;
    JLONG tmp10, tmp11, tmp12, tmp13;
    JLONG z1, z2, z3, z4, z5;
    DCTELEM *dataptr;
    int ctr;

    dataptr = data;
    for (ctr = DCTSIZE - 1; ctr >= 0; ctr--) {

        tmp0 = dataptr[0] + dataptr[7];
        tmp7 = dataptr[0] - dataptr[7];
        tmp1 = dataptr[1] + dataptr[6];
        tmp6 = dataptr[1] - dataptr[6];
        tmp2 = dataptr[2] + dataptr[5];
        tmp5 = dataptr[2] - dataptr[5];
        tmp3 = dataptr[3] + dataptr[4];
        tmp4 = dataptr[3] - dataptr[4];

        tmp10 = tmp0 + tmp3;
        tmp13 = tmp0 - tmp3;
        tmp11 = tmp1 + tmp2;
        tmp12 = tmp1 - tmp2;

        dataptr[0] = (DCTELEM)LEFT_SHIFT(tmp10 + tmp11, PASS1_BITS);
        dataptr[4] = (DCTELEM)LEFT_SHIFT(tmp10 - tmp11, PASS1_BITS);

        z1 = MULTIPLY(tmp12 + tmp13, FIX_0_541196100);
        dataptr[2] = (DCTELEM)DESCALE(z1 + MULTIPLY(tmp13, FIX_0_765366865),
                                      CONST_BITS - PASS1_BITS);
        dataptr[6] = (DCTELEM)DESCALE(z1 + MULTIPLY(tmp12, -FIX_1_847759065),
                                      CONST_BITS - PASS1_BITS);

        z1 = tmp4 + tmp7;
        z2 = tmp5 + tmp6;
        z3 = tmp4 + tmp6;
        z4 = tmp5 + tmp7;
        z5 = MULTIPLY(z3 + z4, FIX_1_175875602);

        tmp4 = MULTIPLY(tmp4, FIX_0_298631336);
        tmp5 = MULTIPLY(tmp5, FIX_2_053119869);
        tmp6 = MULTIPLY(tmp6, FIX_3_072711026);
        tmp7 = MULTIPLY(tmp7, FIX_1_501321110);
        z1 = MULTIPLY(z1, -FIX_0_899976223);
        z2 = MULTIPLY(z2, -FIX_2_562915447);
        z3 = MULTIPLY(z3, -FIX_1_961570560);
        z4 = MULTIPLY(z4, -FIX_0_390180644);

        z3 += z5;
        z4 += z5;

        dataptr[7] = (DCTELEM)DESCALE(tmp4 + z1 + z3, CONST_BITS - PASS1_BITS);
        dataptr[5] = (DCTELEM)DESCALE(tmp5 + z2 + z4, CONST_BITS - PASS1_BITS);
        dataptr[3] = (DCTELEM)DESCALE(tmp6 + z2 + z3, CONST_BITS - PASS1_BITS);
        dataptr[1] = (DCTELEM)DESCALE(tmp7 + z1 + z4, CONST_BITS - PASS1_BITS);

        dataptr += DCTSIZE;
    }

    dataptr = data;
    for (ctr = DCTSIZE - 1; ctr >= 0; ctr--) {

        tmp0 = dataptr[DCTSIZE * 0] + dataptr[DCTSIZE * 7];
        tmp7 = dataptr[DCTSIZE * 0] - dataptr[DCTSIZE * 7];
        tmp1 = dataptr[DCTSIZE * 1] + dataptr[DCTSIZE * 6];
        tmp6 = dataptr[DCTSIZE * 1] - dataptr[DCTSIZE * 6];
        tmp2 = dataptr[DCTSIZE * 2] + dataptr[DCTSIZE * 5];
        tmp5 = dataptr[DCTSIZE * 2] - dataptr[DCTSIZE * 5];
        tmp3 = dataptr[DCTSIZE * 3] + dataptr[DCTSIZE * 4];
        tmp4 = dataptr[DCTSIZE * 3] - dataptr[DCTSIZE * 4];

        tmp10 = tmp0 + tmp3;
        tmp13 = tmp0 - tmp3;
        tmp11 = tmp1 + tmp2;
        tmp12 = tmp1 - tmp2;

        dataptr[DCTSIZE * 0] = (DCTELEM)DESCALE(tmp10 + tmp11, PASS1_BITS);
        dataptr[DCTSIZE * 4] = (DCTELEM)DESCALE(tmp10 - tmp11, PASS1_BITS);

        z1 = MULTIPLY(tmp12 + tmp13, FIX_0_541196100);
        dataptr[DCTSIZE * 2] =
            (DCTELEM)DESCALE(z1 + MULTIPLY(tmp13, FIX_0_765366865),
                             CONST_BITS + PASS1_BITS);
        dataptr[DCTSIZE * 6] =
            (DCTELEM)DESCALE(z1 + MULTIPLY(tmp12, -FIX_1_847759065),
                             CONST_BITS + PASS1_BITS);

        z1 = tmp4 + tmp7;
        z2 = tmp5 + tmp6;
        z3 = tmp4 + tmp6;
        z4 = tmp5 + tmp7;
        z5 = MULTIPLY(z3 + z4, FIX_1_175875602);

        tmp4 = MULTIPLY(tmp4, FIX_0_298631336);
        tmp5 = MULTIPLY(tmp5, FIX_2_053119869);
        tmp6 = MULTIPLY(tmp6, FIX_3_072711026);
        tmp7 = MULTIPLY(tmp7, FIX_1_501321110);
        z1 = MULTIPLY(z1, -FIX_0_899976223);
        z2 = MULTIPLY(z2, -FIX_2_562915447);
        z3 = MULTIPLY(z3, -FIX_1_961570560);
        z4 = MULTIPLY(z4, -FIX_0_390180644);

        z3 += z5;
        z4 += z5;

        dataptr[DCTSIZE * 7] = (DCTELEM)DESCALE(tmp4 + z1 + z3,
                                                CONST_BITS + PASS1_BITS);
        dataptr[DCTSIZE * 5] = (DCTELEM)DESCALE(tmp5 + z2 + z4,
                                                CONST_BITS + PASS1_BITS);
        dataptr[DCTSIZE * 3] = (DCTELEM)DESCALE(tmp6 + z2 + z3,
                                                CONST_BITS + PASS1_BITS);
        dataptr[DCTSIZE * 1] = (DCTELEM)DESCALE(tmp7 + z1 + z4,
                                                CONST_BITS + PASS1_BITS);

        dataptr++;
    }
}
