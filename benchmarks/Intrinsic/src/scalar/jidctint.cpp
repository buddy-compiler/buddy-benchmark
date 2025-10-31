

#include <stdint.h>
#include <stddef.h>

typedef short DCTELEM;
typedef int32_t JLONG;
typedef unsigned char JSAMPLE;
typedef JSAMPLE *JSAMPROW;
typedef JSAMPROW *JSAMPARRAY;
typedef short JCOEF;
typedef JCOEF *JCOEFPTR;
typedef unsigned int JDIMENSION;
typedef short ISLOW_MULT_TYPE;

#define DCTSIZE  8
#define DCTSIZE2 64

#define CONST_BITS   13
#define PASS1_BITS   2
#define MAXJSAMPLE   255
#define CENTERJSAMPLE 128
#define RANGE_MASK   (MAXJSAMPLE * 4 + 3)

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

#define DEQUANTIZE(coef, quantval)  (((ISLOW_MULT_TYPE)(coef)) * (quantval))

#define IDCT_range_limit(sample_range_limit) ((JSAMPLE *)(sample_range_limit) + 128)

void idct_islow(short *quantptr, JCOEFPTR coef_block,
                JSAMPARRAY output_buf, JDIMENSION output_col, JSAMPROW sample_range_limit)
{
    JLONG tmp0, tmp1, tmp2, tmp3;
    JLONG tmp10, tmp11, tmp12, tmp13;
    JLONG z1, z2, z3, z4, z5;
    JCOEFPTR inptr;
    ISLOW_MULT_TYPE *quantptr_slow;
    int *wsptr;
    JSAMPROW outptr;
    int ctr;
    int workspace[DCTSIZE2];

    JSAMPLE *range_limit = IDCT_range_limit(sample_range_limit);

    inptr = coef_block;

    wsptr = workspace;
    for (ctr = DCTSIZE; ctr > 0; ctr--) {

        if (inptr[DCTSIZE * 1] == 0 && inptr[DCTSIZE * 2] == 0 &&
            inptr[DCTSIZE * 3] == 0 && inptr[DCTSIZE * 4] == 0 &&
            inptr[DCTSIZE * 5] == 0 && inptr[DCTSIZE * 6] == 0 &&
            inptr[DCTSIZE * 7] == 0) {

            int dcval = LEFT_SHIFT(DEQUANTIZE(inptr[DCTSIZE * 0],
                                    quantptr[DCTSIZE * 0]), PASS1_BITS);
            wsptr[DCTSIZE * 0] = dcval;
            wsptr[DCTSIZE * 1] = dcval;
            wsptr[DCTSIZE * 2] = dcval;
            wsptr[DCTSIZE * 3] = dcval;
            wsptr[DCTSIZE * 4] = dcval;
            wsptr[DCTSIZE * 5] = dcval;
            wsptr[DCTSIZE * 6] = dcval;
            wsptr[DCTSIZE * 7] = dcval;
            inptr++;
            quantptr++;
            wsptr++;
            continue;
        }

        z2 = DEQUANTIZE(inptr[DCTSIZE * 2], quantptr[DCTSIZE * 2]);
        z3 = DEQUANTIZE(inptr[DCTSIZE * 6], quantptr[DCTSIZE * 6]);

        z1 = MULTIPLY(z2 + z3, FIX_0_541196100);
        tmp2 = z1 + MULTIPLY(z3, -FIX_1_847759065);
        tmp3 = z1 + MULTIPLY(z2, FIX_0_765366865);

        z2 = DEQUANTIZE(inptr[DCTSIZE * 0], quantptr[DCTSIZE * 0]);
        z3 = DEQUANTIZE(inptr[DCTSIZE * 4], quantptr[DCTSIZE * 4]);

        tmp0 = LEFT_SHIFT(z2 + z3, CONST_BITS);
        tmp1 = LEFT_SHIFT(z2 - z3, CONST_BITS);

        tmp10 = tmp0 + tmp3;
        tmp13 = tmp0 - tmp3;
        tmp11 = tmp1 + tmp2;
        tmp12 = tmp1 - tmp2;

        tmp0 = DEQUANTIZE(inptr[DCTSIZE * 7], quantptr[DCTSIZE * 7]);
        tmp1 = DEQUANTIZE(inptr[DCTSIZE * 5], quantptr[DCTSIZE * 5]);
        tmp2 = DEQUANTIZE(inptr[DCTSIZE * 3], quantptr[DCTSIZE * 3]);
        tmp3 = DEQUANTIZE(inptr[DCTSIZE * 1], quantptr[DCTSIZE * 1]);

        z1 = tmp0 + tmp3;
        z2 = tmp1 + tmp2;
        z3 = tmp0 + tmp2;
        z4 = tmp1 + tmp3;
        z5 = MULTIPLY(z3 + z4, FIX_1_175875602);

        tmp0 = MULTIPLY(tmp0, FIX_0_298631336);
        tmp1 = MULTIPLY(tmp1, FIX_2_053119869);
        tmp2 = MULTIPLY(tmp2, FIX_3_072711026);
        tmp3 = MULTIPLY(tmp3, FIX_1_501321110);
        z1 = MULTIPLY(z1, -FIX_0_899976223);
        z2 = MULTIPLY(z2, -FIX_2_562915447);
        z3 = MULTIPLY(z3, -FIX_1_961570560);
        z4 = MULTIPLY(z4, -FIX_0_390180644);

        z3 += z5;
        z4 += z5;

        tmp0 += z1 + z3;
        tmp1 += z2 + z4;
        tmp2 += z2 + z3;
        tmp3 += z1 + z4;

        wsptr[DCTSIZE * 0] = (int)DESCALE(tmp10 + tmp3, CONST_BITS - PASS1_BITS);
        wsptr[DCTSIZE * 7] = (int)DESCALE(tmp10 - tmp3, CONST_BITS - PASS1_BITS);
        wsptr[DCTSIZE * 1] = (int)DESCALE(tmp11 + tmp2, CONST_BITS - PASS1_BITS);
        wsptr[DCTSIZE * 6] = (int)DESCALE(tmp11 - tmp2, CONST_BITS - PASS1_BITS);
        wsptr[DCTSIZE * 2] = (int)DESCALE(tmp12 + tmp1, CONST_BITS - PASS1_BITS);
        wsptr[DCTSIZE * 5] = (int)DESCALE(tmp12 - tmp1, CONST_BITS - PASS1_BITS);
        wsptr[DCTSIZE * 3] = (int)DESCALE(tmp13 + tmp0, CONST_BITS - PASS1_BITS);
        wsptr[DCTSIZE * 4] = (int)DESCALE(tmp13 - tmp0, CONST_BITS - PASS1_BITS);

        inptr++;
        quantptr++;
        wsptr++;
    }

    wsptr = workspace;
    for (ctr = 0; ctr < DCTSIZE; ctr++) {
        outptr = output_buf[ctr] + output_col;

        if (wsptr[1] == 0 && wsptr[2] == 0 && wsptr[3] == 0 && wsptr[4] == 0 &&
            wsptr[5] == 0 && wsptr[6] == 0 && wsptr[7] == 0) {

        JSAMPLE dcval = range_limit[(int)DESCALE((JLONG)wsptr[0],
                                                    PASS1_BITS + 3) & RANGE_MASK];

        outptr[0] = dcval;
        outptr[1] = dcval;
        outptr[2] = dcval;
        outptr[3] = dcval;
        outptr[4] = dcval;
        outptr[5] = dcval;
        outptr[6] = dcval;
        outptr[7] = dcval;

        wsptr += DCTSIZE;
        continue;
        }

        z2 = (JLONG)wsptr[2];
        z3 = (JLONG)wsptr[6];

        z1 = MULTIPLY(z2 + z3, FIX_0_541196100);
        tmp2 = z1 + MULTIPLY(z3, -FIX_1_847759065);
        tmp3 = z1 + MULTIPLY(z2, FIX_0_765366865);

        tmp0 = LEFT_SHIFT((JLONG)wsptr[0] + (JLONG)wsptr[4], CONST_BITS);
        tmp1 = LEFT_SHIFT((JLONG)wsptr[0] - (JLONG)wsptr[4], CONST_BITS);

        tmp10 = tmp0 + tmp3;
        tmp13 = tmp0 - tmp3;
        tmp11 = tmp1 + tmp2;
        tmp12 = tmp1 - tmp2;

        tmp0 = (JLONG)wsptr[7];
        tmp1 = (JLONG)wsptr[5];
        tmp2 = (JLONG)wsptr[3];
        tmp3 = (JLONG)wsptr[1];

        z1 = tmp0 + tmp3;
        z2 = tmp1 + tmp2;
        z3 = tmp0 + tmp2;
        z4 = tmp1 + tmp3;
        z5 = MULTIPLY(z3 + z4, FIX_1_175875602);

        tmp0 = MULTIPLY(tmp0, FIX_0_298631336);
        tmp1 = MULTIPLY(tmp1, FIX_2_053119869);
        tmp2 = MULTIPLY(tmp2, FIX_3_072711026);
        tmp3 = MULTIPLY(tmp3, FIX_1_501321110);
        z1 = MULTIPLY(z1, -FIX_0_899976223);
        z2 = MULTIPLY(z2, -FIX_2_562915447);
        z3 = MULTIPLY(z3, -FIX_1_961570560);
        z4 = MULTIPLY(z4, -FIX_0_390180644);

        z3 += z5;
        z4 += z5;

        tmp0 += z1 + z3;
        tmp1 += z2 + z4;
        tmp2 += z2 + z3;
        tmp3 += z1 + z4;

        outptr[0] = range_limit[(int)DESCALE(tmp10 + tmp3,
                                            CONST_BITS + PASS1_BITS + 3) &
                                RANGE_MASK];
        outptr[7] = range_limit[(int)DESCALE(tmp10 - tmp3,
                                            CONST_BITS + PASS1_BITS + 3) &
                                RANGE_MASK];
        outptr[1] = range_limit[(int)DESCALE(tmp11 + tmp2,
                                            CONST_BITS + PASS1_BITS + 3) &
                                RANGE_MASK];
        outptr[6] = range_limit[(int)DESCALE(tmp11 - tmp2,
                                            CONST_BITS + PASS1_BITS + 3) &
                                RANGE_MASK];
        outptr[2] = range_limit[(int)DESCALE(tmp12 + tmp1,
                                            CONST_BITS + PASS1_BITS + 3) &
                                RANGE_MASK];
        outptr[5] = range_limit[(int)DESCALE(tmp12 - tmp1,
                                            CONST_BITS + PASS1_BITS + 3) &
                                RANGE_MASK];
        outptr[3] = range_limit[(int)DESCALE(tmp13 + tmp0,
                                            CONST_BITS + PASS1_BITS + 3) &
                                RANGE_MASK];
        outptr[4] = range_limit[(int)DESCALE(tmp13 - tmp0,
                                            CONST_BITS + PASS1_BITS + 3) &
                                RANGE_MASK];

        wsptr += DCTSIZE;
    }
}