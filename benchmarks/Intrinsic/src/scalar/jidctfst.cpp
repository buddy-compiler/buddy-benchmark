

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
typedef short IFAST_MULT_TYPE;

#define DCTSIZE  8
#define DCTSIZE2 64

#define CONST_BITS   8
#define PASS1_BITS   2
#define IFAST_SCALE_BITS  2
#define MAXJSAMPLE   255
#define CENTERJSAMPLE 128
#define RANGE_MASK   (MAXJSAMPLE * 4 + 3)

#define FIX_1_082392200  ((JLONG)277)
#define FIX_1_414213562  ((JLONG)362)
#define FIX_1_847759065  ((JLONG)473)
#define FIX_2_613125930  ((JLONG)669)

#define RIGHT_SHIFT(x, shft)  ((x) >> (shft))

#define DESCALE(x, n)  RIGHT_SHIFT(x, n)

#define IDESCALE(x, n)  ((int)RIGHT_SHIFT(x, n))

#define MULTIPLY(var, const)  ((DCTELEM)DESCALE((var) * (const), CONST_BITS))

#define DEQUANTIZE(coef, quantval)  (((IFAST_MULT_TYPE)(coef)) * (quantval))

#define IDCT_range_limit(sample_range_limit) ((JSAMPLE *)(sample_range_limit) + 128)

void idct_ifast(void *dct_table, JCOEFPTR coef_block,
                JSAMPARRAY output_buf, JDIMENSION output_col, JSAMPROW sample_range_limit)
{
    DCTELEM tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;
    DCTELEM tmp10, tmp11, tmp12, tmp13;
    DCTELEM z5, z10, z11, z12, z13;
    JCOEFPTR inptr;
    IFAST_MULT_TYPE *quantptr;
    int *wsptr;
    JSAMPROW outptr;
    int ctr;
    int workspace[DCTSIZE2];

   JSAMPLE *range_limit = IDCT_range_limit(sample_range_limit);

    inptr = coef_block;
    quantptr = (IFAST_MULT_TYPE *)dct_table;
    wsptr = workspace;

    for (ctr = DCTSIZE; ctr > 0; ctr--) {

        if (inptr[DCTSIZE * 1] == 0 && inptr[DCTSIZE * 2] == 0 &&
            inptr[DCTSIZE * 3] == 0 && inptr[DCTSIZE * 4] == 0 &&
            inptr[DCTSIZE * 5] == 0 && inptr[DCTSIZE * 6] == 0 &&
            inptr[DCTSIZE * 7] == 0) {

            int dcval = (int)DEQUANTIZE(inptr[DCTSIZE * 0], quantptr[DCTSIZE * 0]);

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

        tmp0 = DEQUANTIZE(inptr[DCTSIZE * 0], quantptr[DCTSIZE * 0]);
        tmp1 = DEQUANTIZE(inptr[DCTSIZE * 2], quantptr[DCTSIZE * 2]);
        tmp2 = DEQUANTIZE(inptr[DCTSIZE * 4], quantptr[DCTSIZE * 4]);
        tmp3 = DEQUANTIZE(inptr[DCTSIZE * 6], quantptr[DCTSIZE * 6]);

        tmp10 = tmp0 + tmp2;
        tmp11 = tmp0 - tmp2;

        tmp13 = tmp1 + tmp3;
        tmp12 = MULTIPLY(tmp1 - tmp3, FIX_1_414213562) - tmp13;

        tmp0 = tmp10 + tmp13;
        tmp3 = tmp10 - tmp13;
        tmp1 = tmp11 + tmp12;
        tmp2 = tmp11 - tmp12;

        tmp4 = DEQUANTIZE(inptr[DCTSIZE * 1], quantptr[DCTSIZE * 1]);
        tmp5 = DEQUANTIZE(inptr[DCTSIZE * 3], quantptr[DCTSIZE * 3]);
        tmp6 = DEQUANTIZE(inptr[DCTSIZE * 5], quantptr[DCTSIZE * 5]);
        tmp7 = DEQUANTIZE(inptr[DCTSIZE * 7], quantptr[DCTSIZE * 7]);

        z13 = tmp6 + tmp5;
        z10 = tmp6 - tmp5;
        z11 = tmp4 + tmp7;
        z12 = tmp4 - tmp7;

        tmp7 = z11 + z13;
        tmp11 = MULTIPLY(z11 - z13, FIX_1_414213562);

        z5 = MULTIPLY(z10 + z12, FIX_1_847759065);
        tmp10 = MULTIPLY(z12, FIX_1_082392200) - z5;
        tmp12 = MULTIPLY(z10, -FIX_2_613125930) + z5;

        tmp6 = tmp12 - tmp7;
        tmp5 = tmp11 - tmp6;
        tmp4 = tmp10 + tmp5;

        wsptr[DCTSIZE * 0] = (int)(tmp0 + tmp7);
        wsptr[DCTSIZE * 7] = (int)(tmp0 - tmp7);
        wsptr[DCTSIZE * 1] = (int)(tmp1 + tmp6);
        wsptr[DCTSIZE * 6] = (int)(tmp1 - tmp6);
        wsptr[DCTSIZE * 2] = (int)(tmp2 + tmp5);
        wsptr[DCTSIZE * 5] = (int)(tmp2 - tmp5);
        wsptr[DCTSIZE * 4] = (int)(tmp3 + tmp4);
        wsptr[DCTSIZE * 3] = (int)(tmp3 - tmp4);

        inptr++;
        quantptr++;
        wsptr++;
    }

    wsptr = workspace;
    for (ctr = 0; ctr < DCTSIZE; ctr++) {
        outptr = output_buf[ctr] + output_col;

        if (wsptr[1] == 0 && wsptr[2] == 0 && wsptr[3] == 0 && wsptr[4] == 0 &&
            wsptr[5] == 0 && wsptr[6] == 0 && wsptr[7] == 0) {

            JSAMPLE dcval = range_limit[IDESCALE(wsptr[0], PASS1_BITS + 3) & RANGE_MASK];

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

        tmp10 = ((DCTELEM)wsptr[0] + (DCTELEM)wsptr[4]);
        tmp11 = ((DCTELEM)wsptr[0] - (DCTELEM)wsptr[4]);

        tmp13 = ((DCTELEM)wsptr[2] + (DCTELEM)wsptr[6]);
        tmp12 = MULTIPLY((DCTELEM)wsptr[2] - (DCTELEM)wsptr[6], FIX_1_414213562) - tmp13;

        tmp0 = tmp10 + tmp13;
        tmp3 = tmp10 - tmp13;
        tmp1 = tmp11 + tmp12;
        tmp2 = tmp11 - tmp12;

        z13 = (DCTELEM)wsptr[5] + (DCTELEM)wsptr[3];
        z10 = (DCTELEM)wsptr[5] - (DCTELEM)wsptr[3];
        z11 = (DCTELEM)wsptr[1] + (DCTELEM)wsptr[7];
        z12 = (DCTELEM)wsptr[1] - (DCTELEM)wsptr[7];

        tmp7 = z11 + z13;
        tmp11 = MULTIPLY(z11 - z13, FIX_1_414213562);

        z5 = MULTIPLY(z10 + z12, FIX_1_847759065);
        tmp10 = MULTIPLY(z12, FIX_1_082392200) - z5;
        tmp12 = MULTIPLY(z10, -FIX_2_613125930) + z5;

        tmp6 = tmp12 - tmp7;
        tmp5 = tmp11 - tmp6;
        tmp4 = tmp10 + tmp5;

        outptr[0] = range_limit[IDESCALE(tmp0 + tmp7, PASS1_BITS + 3) & RANGE_MASK];
        outptr[7] = range_limit[IDESCALE(tmp0 - tmp7, PASS1_BITS + 3) & RANGE_MASK];
        outptr[1] = range_limit[IDESCALE(tmp1 + tmp6, PASS1_BITS + 3) & RANGE_MASK];
        outptr[6] = range_limit[IDESCALE(tmp1 - tmp6, PASS1_BITS + 3) & RANGE_MASK];
        outptr[2] = range_limit[IDESCALE(tmp2 + tmp5, PASS1_BITS + 3) & RANGE_MASK];
        outptr[5] = range_limit[IDESCALE(tmp2 - tmp5, PASS1_BITS + 3) & RANGE_MASK];
        outptr[4] = range_limit[IDESCALE(tmp3 + tmp4, PASS1_BITS + 3) & RANGE_MASK];
        outptr[3] = range_limit[IDESCALE(tmp3 - tmp4, PASS1_BITS + 3) & RANGE_MASK];

        wsptr += DCTSIZE;
    }
}
