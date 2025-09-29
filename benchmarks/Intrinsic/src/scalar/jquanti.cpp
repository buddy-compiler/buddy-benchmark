#include <stddef.h>

typedef unsigned char JSAMPLE;
typedef short JCOEF;
typedef JCOEF *JCOEFPTR;

#define DCTSIZE         8
#define DCTSIZE2        64

#define MAXJSAMPLE      255
#define CENTERJSAMPLE   128

typedef short DCTELEM;
typedef unsigned short UDCTELEM;
typedef unsigned int JDIMENSION;
typedef unsigned int UDCTELEM2;

typedef JSAMPLE *JSAMPROW;
typedef JSAMPROW *JSAMPARRAY;

void convsamp(JSAMPARRAY sample_data, JDIMENSION start_col, DCTELEM *workspace)
{
    DCTELEM *workspaceptr;
    JSAMPROW elemptr;
    int elemr;

    workspaceptr = workspace;

    for (elemr = 0; elemr < DCTSIZE; elemr++) {
        elemptr = sample_data[elemr] + start_col;

        *workspaceptr++ = (*elemptr++) - CENTERJSAMPLE;
        *workspaceptr++ = (*elemptr++) - CENTERJSAMPLE;
        *workspaceptr++ = (*elemptr++) - CENTERJSAMPLE;
        *workspaceptr++ = (*elemptr++) - CENTERJSAMPLE;
        *workspaceptr++ = (*elemptr++) - CENTERJSAMPLE;
        *workspaceptr++ = (*elemptr++) - CENTERJSAMPLE;
        *workspaceptr++ = (*elemptr++) - CENTERJSAMPLE;
        *workspaceptr++ = (*elemptr++) - CENTERJSAMPLE;
    }
}

void quantize(JCOEFPTR coef_block, DCTELEM *divisors, DCTELEM *workspace)
{
    int i;
    DCTELEM temp;
    JCOEFPTR output_ptr = coef_block;

    UDCTELEM recip, corr;
    int shift;
    UDCTELEM2 product;

    for (i = 0; i < DCTSIZE2; i++) {
        temp = workspace[i];
        recip = divisors[i + DCTSIZE2 * 0];
        corr =  divisors[i + DCTSIZE2 * 1];
        shift = divisors[i + DCTSIZE2 * 3];

        if (temp < 0) {
            temp = -temp;
            product = (UDCTELEM2)(temp + corr) * recip;
            product >>= shift + sizeof(DCTELEM) * 8;
            temp = (DCTELEM)product;
            temp = -temp;
        } else {
            product = (UDCTELEM2)(temp + corr) * recip;
            product >>= shift + sizeof(DCTELEM) * 8;
            temp = (DCTELEM)product;
        }
        output_ptr[i] = (JCOEF)temp;
    }
}
