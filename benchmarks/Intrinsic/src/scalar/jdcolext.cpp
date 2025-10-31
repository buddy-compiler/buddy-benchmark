

#include <stdint.h>
#include <stdlib.h>

typedef unsigned char JSAMPLE;
typedef JSAMPLE *JSAMPROW;
typedef JSAMPROW *JSAMPARRAY;
typedef JSAMPARRAY *JSAMPIMAGE;
typedef unsigned int JDIMENSION;
typedef long JLONG;

#define RGB_RED     0
#define RGB_GREEN   1
#define RGB_BLUE    2
#define RGB_PIXELSIZE 3

#define SCALEBITS       16
#define MAXJSAMPLE      255
#define CENTERJSAMPLE   128
#define ONE_HALF        ((JLONG)1 << (SCALEBITS - 1))
#define FIX(x)          ((JLONG)((x) * (1L << SCALEBITS) + 0.5))

#define RANGE_LIMIT(value)  ((value) < 0 ? 0 : ((value) > MAXJSAMPLE ? MAXJSAMPLE : (value)))

static int Cr_r_tab[MAXJSAMPLE + 1];
static int Cb_b_tab[MAXJSAMPLE + 1];
static JLONG Cr_g_tab[MAXJSAMPLE + 1];
static JLONG Cb_g_tab[MAXJSAMPLE + 1];
static int tables_initialized = 0;

static JSAMPLE range_limit_table[3 * (MAXJSAMPLE + 1)];
static JSAMPLE *range_limit = NULL;

static void build_ycc_rgb_table(void)
{
    int i;
    JLONG x;

    for (i = 0; i <= MAXJSAMPLE; i++) {
        range_limit_table[i + MAXJSAMPLE + 1] = (JSAMPLE)i;
    }

    for (i = 0; i <= MAXJSAMPLE; i++) {
        range_limit_table[i + 2 * (MAXJSAMPLE + 1)] = MAXJSAMPLE;
    }
    range_limit = &range_limit_table[MAXJSAMPLE + 1];

    for (i = 0, x = -CENTERJSAMPLE; i <= MAXJSAMPLE; i++, x++) {

        Cr_r_tab[i] = (int)((FIX(1.40200) * x + ONE_HALF) >> SCALEBITS);

        Cb_b_tab[i] = (int)((FIX(1.77200) * x + ONE_HALF) >> SCALEBITS);

        Cr_g_tab[i] = (-FIX(0.71414)) * x;

        Cb_g_tab[i] = (-FIX(0.34414)) * x + ONE_HALF;
    }

    tables_initialized = 1;
}

void ycc_rgb_convert(JDIMENSION output_width, JSAMPIMAGE input_buf,
                     JDIMENSION input_row, JSAMPARRAY output_buf,
                     int num_rows)
{
    int y, cb, cr;
    JSAMPROW outptr;
    JSAMPROW inptr0, inptr1, inptr2;
    JDIMENSION col;
    JDIMENSION num_cols = output_width;

    build_ycc_rgb_table();

    while (--num_rows >= 0) {
        inptr0 = input_buf[0][input_row];
        inptr1 = input_buf[1][input_row];
        inptr2 = input_buf[2][input_row];
        input_row++;
        outptr = *output_buf++;

        for (col = 0; col < num_cols; col++) {
            y  = inptr0[col];
            cb = inptr1[col];
            cr = inptr2[col];

            outptr[RGB_RED] = range_limit[y + Cr_r_tab[cr]];

            outptr[RGB_GREEN] = range_limit[y + ((int)(Cb_g_tab[cb] + Cr_g_tab[cr]) >> SCALEBITS)];

            outptr[RGB_BLUE] = range_limit[y + Cb_b_tab[cb]];

            outptr += RGB_PIXELSIZE;
        }
    }
}
