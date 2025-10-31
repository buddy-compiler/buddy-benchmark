

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

    if (tables_initialized)
        return;

    for (i = 0; i <= MAXJSAMPLE; i++) {
        range_limit_table[i] = 0;
    }

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

static void free_ycc_rgb_tables(void)
{
    if (tables_initialized) {

        tables_initialized = 0;
    }
}

void h2v1_merged_upsample(JDIMENSION output_width, JSAMPIMAGE input_buf,
                          JDIMENSION in_row_group_ctr,
                          JSAMPARRAY output_buf)
{
    int y, cred, cgreen, cblue;
    int cb, cr;
    JSAMPROW outptr;
    JSAMPROW inptr0, inptr1, inptr2;
    JDIMENSION col;

    build_ycc_rgb_table();

    if (!tables_initialized) {

        return;
    }

    inptr0 = input_buf[0][in_row_group_ctr];
    inptr1 = input_buf[1][in_row_group_ctr];
    inptr2 = input_buf[2][in_row_group_ctr];
    outptr = output_buf[0];

    for (col = output_width >> 1; col > 0; col--) {

        cb = *inptr1++;
        cr = *inptr2++;
        cred = Cr_r_tab[cr];
        cgreen = (int)((Cb_g_tab[cb] + Cr_g_tab[cr]) >> SCALEBITS);
        cblue = Cb_b_tab[cb];

        y = *inptr0++;
        outptr[RGB_RED] = range_limit[y + cred];
        outptr[RGB_GREEN] = range_limit[y + cgreen];
        outptr[RGB_BLUE] = range_limit[y + cblue];
        outptr += RGB_PIXELSIZE;

        y = *inptr0++;
        outptr[RGB_RED] = range_limit[y + cred];
        outptr[RGB_GREEN] = range_limit[y + cgreen];
        outptr[RGB_BLUE] = range_limit[y + cblue];
        outptr += RGB_PIXELSIZE;
    }

    if (output_width & 1) {
        cb = *inptr1;
        cr = *inptr2;
        cred = Cr_r_tab[cr];
        cgreen = (int)((Cb_g_tab[cb] + Cr_g_tab[cr]) >> SCALEBITS);
        cblue = Cb_b_tab[cb];
        y = *inptr0;
        outptr[RGB_RED] = range_limit[y + cred];
        outptr[RGB_GREEN] = range_limit[y + cgreen];
        outptr[RGB_BLUE] = range_limit[y + cblue];
    }
}

void h2v2_merged_upsample(JDIMENSION output_width, JSAMPIMAGE input_buf,
                          JDIMENSION in_row_group_ctr,
                          JSAMPARRAY output_buf)
{
    int y, cred, cgreen, cblue;
    int cb, cr;
    JSAMPROW outptr0, outptr1;
    JSAMPROW inptr00, inptr01, inptr1, inptr2;
    JDIMENSION col;

    build_ycc_rgb_table();

    if (!tables_initialized) {

        return;
    }

    inptr00 = input_buf[0][in_row_group_ctr * 2];
    inptr01 = input_buf[0][in_row_group_ctr * 2 + 1];
    inptr1 = input_buf[1][in_row_group_ctr];
    inptr2 = input_buf[2][in_row_group_ctr];
    outptr0 = output_buf[0];
    outptr1 = output_buf[1];

    for (col = output_width >> 1; col > 0; col--) {

        cb = *inptr1++;
        cr = *inptr2++;
        cred = Cr_r_tab[cr];
        cgreen = (int)((Cb_g_tab[cb] + Cr_g_tab[cr]) >> SCALEBITS);
        cblue = Cb_b_tab[cb];

        y = *inptr00++;
        outptr0[RGB_RED] = range_limit[y + cred];
        outptr0[RGB_GREEN] = range_limit[y + cgreen];
        outptr0[RGB_BLUE] = range_limit[y + cblue];
        outptr0 += RGB_PIXELSIZE;

        y = *inptr00++;
        outptr0[RGB_RED] = range_limit[y + cred];
        outptr0[RGB_GREEN] = range_limit[y + cgreen];
        outptr0[RGB_BLUE] = range_limit[y + cblue];
        outptr0 += RGB_PIXELSIZE;

        y = *inptr01++;
        outptr1[RGB_RED] = range_limit[y + cred];
        outptr1[RGB_GREEN] = range_limit[y + cgreen];
        outptr1[RGB_BLUE] = range_limit[y + cblue];
        outptr1 += RGB_PIXELSIZE;

        y = *inptr01++;
        outptr1[RGB_RED] = range_limit[y + cred];
        outptr1[RGB_GREEN] = range_limit[y + cgreen];
        outptr1[RGB_BLUE] = range_limit[y + cblue];
        outptr1 += RGB_PIXELSIZE;
    }

    if (output_width & 1) {
        cb = *inptr1;
        cr = *inptr2;
        cred = Cr_r_tab[cr];
        cgreen = (int)((Cb_g_tab[cb] + Cr_g_tab[cr]) >> SCALEBITS);
        cblue = Cb_b_tab[cb];

        y = *inptr00;
        outptr0[RGB_RED] = range_limit[y + cred];
        outptr0[RGB_GREEN] = range_limit[y + cgreen];
        outptr0[RGB_BLUE] = range_limit[y + cblue];

        y = *inptr01;
        outptr1[RGB_RED] = range_limit[y + cred];
        outptr1[RGB_GREEN] = range_limit[y + cgreen];
        outptr1[RGB_BLUE] = range_limit[y + cblue];
    }
}
