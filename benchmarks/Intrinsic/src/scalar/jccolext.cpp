#include <stdint.h>

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
#define CBCR_OFFSET     ((JLONG)CENTERJSAMPLE << SCALEBITS)
#define ONE_HALF        ((JLONG)1 << (SCALEBITS - 1))
#define FIX(x)          ((JLONG)((x) * (1L << SCALEBITS) + 0.5))

#define R_Y_OFF         0
#define G_Y_OFF         (1 * (MAXJSAMPLE + 1))
#define B_Y_OFF         (2 * (MAXJSAMPLE + 1))
#define R_CB_OFF        (3 * (MAXJSAMPLE + 1))
#define G_CB_OFF        (4 * (MAXJSAMPLE + 1))
#define B_CB_OFF        (5 * (MAXJSAMPLE + 1))
#define R_CR_OFF        B_CB_OFF
#define G_CR_OFF        (6 * (MAXJSAMPLE + 1))
#define B_CR_OFF        (7 * (MAXJSAMPLE + 1))
#define TABLE_SIZE      (8 * (MAXJSAMPLE + 1))

#define RANGE_LIMIT(value)  ((value) < 0 ? 0 : ((value) > MAXJSAMPLE ? MAXJSAMPLE : (value)))

static JLONG rgb_ycc_tab[TABLE_SIZE];
static int table_initialized = 0;

static void init_rgb_ycc_table(void)
{
    JLONG i;

    if (table_initialized)
        return;

    for (i = 0; i <= MAXJSAMPLE; i++) {
        rgb_ycc_tab[i + R_Y_OFF] = FIX(0.29900) * i;
        rgb_ycc_tab[i + G_Y_OFF] = FIX(0.58700) * i;
        rgb_ycc_tab[i + B_Y_OFF] = FIX(0.11400) * i + ONE_HALF;
        rgb_ycc_tab[i + R_CB_OFF] = (-FIX(0.16874)) * i;
        rgb_ycc_tab[i + G_CB_OFF] = (-FIX(0.33126)) * i;

        rgb_ycc_tab[i + B_CB_OFF] = FIX(0.50000) * i + CBCR_OFFSET + ONE_HALF - 1;

        rgb_ycc_tab[i + G_CR_OFF] = (-FIX(0.41869)) * i;
        rgb_ycc_tab[i + B_CR_OFF] = (-FIX(0.08131)) * i;
    }

    table_initialized = 1;
}

void rgb_ycc_convert(JDIMENSION image_width, JSAMPARRAY input_buf,
                     JSAMPIMAGE output_buf, JDIMENSION output_row, int num_rows)
{
    int r, g, b;
    JLONG *ctab;
    JSAMPROW inptr;
    JSAMPROW outptr0, outptr1, outptr2;
    JDIMENSION col;
    JDIMENSION num_cols = image_width;

    init_rgb_ycc_table();
    ctab = rgb_ycc_tab;

    while (--num_rows >= 0) {
        inptr = *input_buf++;
        outptr0 = output_buf[0][output_row];
        outptr1 = output_buf[1][output_row];
        outptr2 = output_buf[2][output_row];
        output_row++;

        for (col = 0; col < num_cols; col++) {

            r = RANGE_LIMIT(inptr[RGB_RED]);
            g = RANGE_LIMIT(inptr[RGB_GREEN]);
            b = RANGE_LIMIT(inptr[RGB_BLUE]);
            inptr += RGB_PIXELSIZE;

            outptr0[col] = (JSAMPLE)((ctab[r + R_Y_OFF] + ctab[g + G_Y_OFF] +
                                     ctab[b + B_Y_OFF]) >> SCALEBITS);

            outptr1[col] = (JSAMPLE)((ctab[r + R_CB_OFF] + ctab[g + G_CB_OFF] +
                                     ctab[b + B_CB_OFF]) >> SCALEBITS);

            outptr2[col] = (JSAMPLE)((ctab[r + R_CR_OFF] + ctab[g + G_CR_OFF] +
                                     ctab[b + B_CR_OFF]) >> SCALEBITS);
        }
    }
}
