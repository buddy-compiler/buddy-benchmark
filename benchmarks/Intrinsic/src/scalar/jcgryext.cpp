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
#define ONE_HALF        ((JLONG)1 << (SCALEBITS - 1))
#define FIX(x)          ((JLONG)((x) * (1L << SCALEBITS) + 0.5))

#define R_Y_OFF         0
#define G_Y_OFF         (1 * (MAXJSAMPLE + 1))
#define B_Y_OFF         (2 * (MAXJSAMPLE + 1))
#define GRAY_TABLE_SIZE (3 * (MAXJSAMPLE + 1))

#define RANGE_LIMIT(value)  ((value) < 0 ? 0 : ((value) > MAXJSAMPLE ? MAXJSAMPLE : (value)))

static JLONG rgb_gray_tab[GRAY_TABLE_SIZE];
static int gray_table_initialized = 0;

static void init_rgb_gray_table(void)
{
    JLONG i;

    if (gray_table_initialized)
        return;

    for (i = 0; i <= MAXJSAMPLE; i++) {
        rgb_gray_tab[i + R_Y_OFF] = FIX(0.29900) * i;
        rgb_gray_tab[i + G_Y_OFF] = FIX(0.58700) * i;
        rgb_gray_tab[i + B_Y_OFF] = FIX(0.11400) * i + ONE_HALF;
    }

    gray_table_initialized = 1;
}

void rgb_gray_convert(JDIMENSION image_width, JSAMPARRAY input_buf,
                      JSAMPIMAGE output_buf, JDIMENSION output_row,
                      int num_rows)
{
    int r, g, b;
    JLONG *ctab;
    JSAMPROW inptr;
    JSAMPROW outptr;
    JDIMENSION col;
    JDIMENSION num_cols = image_width;

    init_rgb_gray_table();
    ctab = rgb_gray_tab;

    while (--num_rows >= 0) {
        inptr = *input_buf++;
        outptr = output_buf[0][output_row];
        output_row++;

        for (col = 0; col < num_cols; col++) {

            r = RANGE_LIMIT(inptr[RGB_RED]);
            g = RANGE_LIMIT(inptr[RGB_GREEN]);
            b = RANGE_LIMIT(inptr[RGB_BLUE]);
            inptr += RGB_PIXELSIZE;

            outptr[col] = (JSAMPLE)((ctab[r + R_Y_OFF] + ctab[g + G_Y_OFF] +
                                    ctab[b + B_Y_OFF]) >> SCALEBITS);
        }
    }
}