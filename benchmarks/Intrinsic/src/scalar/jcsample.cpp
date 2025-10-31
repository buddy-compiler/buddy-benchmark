

#include <stdint.h>

typedef unsigned char JSAMPLE;
typedef JSAMPLE *JSAMPROW;
typedef JSAMPROW *JSAMPARRAY;
typedef unsigned int JDIMENSION;

static void expand_right_edge(JSAMPARRAY image_data, int num_rows,
                             JDIMENSION input_cols, JDIMENSION output_cols)
{
    JSAMPROW ptr;
    JSAMPLE pixval;
    int count;
    int row;
    int numcols = (int)(output_cols - input_cols);

    if (numcols > 0) {
        for (row = 0; row < num_rows; row++) {
            ptr = image_data[row] + input_cols;
            pixval = ptr[-1];
            for (count = numcols; count > 0; count--)
                *ptr++ = pixval;
        }
    }
}

void h2v1_downsample(JDIMENSION image_width, int max_v_samp_factor,
                     JDIMENSION v_samp_factor, JDIMENSION width_in_blocks,
                     JSAMPARRAY input_data, JSAMPARRAY output_data)
{
    int outrow;
    JDIMENSION outcol;
    JDIMENSION output_cols = width_in_blocks * 8;
    JSAMPROW inptr, outptr;
    int bias;

    expand_right_edge(input_data, max_v_samp_factor, image_width,
                      output_cols * 2);

    for (outrow = 0; outrow < (int)v_samp_factor; outrow++) {
        outptr = output_data[outrow];
        inptr = input_data[outrow];
        bias = 0;

        for (outcol = 0; outcol < output_cols; outcol++) {

            *outptr++ = (JSAMPLE)((inptr[0] + inptr[1] + bias) >> 1);
            bias ^= 1;
            inptr += 2;
        }
    }
}

void h2v2_downsample(JDIMENSION image_width, int max_v_samp_factor,
                     JDIMENSION v_samp_factor, JDIMENSION width_in_blocks,
                     JSAMPARRAY input_data, JSAMPARRAY output_data)
{
    int inrow, outrow;
    JDIMENSION outcol;
    JDIMENSION output_cols = width_in_blocks * 8;
    JSAMPROW inptr0, inptr1, outptr;
    int bias;

    expand_right_edge(input_data, max_v_samp_factor, image_width,
                      output_cols * 2);

    inrow = 0;
    for (outrow = 0; outrow < (int)v_samp_factor; outrow++) {
        outptr = output_data[outrow];
        inptr0 = input_data[inrow];
        inptr1 = input_data[inrow + 1];
        bias = 1;

        for (outcol = 0; outcol < output_cols; outcol++) {

            *outptr++ = (JSAMPLE)
                ((inptr0[0] + inptr0[1] + inptr1[0] + inptr1[1] + bias) >> 2);
            bias ^= 3;
            inptr0 += 2;
            inptr1 += 2;
        }
        inrow += 2;
    }
}
