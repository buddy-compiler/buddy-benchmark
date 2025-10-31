

#include <stdint.h>
#include <string.h>

typedef uint8_t JSAMPLE;
typedef JSAMPLE *JSAMPROW;
typedef JSAMPROW *JSAMPARRAY;
typedef uint32_t JDIMENSION;

static void
jcopy_sample_rows(JSAMPARRAY input_array, int source_row,
                  JSAMPARRAY output_array, int dest_row,
                  int num_rows, JDIMENSION num_cols)
{
  JSAMPROW inptr, outptr;
  JDIMENSION count;
  int row;

  for (row = 0; row < num_rows; row++) {
    inptr = input_array[source_row + row];
    outptr = output_array[dest_row + row];
    for (count = 0; count < num_cols; count++) {
      *outptr++ = *inptr++;
    }
  }
}

void h2v1_upsample(int max_v_samp_factor, JDIMENSION output_width,
                   JSAMPARRAY input_data, JSAMPARRAY *output_data_ptr)
{
  JSAMPARRAY output_data = *output_data_ptr;
  JSAMPROW inptr, outptr;
  JSAMPLE invalue;
  JSAMPROW outend;
  int inrow;

  for (inrow = 0; inrow < max_v_samp_factor; inrow++) {
    inptr = input_data[inrow];
    outptr = output_data[inrow];
    outend = outptr + output_width;
    while (outptr < outend) {
      invalue = *inptr++;
      *outptr++ = invalue;
      *outptr++ = invalue;
    }
  }
}

void h2v2_upsample(int max_v_samp_factor, JDIMENSION output_width,
                   JSAMPARRAY input_data, JSAMPARRAY *output_data_ptr)
{
  JSAMPARRAY output_data = *output_data_ptr;
  JSAMPROW inptr, outptr;
  JSAMPLE invalue;
  JSAMPROW outend;
  int inrow, outrow;

  inrow = 0;
  outrow = 0;
  while (outrow < max_v_samp_factor) {
    inptr = input_data[inrow];
    outptr = output_data[outrow];
    outend = outptr + output_width;
    while (outptr < outend) {
      invalue = *inptr++;
      *outptr++ = invalue;
      *outptr++ = invalue;
    }
    jcopy_sample_rows(output_data, outrow, output_data, outrow + 1, 1,
                      output_width);
    inrow++;
    outrow += 2;
  }
}

void h2v1_fancy_upsample(int max_v_samp_factor, JDIMENSION downsampled_width,
                         JSAMPARRAY input_data, JSAMPARRAY *output_data_ptr)
{
  JSAMPARRAY output_data = *output_data_ptr;
  JSAMPROW inptr, outptr;
  int invalue;
  JDIMENSION colctr;
  int inrow;

  for (inrow = 0; inrow < max_v_samp_factor; inrow++) {
    inptr = input_data[inrow];
    outptr = output_data[inrow];

    invalue = *inptr++;
    *outptr++ = (JSAMPLE)invalue;
    *outptr++ = (JSAMPLE)((invalue * 3 + inptr[0] + 2) >> 2);

    for (colctr = downsampled_width - 2; colctr > 0; colctr--) {

      invalue = (*inptr++) * 3;
      *outptr++ = (JSAMPLE)((invalue + inptr[-2] + 1) >> 2);
      *outptr++ = (JSAMPLE)((invalue + inptr[0] + 2) >> 2);
    }

    invalue = *inptr;
    *outptr++ = (JSAMPLE)((invalue * 3 + inptr[-1] + 1) >> 2);
    *outptr++ = (JSAMPLE)invalue;
  }
}

void h2v2_fancy_upsample(int max_v_samp_factor, JDIMENSION downsampled_width,
                         JSAMPARRAY input_data, JSAMPARRAY *output_data_ptr)
{
  JSAMPARRAY output_data = *output_data_ptr;
  JSAMPROW inptr0, inptr1, outptr;
  int thiscolsum, lastcolsum, nextcolsum;
  JDIMENSION colctr;
  int inrow, outrow, v;

  inrow = outrow = 0;
  while (outrow < max_v_samp_factor) {
    for (v = 0; v < 2; v++) {

      inptr0 = input_data[inrow];
      if (v == 0)
        inptr1 = input_data[inrow - 1];
      else
        inptr1 = input_data[inrow + 1];
      outptr = output_data[outrow++];

      thiscolsum = (*inptr0++) * 3 + (*inptr1++);
      nextcolsum = (*inptr0++) * 3 + (*inptr1++);
      *outptr++ = (JSAMPLE)((thiscolsum * 4 + 8) >> 4);
      *outptr++ = (JSAMPLE)((thiscolsum * 3 + nextcolsum + 7) >> 4);
      lastcolsum = thiscolsum;  thiscolsum = nextcolsum;

      for (colctr = downsampled_width - 2; colctr > 0; colctr--) {

        nextcolsum = (*inptr0++) * 3 + (*inptr1++);
        *outptr++ = (JSAMPLE)((thiscolsum * 3 + lastcolsum + 8) >> 4);
        *outptr++ = (JSAMPLE)((thiscolsum * 3 + nextcolsum + 7) >> 4);
        lastcolsum = thiscolsum;  thiscolsum = nextcolsum;
      }

      *outptr++ = (JSAMPLE)((thiscolsum * 3 + lastcolsum + 8) >> 4);
      *outptr++ = (JSAMPLE)((thiscolsum * 4 + 7) >> 4);
    }
    inrow++;
  }
}
