#include <arm_neon.h>

typedef unsigned int JDIMENSION;
typedef unsigned char JSAMPLE;
typedef JSAMPLE *JSAMPROW;
typedef JSAMPROW *JSAMPARRAY;
typedef JSAMPARRAY *JSAMPIMAGE;

typedef unsigned char JSAMPLE;
#define GETJSAMPLE(value)  ((int)(value))

void h2v1_fancy_upsample(int max_v_samp_factor,
                               JDIMENSION downsampled_width,
                               JSAMPARRAY input_data,
                               JSAMPARRAY *output_data_ptr)
{
  JSAMPARRAY output_data = *output_data_ptr;
  JSAMPROW inptr, outptr;
  int inrow;
  unsigned colctr;
  const uint16x8_t one_u16 = vdupq_n_u16(1);
  const uint8x8_t three_u8 = vdup_n_u8(3);

  for (inrow = 0; inrow < max_v_samp_factor; inrow++) {
    inptr = input_data[inrow];
    outptr = output_data[inrow];

    *outptr = (JSAMPLE)GETJSAMPLE(*inptr);

    uint8x16_t s0 = vld1q_u8(inptr);
    uint8x16_t s1 = vld1q_u8(inptr + 1);

    uint16x8_t s1_add_3s0_l =
      vmlal_u8(vmovl_u8(vget_low_u8(s1)), vget_low_u8(s0), three_u8);
    uint16x8_t s1_add_3s0_h =
      vmlal_u8(vmovl_u8(vget_high_u8(s1)), vget_high_u8(s0), three_u8);
    uint16x8_t s0_add_3s1_l =
      vmlal_u8(vmovl_u8(vget_low_u8(s0)), vget_low_u8(s1), three_u8);
    uint16x8_t s0_add_3s1_h =
      vmlal_u8(vmovl_u8(vget_high_u8(s0)), vget_high_u8(s1), three_u8);

    s0_add_3s1_l = vaddq_u16(s0_add_3s1_l, one_u16);
    s0_add_3s1_h = vaddq_u16(s0_add_3s1_h, one_u16);

    unsigned outptr_offset = 1;
    uint8x16x2_t output_pixels;

    for (colctr = 16; colctr < downsampled_width; colctr += 16) {

        s0 = vld1q_u8(inptr + colctr - 1);
        s1 = vld1q_u8(inptr + colctr);

      output_pixels.val[0] = vcombine_u8(vrshrn_n_u16(s1_add_3s0_l, 2),
                                         vrshrn_n_u16(s1_add_3s0_h, 2));
      output_pixels.val[1] = vcombine_u8(vshrn_n_u16(s0_add_3s1_l, 2),
                                         vshrn_n_u16(s0_add_3s1_h, 2));

        s1_add_3s0_l =
          vmlal_u8(vmovl_u8(vget_low_u8(s1)), vget_low_u8(s0), three_u8);
        s1_add_3s0_h =
          vmlal_u8(vmovl_u8(vget_high_u8(s1)), vget_high_u8(s0), three_u8);
        s0_add_3s1_l =
          vmlal_u8(vmovl_u8(vget_low_u8(s0)), vget_low_u8(s1), three_u8);
        s0_add_3s1_h =
          vmlal_u8(vmovl_u8(vget_high_u8(s0)), vget_high_u8(s1), three_u8);
        s0_add_3s1_l = vaddq_u16(s0_add_3s1_l, one_u16);
        s0_add_3s1_h = vaddq_u16(s0_add_3s1_h, one_u16);

      vst2q_u8(outptr + outptr_offset, output_pixels);
      outptr_offset = 2 * colctr - 1;
    }

    output_pixels.val[0] = vcombine_u8(vrshrn_n_u16(s1_add_3s0_l, 2),
                                       vrshrn_n_u16(s1_add_3s0_h, 2));
    output_pixels.val[1] = vcombine_u8(vshrn_n_u16(s0_add_3s1_l, 2),
                                       vshrn_n_u16(s0_add_3s1_h, 2));
    vst2q_u8(outptr + outptr_offset, output_pixels);

    outptr[2 * downsampled_width - 1] =
      GETJSAMPLE(inptr[downsampled_width - 1]);
  }
}

void h2v2_fancy_upsample(int max_v_samp_factor,
                               JDIMENSION downsampled_width,
                               JSAMPARRAY input_data,
                               JSAMPARRAY *output_data_ptr)
{
  JSAMPARRAY output_data = *output_data_ptr;
  JSAMPROW inptr0, inptr1, inptr2, outptr0, outptr1;
  int inrow, outrow;
  unsigned colctr;

  const uint16x8_t seven_u16 = vdupq_n_u16(7);
  const uint8x8_t three_u8 = vdup_n_u8(3);
  const uint16x8_t three_u16 = vdupq_n_u16(3);

  inrow = outrow = 0;
  while (outrow < max_v_samp_factor) {
    inptr0 = input_data[inrow - 1];
    inptr1 = input_data[inrow];
    inptr2 = input_data[inrow + 1];

    outptr0 = output_data[outrow++];
    outptr1 = output_data[outrow++];

    int s0colsum0 = GETJSAMPLE(*inptr1) * 3 + GETJSAMPLE(*inptr0);
    *outptr0 = (JSAMPLE)((s0colsum0 * 4 + 8) >> 4);
    int s0colsum1 = GETJSAMPLE(*inptr1) * 3 + GETJSAMPLE(*inptr2);
    *outptr1 = (JSAMPLE)((s0colsum1 * 4 + 8) >> 4);

    uint8x16_t s0A = vld1q_u8(inptr0);
    uint8x16_t s0B = vld1q_u8(inptr1);
    uint8x16_t s0C = vld1q_u8(inptr2);

    uint16x8_t s0colsum0_l = vmlal_u8(vmovl_u8(vget_low_u8(s0A)),
                                      vget_low_u8(s0B), three_u8);
    uint16x8_t s0colsum0_h = vmlal_u8(vmovl_u8(vget_high_u8(s0A)),
                                      vget_high_u8(s0B), three_u8);
    uint16x8_t s0colsum1_l = vmlal_u8(vmovl_u8(vget_low_u8(s0C)),
                                      vget_low_u8(s0B), three_u8);
    uint16x8_t s0colsum1_h = vmlal_u8(vmovl_u8(vget_high_u8(s0C)),
                                      vget_high_u8(s0B), three_u8);

    uint8x16_t s1A = vld1q_u8(inptr0 + 1);
    uint8x16_t s1B = vld1q_u8(inptr1 + 1);
    uint8x16_t s1C = vld1q_u8(inptr2 + 1);
    uint16x8_t s1colsum0_l = vmlal_u8(vmovl_u8(vget_low_u8(s1A)),
                                      vget_low_u8(s1B), three_u8);
    uint16x8_t s1colsum0_h = vmlal_u8(vmovl_u8(vget_high_u8(s1A)),
                                      vget_high_u8(s1B), three_u8);
    uint16x8_t s1colsum1_l = vmlal_u8(vmovl_u8(vget_low_u8(s1C)),
                                      vget_low_u8(s1B), three_u8);
    uint16x8_t s1colsum1_h = vmlal_u8(vmovl_u8(vget_high_u8(s1C)),
                                      vget_high_u8(s1B), three_u8);

    uint16x8_t output0_p1_l = vmlaq_u16(s1colsum0_l, s0colsum0_l, three_u16);
    uint16x8_t output0_p1_h = vmlaq_u16(s1colsum0_h, s0colsum0_h, three_u16);
    uint16x8_t output0_p2_l = vmlaq_u16(s0colsum0_l, s1colsum0_l, three_u16);
    uint16x8_t output0_p2_h = vmlaq_u16(s0colsum0_h, s1colsum0_h, three_u16);
    uint16x8_t output1_p1_l = vmlaq_u16(s1colsum1_l, s0colsum1_l, three_u16);
    uint16x8_t output1_p1_h = vmlaq_u16(s1colsum1_h, s0colsum1_h, three_u16);
    uint16x8_t output1_p2_l = vmlaq_u16(s0colsum1_l, s1colsum1_l, three_u16);
    uint16x8_t output1_p2_h = vmlaq_u16(s0colsum1_h, s1colsum1_h, three_u16);

    output0_p1_l = vaddq_u16(output0_p1_l, seven_u16);
    output0_p1_h = vaddq_u16(output0_p1_h, seven_u16);
    output1_p1_l = vaddq_u16(output1_p1_l, seven_u16);
    output1_p1_h = vaddq_u16(output1_p1_h, seven_u16);

    uint8x16x2_t output_pixels0 = { {
      vcombine_u8(vshrn_n_u16(output0_p1_l, 4), vshrn_n_u16(output0_p1_h, 4)),
      vcombine_u8(vrshrn_n_u16(output0_p2_l, 4), vrshrn_n_u16(output0_p2_h, 4))
    } };
    uint8x16x2_t output_pixels1 = { {
      vcombine_u8(vshrn_n_u16(output1_p1_l, 4), vshrn_n_u16(output1_p1_h, 4)),
      vcombine_u8(vrshrn_n_u16(output1_p2_l, 4), vrshrn_n_u16(output1_p2_h, 4))
    } };

    vst2q_u8(outptr0 + 1, output_pixels0);
    vst2q_u8(outptr1 + 1, output_pixels1);

    for (colctr = 16; colctr < downsampled_width; colctr += 16) {

      s0A = vld1q_u8(inptr0 + colctr - 1);
      s0B = vld1q_u8(inptr1 + colctr - 1);
      s0C = vld1q_u8(inptr2 + colctr - 1);
      s0colsum0_l = vmlal_u8(vmovl_u8(vget_low_u8(s0A)), vget_low_u8(s0B),
                             three_u8);
      s0colsum0_h = vmlal_u8(vmovl_u8(vget_high_u8(s0A)), vget_high_u8(s0B),
                             three_u8);
      s0colsum1_l = vmlal_u8(vmovl_u8(vget_low_u8(s0C)), vget_low_u8(s0B),
                             three_u8);
      s0colsum1_h = vmlal_u8(vmovl_u8(vget_high_u8(s0C)), vget_high_u8(s0B),
                             three_u8);

      s1A = vld1q_u8(inptr0 + colctr);
      s1B = vld1q_u8(inptr1 + colctr);
      s1C = vld1q_u8(inptr2 + colctr);
      s1colsum0_l = vmlal_u8(vmovl_u8(vget_low_u8(s1A)), vget_low_u8(s1B),
                             three_u8);
      s1colsum0_h = vmlal_u8(vmovl_u8(vget_high_u8(s1A)), vget_high_u8(s1B),
                             three_u8);
      s1colsum1_l = vmlal_u8(vmovl_u8(vget_low_u8(s1C)), vget_low_u8(s1B),
                             three_u8);
      s1colsum1_h = vmlal_u8(vmovl_u8(vget_high_u8(s1C)), vget_high_u8(s1B),
                             three_u8);

      output0_p1_l = vmlaq_u16(s1colsum0_l, s0colsum0_l, three_u16);
      output0_p1_h = vmlaq_u16(s1colsum0_h, s0colsum0_h, three_u16);
      output0_p2_l = vmlaq_u16(s0colsum0_l, s1colsum0_l, three_u16);
      output0_p2_h = vmlaq_u16(s0colsum0_h, s1colsum0_h, three_u16);
      output1_p1_l = vmlaq_u16(s1colsum1_l, s0colsum1_l, three_u16);
      output1_p1_h = vmlaq_u16(s1colsum1_h, s0colsum1_h, three_u16);
      output1_p2_l = vmlaq_u16(s0colsum1_l, s1colsum1_l, three_u16);
      output1_p2_h = vmlaq_u16(s0colsum1_h, s1colsum1_h, three_u16);

      output0_p1_l = vaddq_u16(output0_p1_l, seven_u16);
      output0_p1_h = vaddq_u16(output0_p1_h, seven_u16);
      output1_p1_l = vaddq_u16(output1_p1_l, seven_u16);
      output1_p1_h = vaddq_u16(output1_p1_h, seven_u16);

      output_pixels0.val[0] = vcombine_u8(vshrn_n_u16(output0_p1_l, 4),
                                          vshrn_n_u16(output0_p1_h, 4));
      output_pixels0.val[1] = vcombine_u8(vrshrn_n_u16(output0_p2_l, 4),
                                          vrshrn_n_u16(output0_p2_h, 4));
      output_pixels1.val[0] = vcombine_u8(vshrn_n_u16(output1_p1_l, 4),
                                          vshrn_n_u16(output1_p1_h, 4));
      output_pixels1.val[1] = vcombine_u8(vrshrn_n_u16(output1_p2_l, 4),
                                          vrshrn_n_u16(output1_p2_h, 4));

      vst2q_u8(outptr0 + 2 * colctr - 1, output_pixels0);
      vst2q_u8(outptr1 + 2 * colctr - 1, output_pixels1);
    }

    int s1colsum0 = GETJSAMPLE(inptr1[downsampled_width - 1]) * 3 +
                    GETJSAMPLE(inptr0[downsampled_width - 1]);
    outptr0[2 * downsampled_width - 1] = (JSAMPLE)((s1colsum0 * 4 + 7) >> 4);
    int s1colsum1 = GETJSAMPLE(inptr1[downsampled_width - 1]) * 3 +
                    GETJSAMPLE(inptr2[downsampled_width - 1]);
    outptr1[2 * downsampled_width - 1] = (JSAMPLE)((s1colsum1 * 4 + 7) >> 4);
    inrow++;
  }
}

void h2v1_upsample(int max_v_samp_factor, JDIMENSION output_width,
                         JSAMPARRAY input_data, JSAMPARRAY *output_data_ptr)
{
  JSAMPARRAY output_data = *output_data_ptr;
  JSAMPROW inptr, outptr;
  int inrow;
  unsigned colctr;

  for (inrow = 0; inrow < max_v_samp_factor; inrow++) {
    inptr = input_data[inrow];
    outptr = output_data[inrow];
    for (colctr = 0; 2 * colctr < output_width; colctr += 16) {
      uint8x16_t samples = vld1q_u8(inptr + colctr);
      uint8x16x2_t output_pixels = { { samples, samples } };
      vst2q_u8(outptr + 2 * colctr, output_pixels);
    }
  }
}

void h2v2_upsample(int max_v_samp_factor, JDIMENSION output_width,
                         JSAMPARRAY input_data, JSAMPARRAY *output_data_ptr)
{
  JSAMPARRAY output_data = *output_data_ptr;
  JSAMPROW inptr, outptr0, outptr1;
  int inrow, outrow;
  unsigned colctr;

  for (inrow = 0, outrow = 0; outrow < max_v_samp_factor; inrow++) {
    inptr = input_data[inrow];
    outptr0 = output_data[outrow++];
    outptr1 = output_data[outrow++];

    for (colctr = 0; 2 * colctr < output_width; colctr += 16) {
      uint8x16_t samples = vld1q_u8(inptr + colctr);
      uint8x16x2_t output_pixels = { { samples, samples } };
      vst2q_u8(outptr0 + 2 * colctr, output_pixels);
      vst2q_u8(outptr1 + 2 * colctr, output_pixels);
    }
  }
}
