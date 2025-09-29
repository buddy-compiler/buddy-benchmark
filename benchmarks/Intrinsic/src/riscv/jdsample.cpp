#include <riscv_vector.h>

#include <stdint.h>
#define GETJSAMPLE(value)  ((int)(value))
typedef unsigned int JDIMENSION;
typedef unsigned char JSAMPLE;
typedef JSAMPLE *JSAMPROW;
typedef JSAMPROW *JSAMPARRAY;
typedef JSAMPARRAY *JSAMPIMAGE;

void h2v1_fancy_upsample(int max_v_samp_factor,
                                   JDIMENSION downsampled_width,
                                   JSAMPARRAY input_data,
                                   JSAMPARRAY *output_data_ptr)
{
    JSAMPARRAY output_data = *output_data_ptr;
    JSAMPROW inptr, outptr;
    int inrow, incol;
    size_t vl;

    vuint8m4_t this_, shift, v0, v1;

    vuint16m8_t w0, w1;

    for (inrow = 0; inrow < max_v_samp_factor; inrow++)
    {
        inptr = input_data[inrow];
        outptr = output_data[inrow];

        *outptr++ = (JSAMPLE)GETJSAMPLE(*inptr);

        size_t VL = __riscv_vsetvlmax_e8m4();
        for (incol = downsampled_width - 1; incol > 0;
             incol -= vl, inptr += vl, outptr += 2 * vl)
        {
            vl = __riscv_vsetvl_e8m4(incol < VL ? incol : VL);

            this_ = __riscv_vle8_v_u8m4(inptr, vl);
            shift = __riscv_vle8_v_u8m4(inptr + 1, vl);

            w0 = __riscv_vwmaccu_vx_u16m8(__riscv_vwaddu_vx_u16m8(shift, 2, vl), 3, this_, vl);

            w1 = __riscv_vwmaccu_vx_u16m8(__riscv_vwaddu_vx_u16m8(this_, 1, vl), 3, shift, vl);

            v0 = __riscv_vnclipu_wx_u8m4(w0, 2, __RISCV_VXRM_RDN, vl);
            v1 = __riscv_vnclipu_wx_u8m4(w1, 2, __RISCV_VXRM_RDN, vl);

            __riscv_vsseg2e8_v_u8m4x2(outptr, __riscv_vcreate_v_u8m4x2(v0, v1), vl);
        }

        *outptr = (JSAMPLE)GETJSAMPLE(*inptr);
    }
}

void h2v2_fancy_upsample(int max_v_samp_factor,
                                   JDIMENSION downsampled_width,
                                   JSAMPARRAY input_data,
                                   JSAMPARRAY *output_data_ptr)
{
    JSAMPARRAY output_data = *output_data_ptr;
    JSAMPROW inptr_1, inptr0, inptr1, outptr0, outptr1;
    int inrow, outrow, incol;
    size_t vl;

    vuint8m1_t row_1, row0, row1, row_1rs1, row0rs1, row1rs1,
                p_odd0, p_even0, p_odd1, p_even1;

    vuint16m2_t row_1_u16, row_1rs1_u16, row0_u16, row0rs1_u16, row1_u16, row1rs1_u16,
                s0_add_3s1_ctr, s1_add_3s0_ctr, s0_add_3s1_sd, s1_add_3s0_sd,
                p_odd0_u16, p_even0_u16, p_odd1_u16, p_even1_u16;

    for (inrow = 0, outrow = 0; outrow < max_v_samp_factor; inrow++)
    {
        inptr_1 = input_data[inrow - 1];
        inptr0 = input_data[inrow];
        inptr1 = input_data[inrow + 1];
        outptr0 = output_data[outrow++];
        outptr1 = output_data[outrow++];

        int s0colsum0 = GETJSAMPLE(*inptr0) * 3 + GETJSAMPLE(*inptr_1);
        *outptr0++ = (JSAMPLE)((s0colsum0 * 4 + 8) >> 4);
        int s0colsum1 = GETJSAMPLE(*inptr0) * 3 + GETJSAMPLE(*inptr1);
        *outptr1++ = (JSAMPLE)((s0colsum1 * 4 + 8) >> 4);

        for (incol = downsampled_width - 1; incol > 0;
             incol -= vl, inptr_1 += vl, inptr0 += vl, inptr1 += vl,
             outptr0 += 2 * vl, outptr1 += 2 * vl)
        {
            vl = __riscv_vsetvl_e16m2(incol);

            row_1 = __riscv_vle8_v_u8m1(inptr_1, vl);
            row_1rs1 = __riscv_vle8_v_u8m1(inptr_1 + 1, vl);
            row0 = __riscv_vle8_v_u8m1(inptr0, vl);
            row0rs1 = __riscv_vle8_v_u8m1(inptr0 + 1, vl);
            row1 = __riscv_vle8_v_u8m1(inptr1, vl);
            row1rs1 = __riscv_vle8_v_u8m1(inptr1 + 1, vl);

            row_1_u16 = __riscv_vwaddu_vx_u16m2(row_1, 0, vl);
            row_1rs1_u16 = __riscv_vwaddu_vx_u16m2(row_1rs1, 0, vl);
            row0_u16 = __riscv_vwaddu_vx_u16m2(row0, 0, vl);
            row0rs1_u16 = __riscv_vwaddu_vx_u16m2(row0rs1, 0, vl);
            row1_u16 = __riscv_vwaddu_vx_u16m2(row1, 0, vl);
            row1rs1_u16 = __riscv_vwaddu_vx_u16m2(row1rs1, 0, vl);

            s1_add_3s0_sd = __riscv_vmul_vx_u16m2(row_1_u16, 3, vl);
            s1_add_3s0_sd = __riscv_vadd_vv_u16m2(s1_add_3s0_sd, row_1rs1_u16, vl);

            s0_add_3s1_sd = __riscv_vmul_vx_u16m2(row_1rs1_u16, 3, vl);
            s0_add_3s1_sd = __riscv_vadd_vv_u16m2(s0_add_3s1_sd, row_1_u16, vl);

            s1_add_3s0_ctr = __riscv_vmul_vx_u16m2(row0_u16, 3, vl);
            s1_add_3s0_ctr = __riscv_vadd_vv_u16m2(s1_add_3s0_ctr, row0rs1_u16, vl);
            s1_add_3s0_ctr = __riscv_vmul_vx_u16m2(s1_add_3s0_ctr, 3, vl);

            s0_add_3s1_ctr = __riscv_vmul_vx_u16m2(row0rs1_u16, 3, vl);
            s0_add_3s1_ctr = __riscv_vadd_vv_u16m2(s0_add_3s1_ctr, row0_u16, vl);
            s0_add_3s1_ctr = __riscv_vmul_vx_u16m2(s0_add_3s1_ctr, 3, vl);

            p_odd0_u16 = __riscv_vadd_vv_u16m2(s1_add_3s0_ctr, s1_add_3s0_sd, vl);
            p_odd0_u16 = __riscv_vadd_vx_u16m2(p_odd0_u16, 7, vl);

            p_even0_u16 = __riscv_vadd_vv_u16m2(s0_add_3s1_ctr, s0_add_3s1_sd, vl);
            p_even0_u16 = __riscv_vadd_vx_u16m2(p_even0_u16, 8, vl);

            p_odd0 = __riscv_vnsrl_wx_u8m1(p_odd0_u16, 4, vl);
            p_even0 = __riscv_vnsrl_wx_u8m1(p_even0_u16, 4, vl);

            __riscv_vsse8_v_u8m1(outptr0, 2 * sizeof(JSAMPLE), p_odd0, vl);
            __riscv_vsse8_v_u8m1(outptr0 + 1, 2 * sizeof(JSAMPLE), p_even0, vl);

            s1_add_3s0_sd = __riscv_vmul_vx_u16m2(row1_u16, 3, vl);
            s1_add_3s0_sd = __riscv_vadd_vv_u16m2(s1_add_3s0_sd, row1rs1_u16, vl);

            s0_add_3s1_sd = __riscv_vmul_vx_u16m2(row1rs1_u16, 3, vl);
            s0_add_3s1_sd = __riscv_vadd_vv_u16m2(s0_add_3s1_sd, row1_u16, vl);

            p_odd1_u16 = __riscv_vadd_vv_u16m2(s1_add_3s0_ctr, s1_add_3s0_sd, vl);
            p_odd1_u16 = __riscv_vadd_vx_u16m2(p_odd1_u16, 7, vl);

            p_even1_u16 = __riscv_vadd_vv_u16m2(s0_add_3s1_ctr, s0_add_3s1_sd, vl);
            p_even1_u16 = __riscv_vadd_vx_u16m2(p_even1_u16, 8, vl);

            p_odd1 = __riscv_vnsrl_wx_u8m1(p_odd1_u16, 4, vl);
            p_even1 = __riscv_vnsrl_wx_u8m1(p_even1_u16, 4, vl);

            __riscv_vsse8_v_u8m1(outptr1, 2 * sizeof(JSAMPLE), p_odd1, vl);
            __riscv_vsse8_v_u8m1(outptr1 + 1, 2 * sizeof(JSAMPLE), p_even1, vl);
        }

        int s1colsum0 = GETJSAMPLE(*inptr0) * 3 + GETJSAMPLE(*inptr_1);
        *outptr0 = (JSAMPLE)((s1colsum0 * 4 + 7) >> 4);
        int s1colsum1 = GETJSAMPLE(*inptr0) * 3 + GETJSAMPLE(*inptr1);
        *outptr1 = (JSAMPLE)((s1colsum1 * 4 + 7) >> 4);

    }
}

void h2v1_upsample(int max_v_samp_factor,
                             JDIMENSION output_width,
                             JSAMPARRAY input_data,
                             JSAMPARRAY *output_data_ptr)
{
    JSAMPARRAY output_data = *output_data_ptr;
    JSAMPROW inptr, outptr;
    int inrow, outcol;
    size_t vl;

    vuint8m8_t samples;

    for (inrow = 0; inrow < max_v_samp_factor; inrow++)
    {
        inptr = input_data[inrow];
        outptr = output_data[inrow];

        for (outcol = output_width; outcol > 0;
             outcol -= 2 * vl, inptr += vl, outptr += 2 * vl)
        {
            vl = __riscv_vsetvl_e8m8((outcol + 1) / 2);

            samples = __riscv_vle8_v_u8m8(inptr, vl);

            __riscv_vsse8_v_u8m8(outptr, 2 * sizeof(JSAMPLE), samples, vl);
            __riscv_vsse8_v_u8m8(outptr + 1, 2 * sizeof(JSAMPLE), samples, vl);
        }

    }
}

void h2v2_upsample(int max_v_samp_factor,
                             JDIMENSION output_width,
                             JSAMPARRAY input_data,
                             JSAMPARRAY *output_data_ptr)
{
    JSAMPARRAY output_data = *output_data_ptr;
    JSAMPROW inptr, outptr0, outptr1;
    int inrow, outrow, outcol;
    size_t vl;

    vuint8m8_t samples;

    for (inrow = 0, outrow = 0; outrow < max_v_samp_factor; inrow++)
    {
        inptr = input_data[inrow];
        outptr0 = output_data[outrow++];
        outptr1 = output_data[outrow++];

        for (outcol = output_width; outcol > 0;
             outcol -= 2 * vl, inptr += vl, outptr0 += 2 * vl, outptr1 += 2 * vl)
        {
            vl = __riscv_vsetvl_e8m8((outcol + 1) / 2);

            samples = __riscv_vle8_v_u8m8(inptr, vl);

            __riscv_vsse8_v_u8m8(outptr0, 2 * sizeof(JSAMPLE), samples, vl);
            __riscv_vsse8_v_u8m8(outptr0 + 1, 2 * sizeof(JSAMPLE), samples, vl);
            __riscv_vsse8_v_u8m8(outptr1, 2 * sizeof(JSAMPLE), samples, vl);
            __riscv_vsse8_v_u8m8(outptr1 + 1, 2 * sizeof(JSAMPLE), samples, vl);
        }
    }
}
