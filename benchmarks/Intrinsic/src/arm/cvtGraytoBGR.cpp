#include <arm_neon.h>
#include <cstdint>
#include <algorithm>
int cvtGraytoBGR(const unsigned char* src_data, size_t src_step,
                 unsigned char* dst_data, size_t dst_step,
                 int width, int height,
                 int depth, int scn)
{
    if (depth != 0)
        return 1;
    if (scn != 3 && scn != 4)
        return 1;
    size_t roiw16 = width >= 15 ? width - 15 : 0;
    size_t roiw8 = width >= 7 ? width - 7 : 0;

    for (size_t i = 0u; i < (size_t)height; ++i)
    {
        const unsigned char * src = src_data + i * src_step;
        unsigned char * dst = dst_data + i * dst_step;
        size_t sj = 0u, dj = 0u;

        if (scn == 3) {
            for (; sj < roiw16; sj += 16, dj += 48)
            {
                uint8x16x3_t vRgb1;
                vRgb1.val[0] = vld1q_u8(src + sj);

                vRgb1.val[1] = vRgb1.val[0];
                vRgb1.val[2] = vRgb1.val[0];

                vst3q_u8(dst + dj, vRgb1);
            }

            if (sj < roiw8)
            {
                uint8x8x3_t vRgb2;
                vRgb2.val[0] = vld1_u8(src + sj);
                vRgb2.val[1] = vRgb2.val[0];
                vRgb2.val[2] = vRgb2.val[0];

                vst3_u8(dst + dj, vRgb2);

                sj += 8; dj += 24;
            }
            for (; sj < (size_t)width; ++sj, dj += 3)
            {
                dst[dj] = src[sj];
                dst[dj+1] = src[sj];
                dst[dj+2] = src[sj];
            }
        }
        else if (scn == 4) {
            uint8x16x4_t vRgba;
            uint8x8x4_t  vRgba2;
            vRgba.val[3]  = vmovq_n_u8(255);
            vRgba2.val[3] = vget_low_u8(vRgba.val[3]);
            for (; sj < roiw16; sj += 16, dj += 64)
            {
                vRgba.val[0]  = vld1q_u8(src + sj);
                vRgba.val[1] = vRgba.val[0];
                vRgba.val[2] = vRgba.val[0];

                vst4q_u8(dst + dj, vRgba);
            }
            if (sj < roiw8)
            {
                vRgba2.val[0] = vld1_u8(src + sj);
                vRgba2.val[1] = vRgba2.val[0];
                vRgba2.val[2] = vRgba2.val[0];

                vst4_u8(dst + dj, vRgba2);
                sj += 8; dj += 32;
            }
            for (; sj < (size_t)width; ++sj, dj += 4)
            {
                dst[dj] = src[sj];
                dst[dj+1] = src[sj];
                dst[dj+2] = src[sj];
                dst[dj+3] = 255;
            }
        }
    }

    return 0;
}
