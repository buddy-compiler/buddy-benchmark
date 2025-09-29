#include <arm_neon.h>
#include <string.h>

int cvtBGRtoBGR(const unsigned char* src_data, size_t src_step,
                unsigned char* dst_data, size_t dst_step,
                int width, int height,
                int depth, int scn, int dcn)
{
    if (depth != 0)
        return 1;
    if (!((scn == 3 && dcn == 4) || (scn == 4 && dcn == 3) || (scn == dcn)))
        return 1;
    if (scn == dcn) {
        for (int i = 0; i < height; ++i) {
            const unsigned char* src = src_data + src_step * i;
            unsigned char* dst = dst_data + dst_step * i;
            memcpy(dst, src, width * scn);
        }
        return 0;
    }

    if (scn == 3 && dcn == 4) {
        size_t roiw8 = width >= 7 ? width - 7 : 0;
        size_t roiw16 = width >= 15 ? width - 15 : 0;
        union { uint8x16x4_t v4; uint8x16x3_t v3; } v_dst0;
        v_dst0.v4.val[3] = vdupq_n_u8(255);
        union { uint8x8x4_t v4; uint8x8x3_t v3; } v_dst;
        v_dst.v4.val[3] = vdup_n_u8(255);

        for (int i = 0; i < height; ++i) {
            const unsigned char* src = src_data + src_step * i;
            unsigned char* dst = dst_data + dst_step * i;
            size_t sj = 0u, dj = 0u, j = 0u;

            for (; j < roiw16; sj += 48, dj += 64, j += 16) {
                v_dst0.v3 = vld3q_u8(src + sj);
                vst4q_u8(dst + dj, v_dst0.v4);
            }

            if (j < roiw8) {
                v_dst.v3 = vld3_u8(src + sj);
                vst4_u8(dst + dj, v_dst.v4);
                sj += 24; dj += 32; j += 8;
            }

            for (; j < width; ++j, sj += 3, dj += 4) {
                dst[dj] = src[sj];
                dst[dj + 1] = src[sj + 1];
                dst[dj + 2] = src[sj + 2];
                dst[dj + 3] = 255;
            }
        }
        return 0;
    }

    if (scn == 4 && dcn == 3) {
        size_t roiw8 = width >= 7 ? width - 7 : 0;
        size_t roiw16 = width >= 15 ? width - 15 : 0;

        union { uint8x16x4_t v4; uint8x16x3_t v3; } v_dst0;
        union { uint8x8x4_t v4; uint8x8x3_t v3; } v_dst;

        for (int i = 0; i < height; ++i) {
            const unsigned char* src = src_data + src_step * i;
            unsigned char* dst = dst_data + dst_step * i;
            size_t sj = 0u, dj = 0u, j = 0u;

            for (; j < roiw16; sj += 64, dj += 48, j += 16) {
                v_dst0.v4 = vld4q_u8(src + sj);
                vst3q_u8(dst + dj, v_dst0.v3);
            }

            if (j < roiw8) {
                v_dst.v4 = vld4_u8(src + sj);
                vst3_u8(dst + dj, v_dst.v3);
                sj += 32; dj += 24; j += 8;
            }

            for (; j < width; ++j, sj += 4, dj += 3) {
                dst[dj] = src[sj];
                dst[dj + 1] = src[sj + 1];
                dst[dj + 2] = src[sj + 2];

            }
        }
        return 0;
    }

    return 1;
}
