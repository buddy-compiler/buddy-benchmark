#include <arm_neon.h>

void merge8u(const unsigned char** src, unsigned char* dst, int len, int cn)
{
    if (cn == 2) {

        int i = 0;
        for (; i <= len - 16; i += 16) {
            uint8x16x2_t v_dst;
            v_dst.val[0] = vld1q_u8(src[0] + i);
            v_dst.val[1] = vld1q_u8(src[1] + i);
            vst2q_u8(dst + i * 2, v_dst);
        }
        if (i <= len - 8) {
            uint8x8x2_t v_dst;
            v_dst.val[0] = vld1_u8(src[0] + i);
            v_dst.val[1] = vld1_u8(src[1] + i);
            vst2_u8(dst + i * 2, v_dst);
            i += 8;
        }
        for (; i < len; ++i) {
            dst[i * 2 + 0] = src[0][i];
            dst[i * 2 + 1] = src[1][i];
        }
    }
    else if (cn == 3) {

        int i = 0;
        for (; i <= len - 16; i += 16) {
            uint8x16x3_t v_dst;
            v_dst.val[0] = vld1q_u8(src[0] + i);
            v_dst.val[1] = vld1q_u8(src[1] + i);
            v_dst.val[2] = vld1q_u8(src[2] + i);

            vst3q_u8(dst + i * 3, v_dst);
        }
        if (i <= len - 8) {
            uint8x8x3_t v_dst;
            v_dst.val[0] = vld1_u8(src[0] + i);
            v_dst.val[1] = vld1_u8(src[1] + i);
            v_dst.val[2] = vld1_u8(src[2] + i);

            vst3_u8(dst + i * 3, v_dst);
            i += 8;
        }
        for (; i < len; ++i) {
            dst[i * 3 + 0] = src[0][i];
            dst[i * 3 + 1] = src[1][i];
            dst[i * 3 + 2] = src[2][i];
        }
    }
    else if (cn == 4) {

        int i = 0;
        for (; i <= len - 16; i += 16) {
            uint8x16x4_t v_dst;
            v_dst.val[0] = vld1q_u8(src[0] + i);
            v_dst.val[1] = vld1q_u8(src[1] + i);
            v_dst.val[2] = vld1q_u8(src[2] + i);
            v_dst.val[3] = vld1q_u8(src[3] + i);

            vst4q_u8(dst + i * 4, v_dst);
        }
        if (i <= len - 8) {
            uint8x8x4_t v_dst;
            v_dst.val[0] = vld1_u8(src[0] + i);
            v_dst.val[1] = vld1_u8(src[1] + i);
            v_dst.val[2] = vld1_u8(src[2] + i);
            v_dst.val[3] = vld1_u8(src[3] + i);

            vst4_u8(dst + i * 4, v_dst);
            i += 8;
        }
        for (; i < len; ++i) {
            dst[i * 4 + 0] = src[0][i];
            dst[i * 4 + 1] = src[1][i];
            dst[i * 4 + 2] = src[2][i];
            dst[i * 4 + 3] = src[3][i];
        }
    }
    else {
        for (int i = 0; i < len; ++i) {
            for (int ch = 0; ch < cn; ++ch) {
                dst[i * cn + ch] = src[ch][i];
            }
        }
    }
}
