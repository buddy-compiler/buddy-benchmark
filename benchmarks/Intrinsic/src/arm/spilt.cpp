#include <arm_neon.h>

void split8u(const unsigned char* src, unsigned char** dst, int len, int cn)
{
    if (cn == 2) {

        int i = 0;
        for (; i <= len - 16; i += 16) {
            uint8x16x2_t v_src = vld2q_u8(src + i * 2);
            vst1q_u8(dst[0] + i, v_src.val[0]);
            vst1q_u8(dst[1] + i, v_src.val[1]);
        }
        if (i <= len - 8) {
            uint8x8x2_t v_src = vld2_u8(src + i * 2);
            vst1_u8(dst[0] + i, v_src.val[0]);
            vst1_u8(dst[1] + i, v_src.val[1]);
            i += 8;
        }
        for (; i < len; ++i) {
            dst[0][i] = src[i * 2 + 0];
            dst[1][i] = src[i * 2 + 1];
        }
    }
    else if (cn == 3) {

        int i = 0;
        for (; i <= len - 16; i += 16) {
            uint8x16x3_t v_src = vld3q_u8(src + i * 3);
            vst1q_u8(dst[0] + i, v_src.val[0]);
            vst1q_u8(dst[1] + i, v_src.val[1]);
            vst1q_u8(dst[2] + i, v_src.val[2]);
        }
        if (i <= len - 8) {
            uint8x8x3_t v_src = vld3_u8(src + i * 3);
            vst1_u8(dst[0] + i, v_src.val[0]);
            vst1_u8(dst[1] + i, v_src.val[1]);
            vst1_u8(dst[2] + i, v_src.val[2]);
            i += 8;
        }
        for (; i < len; ++i) {
            dst[0][i] = src[i * 3 + 0];
            dst[1][i] = src[i * 3 + 1];
            dst[2][i] = src[i * 3 + 2];
        }
    }
    else if (cn == 4) {

        int i = 0;
        for (; i <= len - 16; i += 16) {
            uint8x16x4_t v_src = vld4q_u8(src + i * 4);
            vst1q_u8(dst[0] + i, v_src.val[0]);
            vst1q_u8(dst[1] + i, v_src.val[1]);
            vst1q_u8(dst[2] + i, v_src.val[2]);
            vst1q_u8(dst[3] + i, v_src.val[3]);
        }
        if (i <= len - 8) {
            uint8x8x4_t v_src = vld4_u8(src + i * 4);
            vst1_u8(dst[0] + i, v_src.val[0]);
            vst1_u8(dst[1] + i, v_src.val[1]);
            vst1_u8(dst[2] + i, v_src.val[2]);
            vst1_u8(dst[3] + i, v_src.val[3]);
            i += 8;
        }
        for (; i < len; ++i) {
            dst[0][i] = src[i * 4 + 0];
            dst[1][i] = src[i * 4 + 1];
            dst[2][i] = src[i * 4 + 2];
            dst[3][i] = src[i * 4 + 3];
        }
    } else {
      for (int i = 0; i < len; ++i) {
        for (int ch = 0; ch < cn; ++ch) {
          dst[ch][i] = src[i * cn + ch];
        }
      }
    }
}
