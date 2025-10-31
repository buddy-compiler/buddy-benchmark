#include <arm_neon.h>
#include <cstdint>
#include <cstddef>
#include <algorithm>

using uchar = unsigned char;
using u8 = uint8_t;
using u16 = uint16_t;
template<typename T>
inline T saturate_cast(int val) {
    if (sizeof(T) == 1) {
        return static_cast<T>(std::max(0, std::min(255, val)));
    }
    return static_cast<T>(val);
}
inline uint8x16x2_t convertTo565_neon(const uint8x16_t& vR, const uint8x16_t& vG, const uint8x16_t& vB, int greenBits)
{
    uint8x16x2_t vResult;

    if (greenBits == 6) {
        vResult.val[1] = vsriq_n_u8(vB, vG, 5);
        vResult.val[0] = vshlq_n_u8(vG, 3);
        vResult.val[0] = vsriq_n_u8(vResult.val[0], vR, 3);
    } else {

        uint8x16_t vR_shifted = vshrq_n_u8(vR, 3);
        uint8x16_t vG_shifted = vshrq_n_u8(vG, 3);
        uint8x16_t vB_shifted = vshrq_n_u8(vB, 3);

        vResult.val[1] = vshlq_n_u8(vB_shifted, 2);
        vResult.val[1] = vorrq_u8(vResult.val[1], vshrq_n_u8(vG_shifted, 3));
        vResult.val[0] = vshlq_n_u8(vG_shifted, 5);
        vResult.val[0] = vorrq_u8(vResult.val[0], vR_shifted);
    }

    return vResult;
}
inline void convertTo565_scalar(const u16 R, const u16 G, const u16 B, u8 * dst, int greenBits)
{
    if (greenBits == 6) {
        *((u16*)dst) = (R >> 3)|((G&~3) << 3)|((B&~7) << 8);
    } else {
        *((u16*)dst) = (R >> 3)|((G&~7) << 2)|((B&~7) << 7);
    }
}

int cvtBGRtoBGR5x5(const uchar * src_data, size_t src_step,
                   uchar * dst_data, size_t dst_step,
                   int width, int height,
                   int scn, bool swapBlue, int greenBits)
{
    if (!src_data || !dst_data || width <= 0 || height <= 0)
        return 1;

    if (scn != 3 && scn != 4)
        return 1;

    if (greenBits != 5 && greenBits != 6)
        return 1;
    size_t roiw16 = width >= 15 ? width - 15 : 0;

    for (int i = 0; i < height; ++i)
    {
        const u8 * src = src_data + i * src_step;
        u8 * dst = dst_data + i * dst_step;
        size_t sj = 0u, dj = 0u, j = 0u;
        for (; j < roiw16; sj += scn * 16, dj += 32, j += 16)
        {
            uint8x16_t vR, vG, vB;

            if (scn == 3) {
                uint8x16x3_t vBgr = vld3q_u8(src + sj);
                if (swapBlue) {
                    vR = vBgr.val[0];
                    vG = vBgr.val[1];
                    vB = vBgr.val[2];
                } else {
                    vB = vBgr.val[0];
                    vG = vBgr.val[1];
                    vR = vBgr.val[2];
                }
            } else {
                uint8x16x4_t vBgra = vld4q_u8(src + sj);
                if (swapBlue) {
                    vR = vBgra.val[0];
                    vG = vBgra.val[1];
                    vB = vBgra.val[2];
                } else {
                    vB = vBgra.val[0];
                    vG = vBgra.val[1];
                    vR = vBgra.val[2];
                }
            }
            uint8x16x2_t vVal565 = convertTo565_neon(vB, vG, vR, greenBits);
            vst2q_u8(dst + dj, vVal565);
        }
        for (; j < width; ++j, sj += scn, dj += 2)
        {
            u8 b = src[sj];
            u8 g = src[sj + 1];
            u8 r = src[sj + 2];

            if (swapBlue) {
                std::swap(r, b);
            }

            convertTo565_scalar(b, g, r, dst + dj, greenBits);
        }
    }

    return 0;
}
int cvtBGRtoBGR565(const uchar * src_data, size_t src_step,
                   uchar * dst_data, size_t dst_step,
                   int width, int height, int scn, bool swapBlue)
{
    return cvtBGRtoBGR5x5(src_data, src_step, dst_data, dst_step,
                         width, height, scn, swapBlue, 6);
}
int cvtBGRtoBGR555(const uchar * src_data, size_t src_step,
                   uchar * dst_data, size_t dst_step,
                   int width, int height, int scn, bool swapBlue)
{
    return cvtBGRtoBGR5x5(src_data, src_step, dst_data, dst_step,
                         width, height, scn, swapBlue, 5);
}
