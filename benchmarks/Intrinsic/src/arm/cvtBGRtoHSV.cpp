#include <arm_neon.h>
#include <algorithm>
#include <cstdint>
static inline uint8_t saturate_cast_u8(int v)
{
    return (uint8_t)((unsigned)v <= 255 ? v : v > 0 ? 255 : 0);
}
static const uint8_t fastSaturate8u[] =
{
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14,  15,
     16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  31,
     32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47,
     48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,
     64,  65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,  79,
     80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,
     96,  97,  98,  99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
    112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127,
    128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143,
    144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159,
    160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175,
    176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191,
    192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207,
    208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223,
    224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239,
    240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255
};
static inline uint32x4_t vroundq_u32_f32(float32x4_t x)
{
    static const float32x4_t v_half = vdupq_n_f32(0.5f);
    return vcvtq_u32_f32(vaddq_f32(x, v_half));
}
static inline uint8x8x3_t convertToHSV_neon(const uint8x8_t vR, const uint8x8_t vG, const uint8x8_t vB, const int32_t hrange)
{
    const int32_t hsv_shift = 12;
    const float vsdiv_table = float(255 << hsv_shift);
    float vhdiv_table = float(hrange << hsv_shift);
    const int32_t vhrange = hrange;
    const int32_t v0 = int32_t(0);
    const int32_t vshift = int32_t(1 << (hsv_shift-1));
    const int32_t v6 = int32_t(6);
    uint8x8_t vMin = vmin_u8(vR, vG);
    uint8x8_t vMax = vmax_u8(vR, vG);

    uint16x8_t vR_u16 = vmovl_u8(vR);
    uint16x8_t vG_u16 = vmovl_u8(vG);

    vMax = vmax_u8(vMax, vB);
    vMin = vmin_u8(vMin, vB);
    uint16x8_t vB_u16 = vmovl_u8(vB);

    uint16x8_t vDiff = vsubl_u8(vMax, vMin);
    uint16x8_t vV = vmovl_u8(vMax);
    uint16x8_t vDiffx2 = vaddq_u16(vDiff, vDiff);
    uint32x4_t vDiffL = vmovl_u16(vget_low_u16(vDiff));
    uint32x4_t vDiffH = vmovl_u16(vget_high_u16(vDiff));
    uint16x8_t vVEqR = vceqq_u16(vR_u16, vV);
    uint16x8_t vVEqG = vceqq_u16(vG_u16, vV);

    int16x8_t vG_B = vsubq_s16(vreinterpretq_s16_u16(vG_u16), vreinterpretq_s16_u16(vB_u16));
    uint16x8_t vInvR = vmvnq_u16(vVEqR);
    int16x8_t vB_R = vsubq_s16(vreinterpretq_s16_u16(vB_u16), vreinterpretq_s16_u16(vR_u16));
    int16x8_t vR_G = vsubq_s16(vreinterpretq_s16_u16(vR_u16), vreinterpretq_s16_u16(vG_u16));

    uint16x8_t vMask2 = vandq_u16(vVEqG, vInvR);
    vR_u16 = vandq_u16(vreinterpretq_u16_s16(vG_B), vVEqR);
    int16x8_t vH2 = vaddq_s16(vB_R, vreinterpretq_s16_u16(vDiffx2));

    vVEqR = vmvnq_u16(vVEqG);
    vB_R = vaddq_s16(vreinterpretq_s16_u16(vDiffx2), vreinterpretq_s16_u16(vDiffx2));
    vG_B = vandq_s16(vreinterpretq_s16_u16(vInvR), vreinterpretq_s16_u16(vVEqR));
    vInvR = vandq_u16(vreinterpretq_u16_s16(vH2), vMask2);
    vR_G = vaddq_s16(vR_G, vB_R);
    int16x8_t vH = vaddq_s16(vreinterpretq_s16_u16(vR_u16), vreinterpretq_s16_u16(vInvR));

    uint32x4_t vV_L = vmovl_u16(vget_low_u16(vV));
    vR_G = vandq_s16(vR_G, vG_B);
    uint32x4_t vV_H = vmovl_u16(vget_high_u16(vV));
    int16x8_t vDiff4 = vaddq_s16(vH, vR_G);
    int32x4_t vc6 = vdupq_n_s32(v6);
    uint32x4_t vLine1 = vmulq_u32(vDiffL, vreinterpretq_u32_s32(vc6));
    uint32x4_t vLine2 = vmulq_u32(vDiffH, vreinterpretq_u32_s32(vc6));

    float32x4_t vF1 = vcvtq_f32_u32(vV_L);
    float32x4_t vF2 = vcvtq_f32_u32(vV_H);
    float32x4_t vHF1 = vcvtq_f32_u32(vLine1);
    float32x4_t vHF2 = vcvtq_f32_u32(vLine2);
    float32x4_t vXInv1 = vrecpeq_f32(vF1);
    float32x4_t vXInv2 = vrecpeq_f32(vF2);
    float32x4_t vXInv3 = vrecpeq_f32(vHF1);
    float32x4_t vXInv4 = vrecpeq_f32(vHF2);

    float32x4_t vSt1 = vrecpsq_f32(vXInv1, vF1);
    float32x4_t vSt2 = vrecpsq_f32(vXInv2, vF2);
    float32x4_t vSt3 = vrecpsq_f32(vXInv3, vHF1);
    float32x4_t vSt4 = vrecpsq_f32(vXInv4, vHF2);

    vF1 = vmulq_f32(vXInv1, vSt1);
    vF2 = vmulq_f32(vXInv2, vSt2);
    vHF1 = vmulq_f32(vXInv3, vSt3);
    vHF2 = vmulq_f32(vXInv4, vSt4);

    float32x4_t vDivTab = vdupq_n_f32(vsdiv_table);
    vSt1 = vmulq_f32(vF1, vDivTab);
    vSt2 = vmulq_f32(vF2, vDivTab);
    vDivTab = vdupq_n_f32(vhdiv_table);
    vSt3 = vmulq_f32(vHF1, vDivTab);
    vSt4 = vmulq_f32(vHF2, vDivTab);

    uint32x4_t vRes1 = vroundq_u32_f32(vSt1);
    uint32x4_t vRes2 = vroundq_u32_f32(vSt2);
    uint32x4_t vRes3 = vroundq_u32_f32(vSt3);
    uint32x4_t vRes4 = vroundq_u32_f32(vSt4);
    int32x4_t vH_L = vmovl_s16(vget_low_s16(vDiff4));
    int32x4_t vH_H = vmovl_s16(vget_high_s16(vDiff4));

    uint32x4_t vDiff_Res1 = vmulq_u32(vDiffL, vRes1);
    uint32x4_t vDiff_Res2 = vmulq_u32(vDiffH, vRes2);
    uint32x4_t vDiff_Res3 = vmulq_u32(vreinterpretq_u32_s32(vH_L), vRes3);
    uint32x4_t vDiff_Res4 = vmulq_u32(vreinterpretq_u32_s32(vH_H), vRes4);

    int32x4_t vShift = vdupq_n_s32(vshift);
    uint32x4_t vAddRes1 = vaddq_u32(vDiff_Res1, vreinterpretq_u32_s32(vShift));
    uint32x4_t vAddRes2 = vaddq_u32(vDiff_Res2, vreinterpretq_u32_s32(vShift));
    uint32x4_t vAddRes3 = vaddq_u32(vDiff_Res3, vreinterpretq_u32_s32(vShift));
    uint32x4_t vAddRes4 = vaddq_u32(vDiff_Res4, vreinterpretq_u32_s32(vShift));

    int16x4_t vShrRes1 = vshrn_n_s32(vreinterpretq_s32_u32(vAddRes1), 8);
    int16x4_t vShrRes2 = vshrn_n_s32(vreinterpretq_s32_u32(vAddRes2), 8);
    int16x4_t vShrRes3 = vshrn_n_s32(vreinterpretq_s32_u32(vAddRes3), 8);
    int16x4_t vShrRes4 = vshrn_n_s32(vreinterpretq_s32_u32(vAddRes4), 8);

    int16x8_t vc0 = vdupq_n_s16((int16_t)v0);
    int8x8_t vShrRes1_s8 = vshrn_n_s16(vcombine_s16(vShrRes1, vShrRes2), 4);
    uint16x8_t vCltRes_u16 = vcltq_s16(vcombine_s16(vShrRes3, vShrRes4), vc0);
    int8x8_t vShrRes2_s8 = vshrn_n_s16(vcombine_s16(vShrRes3, vShrRes4), 4);

    int8x8_t vCltRes_s8 = vmovn_s16(vreinterpretq_s16_u16(vCltRes_u16));
    int8x8_t vcHRange = vdup_n_s8((int8_t)vhrange);
    uint8x8_t vHResAdd = vand_u8(vreinterpret_u8_s8(vCltRes_s8), vreinterpret_u8_s8(vcHRange));
    int8x8_t vHRes = vadd_s8(vShrRes2_s8, vreinterpret_s8_u8(vHResAdd));

    uint8x8x3_t vHsv;
    vHsv.val[0] = vreinterpret_u8_s8(vHRes);
    vHsv.val[1] = vreinterpret_u8_s8(vShrRes1_s8);
    vHsv.val[2] = vMax;

    return vHsv;
}
static inline void convertToHSV_scalar(const int32_t r, const int32_t g, const int32_t b,
                                       const int32_t hrange, const int32_t hsv_shift,
                                       uint8_t* dst)
{
    int32_t h, s, v = b;
    int32_t vmin = b, diff;
    int32_t vr, vg;
    v += fastSaturate8u[g-v+256];
    v += fastSaturate8u[r-v+256];
    vmin -= fastSaturate8u[vmin-g+256];
    vmin -= fastSaturate8u[vmin-r+256];

    diff = v - vmin;
    vr = v == r ? -1 : 0;
    vg = v == g ? -1 : 0;
    s = (int32_t(diff * (255 << hsv_shift) * (1.0f/(float)v)) + (1 << (hsv_shift-1))) >> hsv_shift;
    h = (vr & (g - b)) + (~vr & ((vg & (b - r + 2 * diff)) + ((~vg) & (r - g + 4 * diff))));
    h = ((h * int32_t((hrange << hsv_shift)/(6.f*diff) + 0.5)) + (1 << (hsv_shift-1))) >> hsv_shift;
    h += h < 0 ? hrange : 0;

    dst[0] = saturate_cast_u8(h);
    dst[1] = (uint8_t)s;
    dst[2] = (uint8_t)v;
}
int cvtBGRtoHSV(int start, int end, const unsigned char * src, size_t src_step,
                unsigned char * dst, size_t dst_step, int width, int scn,
                bool swapBlue, bool isFullRange, bool isHSV)
{
    if (!src || !dst || width <= 0 || start < 0 || end <= start)
        return 1;
    const int32_t hrange = isFullRange ? 256 : 180;
    const int32_t hsv_shift = 12;
    size_t roiw8 = width >= 7 ? width - 7 : 0;
    for (int i = start; i < end; ++i)
    {
        const uint8_t * src_row = src + i * src_step;
        uint8_t * dst_row = dst + i * dst_step;
        size_t sj = 0u, dj = 0u, j = 0u;
        for (; j < roiw8; sj += 8 * scn, dj += 24, j += 8)
        {
            if (scn == 3)
            {
                uint8x8x3_t vRgb = vld3_u8(src_row + sj);

                uint8x8x3_t vHsv;
                if (swapBlue) {
                    vHsv = convertToHSV_neon(vRgb.val[0], vRgb.val[1], vRgb.val[2], hrange);
                } else {
                    vHsv = convertToHSV_neon(vRgb.val[2], vRgb.val[1], vRgb.val[0], hrange);
                }
                vst3_u8(dst_row + dj, vHsv);
            }
            else if (scn == 4)
            {
                uint8x8x4_t vRgba = vld4_u8(src_row + sj);

                uint8x8x3_t vHsv;
                if (swapBlue) {
                    vHsv = convertToHSV_neon(vRgba.val[0], vRgba.val[1], vRgba.val[2], hrange);
                } else {
                    vHsv = convertToHSV_neon(vRgba.val[2], vRgba.val[1], vRgba.val[0], hrange);
                }
                vst3_u8(dst_row + dj, vHsv);
            }
        }
        for (; j < width; ++j, sj += scn, dj += 3)
        {
            if (swapBlue) {
                convertToHSV_scalar(src_row[sj], src_row[sj+1], src_row[sj+2],
                                    hrange, hsv_shift, dst_row + dj);
            } else {
                convertToHSV_scalar(src_row[sj+2], src_row[sj+1], src_row[sj],
                                    hrange, hsv_shift, dst_row + dj);
            }
        }
    }

    return 0;
}
