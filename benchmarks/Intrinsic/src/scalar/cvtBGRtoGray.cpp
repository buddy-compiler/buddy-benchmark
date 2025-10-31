#include <cstdint>
#include <algorithm>
enum
{
    SHIFT = 15,
    SHIFT_DELTA = 1 << (SHIFT - 1),

    R2Y_BT601   = 9798,
    G2Y_BT601   = 19235,
    B2Y_BT601   = 3735,

    R2Y_BT709   = 3483,
    G2Y_BT709   = 11718,
    B2Y_BT709   = 1183,
};
template<typename T>
inline T saturate_cast(int val)
{
    return static_cast<T>(std::max(0, std::min(val, 255)));
}
template<int scn>
static void cvtBGRtoGray_(const unsigned char* src_data, size_t src_step,
                         unsigned char* dst_data, size_t dst_step,
                         int width, int height)
{
    const uint32_t R2Y = R2Y_BT601;
    const uint32_t G2Y = G2Y_BT601;
    const uint32_t B2Y = B2Y_BT601;

    for (int i = 0; i < height; ++i)
    {
        const unsigned char* src = src_data + i * src_step;
        unsigned char* dst = dst_data + i * dst_step;

        int j = 0;

        if (scn == 3) {
            for (; j < width; ++j, src += 3)
            {
                uint32_t val = src[0] * B2Y + src[1] * G2Y + src[2] * R2Y;
                dst[j] = saturate_cast<unsigned char>((val + SHIFT_DELTA) >> SHIFT);
            }
        }
        else if (scn == 4) {
            for (; j < width; ++j, src += 4)
            {
                uint32_t val = src[0] * B2Y + src[1] * G2Y + src[2] * R2Y;
                dst[j] = saturate_cast<unsigned char>((val + SHIFT_DELTA) >> SHIFT);
            }
        }
    }
}
int cvtBGRtoGray(const unsigned char* src_data, size_t src_step,
                 unsigned char* dst_data, size_t dst_step,
                 int width, int height,
                 int depth, int scn)
{
    if (depth != 0)
      return 1;
    if (scn == 3) {
        cvtBGRtoGray_<3>(src_data, src_step, dst_data, dst_step, width, height);
    }
    else if (scn == 4) {
        cvtBGRtoGray_<4>(src_data, src_step, dst_data, dst_step, width, height);
    }
    else {
      return 1;
    }

    return 0;
}
