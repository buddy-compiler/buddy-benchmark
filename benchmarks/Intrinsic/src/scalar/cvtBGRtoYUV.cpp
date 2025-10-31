#include <cstdint>
#include <algorithm>
template<typename T>
inline T saturate_cast(int32_t val)
{
    if (val > 255) return 255;
    if (val < 0) return 0;
    return static_cast<T>(val);
}
static const int yuv_shift = 14;
static const int R2Y = 4899;
static const int G2Y = 9617;
static const int B2Y = 1868;
static const int YCRI = 11682;
static const int YCBI = 9241;
void cvtBGRtoYUV(const unsigned char* src_data, size_t src_step,
                 unsigned char* dst_data, size_t dst_step,
                 int width, int height, int depth, int scn)
{
    for (int i = 0; i < height; i++)
    {
        const unsigned char* src = src_data + i * src_step;
        unsigned char* dst = dst_data + i * dst_step;
        for (int j = 0; j < width; j++, src += scn, dst += 3)
        {
            int B = src[0];
            int G = src[1];
            int R = src[2];
            int Y = (B * B2Y + G * G2Y + R * R2Y + (1 << (yuv_shift - 1))) >> yuv_shift;
            int delta = 128 * (1 << yuv_shift);
            int Cr = ((R - Y) * YCRI + delta + (1 << (yuv_shift - 1))) >> yuv_shift;
            int Cb = ((B - Y) * YCBI + delta + (1 << (yuv_shift - 1))) >> yuv_shift;
            dst[0] = saturate_cast<unsigned char>(Y);
            dst[1] = saturate_cast<unsigned char>(Cr);
            dst[2] = saturate_cast<unsigned char>(Cb);
        }
    }
}
