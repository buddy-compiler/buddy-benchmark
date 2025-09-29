#include <cstdint>
#include <cstddef>
#include <algorithm>

using uchar = unsigned char;
using ushort = unsigned short;
int cvtBGRtoBGR5x5(const unsigned char * src_data, size_t src_step,
                   unsigned char * dst_data, size_t dst_step,
                   int width, int height,
                   int scn, bool swapBlue, int greenBits)
{
    if (!src_data || !dst_data || width <= 0 || height <= 0)
        return 1;

    if (scn != 3 && scn != 4)
        return 1;

    if (greenBits != 5 && greenBits != 6)
        return 1;
    int bidx = swapBlue ? 2 : 0;
    for (int i = 0; i < height; i++)
    {
        const unsigned char * src = src_data + i * src_step;
        unsigned char * dst = dst_data + i * dst_step;
        for (int j = 0; j < width; j++, src += scn, dst += sizeof(ushort))
        {
            uchar r = src[bidx^2];
            uchar g = src[1];
            uchar b = src[bidx];
            uchar a = scn == 4 ? src[3] : 0;

            ushort d;
            if (greenBits == 6)
            {
                d = (ushort)((b >> 3)|((g & ~3) << 3)|((r & ~7) << 8));
            }
            else
            {
                d = (ushort)((b >> 3)|((g & ~7) << 2)|((r & ~7) << 7)|(a ? 0x8000 : 0));
            }
            ((ushort*)dst)[0] = d;
        }
    }

    return 0;
}
