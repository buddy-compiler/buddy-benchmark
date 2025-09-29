#include <cstdint>
#include <algorithm>
static void cvtGraytoBGR_3(const unsigned char* src_data, size_t src_step,
                         unsigned char* dst_data, size_t dst_step,
                         int width, int height)
{
    for (int i = 0; i < height; ++i)
    {
        const unsigned char* src = src_data + i * src_step;
        unsigned char* dst = dst_data + i * dst_step;

        int j = 0;
        for (; j < width; ++j, dst += 3)
        {
            dst[0] = src[j];
            dst[1] = src[j];
            dst[2] = src[j];
        }
    }
}

static void cvtGraytoBGR_4(const unsigned char* src_data, size_t src_step,
                         unsigned char* dst_data, size_t dst_step,
                         int width, int height)
{
    for (int i = 0; i < height; ++i)
    {
        const unsigned char* src = src_data + i * src_step;
        unsigned char* dst = dst_data + i * dst_step;

        int j = 0;
        for (; j < width; ++j, dst += 4)
        {
            dst[0] = src[j];
            dst[1] = src[j];
            dst[2] = src[j];
            dst[3] = 255;
        }
    }
}
int cvtGraytoBGR(const unsigned char* src_data, size_t src_step,
                 unsigned char* dst_data, size_t dst_step,
                 int width, int height,
                 int depth, int scn)
{
    if (depth != 0)
        return 1;
    if (scn == 3) {
        cvtGraytoBGR_3(src_data, src_step, dst_data, dst_step, width, height);
    }
    else if (scn == 4) {
        cvtGraytoBGR_4(src_data, src_step, dst_data, dst_step, width, height);
    }
    else {
        return 1;
    }

    return 0;
}

