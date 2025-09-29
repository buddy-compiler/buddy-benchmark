#include <cstdint>
#include <algorithm>
#include <limits>
#include <cstring>
#include <cfloat>
#include <iostream>
#define CV_HAL_MORPH_ERODE   0
#define CV_HAL_MORPH_DILATE  1
#define CV_HAL_ERROR_OK      0
#define CV_HAL_ERROR_NOT_IMPLEMENTED  1

#define CV_8UC1  0
#define CV_8UC4  4

template<typename T> struct MinOp
{
    T operator()(const T a, const T b) const { return std::min(a, b); }
};

template<typename T> struct MaxOp
{
    T operator()(const T a, const T b) const { return std::max(a, b); }
};
template<> inline uint8_t MinOp<uint8_t>::operator()(const uint8_t a, const uint8_t b) const {
    return std::min(a, b);
}
template<> inline uint8_t MaxOp<uint8_t>::operator()(const uint8_t a, const uint8_t b) const {
    return std::max(a, b);
}

int morph_3x3_u8(int op,
                 const uint8_t* src_data, size_t src_step,
                 uint8_t* dst_data, size_t dst_step,
                 int width, int height,
                 const uint8_t* kernel_data,
                 int anchor_x = 1, int anchor_y = 1,
                 int borderType = 0, uint8_t borderValue = 0)
{
    if (op != CV_HAL_MORPH_ERODE && op != CV_HAL_MORPH_DILATE)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    if (!src_data || !dst_data || !kernel_data)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    if (width <= 0 || height <= 0)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            uint8_t val = (op == CV_HAL_MORPH_ERODE) ? 255 : 0;
            int pixel_count = 0;

            for (int ky = 0; ky < 3; ky++)
            {
                for (int kx = 0; kx < 3; kx++)
                {
                    if (kernel_data[ky * 3 + kx])
                    {
                        int sy = y + ky - anchor_y;
                        int sx = x + kx - anchor_x;
                        uint8_t pixel;

                        if (sy >= 0 && sy < height && sx >= 0 && sx < width)
                        {
                            pixel = src_data[sy * src_step + sx];
                        }
                        else
                        {
                            if (borderType == 0)
                                pixel = borderValue;
                            else
                            {
                                int csy = sy < 0 ? 0 : (sy >= height ? height - 1 : sy);
                                int csx = sx < 0 ? 0 : (sx >= width ? width - 1 : sx);
                                pixel = src_data[csy * src_step + csx];
                            }
                        }

                        if (op == CV_HAL_MORPH_ERODE)
                            val = MinOp<uint8_t>()(val, pixel);
                        else
                            val = MaxOp<uint8_t>()(val, pixel);

                        pixel_count++;
                    }
                }
            }

            dst_data[y * dst_step + x] = val;
        }
    }
    return CV_HAL_ERROR_OK;

}
int morph(int op, int src_type, int dst_type,
          uint8_t* src_data, size_t src_step,
          uint8_t* dst_data, size_t dst_step,
          int width, int height,
          int roi_width, int roi_height, int roi_x, int roi_y,
          int roi_width2, int roi_height2, int roi_x2, int roi_y2,
          int kernel_type, uint8_t* kernel_data, size_t kernel_step,
          int kernel_width, int kernel_height, int anchor_x, int anchor_y,
          int borderType, const double borderValue[4], int iterations, bool isSubmatrix)
{
    if (iterations != 1 || kernel_width != 3 || kernel_height != 3)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    if (src_type != CV_8UC1 || dst_type != CV_8UC1)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    if (anchor_x < 0) anchor_x = 1;
    if (anchor_y < 0) anchor_y = 1;
    uint8_t borderV = (uint8_t)std::max(0.0, std::min(255.0, borderValue[0]));
    if (op == CV_HAL_MORPH_DILATE && borderValue[0] == DBL_MAX)
        borderV = 0;
    return morph_3x3_u8(op, src_data, src_step, dst_data, dst_step,
                        width, height, kernel_data,
                        anchor_x, anchor_y, borderType, borderV);
}
