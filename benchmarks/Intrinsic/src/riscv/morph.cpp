#include <cstdint>
#include <algorithm>
#include <limits>
#include <cstring>
#include <cfloat>
#include <iostream>
#include <riscv_vector.h>
#define CV_HAL_MORPH_ERODE   0
#define CV_HAL_MORPH_DILATE  1
#define CV_HAL_ERROR_OK      0
#define CV_HAL_ERROR_NOT_IMPLEMENTED  1

#define CV_8UC1  0
template<int op> struct rvv_ops;
template<> struct rvv_ops<CV_HAL_MORPH_ERODE>
{
    static inline uint8_t init() { return std::numeric_limits<uint8_t>::max(); }
    static inline uint8_t mop(uint8_t a, uint8_t b) { return a < b ? a : b; }
    static inline vuint8m4_t vop(vuint8m4_t a, vuint8m4_t b, size_t c) { return __riscv_vminu(a, b, c); }
    static inline vuint8m4_t vop(vuint8m4_t a, uint8_t b, size_t c) { return __riscv_vminu(a, b, c); }
};
template<> struct rvv_ops<CV_HAL_MORPH_DILATE>
{
    static inline uint8_t init() { return std::numeric_limits<uint8_t>::min(); }
    static inline uint8_t mop(uint8_t a, uint8_t b) { return a > b ? a : b; }
    static inline vuint8m4_t vop(vuint8m4_t a, vuint8m4_t b, size_t c) { return __riscv_vmaxu(a, b, c); }
    static inline vuint8m4_t vop(vuint8m4_t a, uint8_t b, size_t c) { return __riscv_vmaxu(a, b, c); }
};
static inline int borderInterpolate(int p, int len, int borderType)
{
    if (borderType == 0) {
        return p < 0 || p >= len ? -1 : p;
    } else if (borderType == 4) {
        return p < 0 ? 0 : (p >= len ? len - 1 : p);
    }
    return 1;
}

template<int op>
static int morph_rvv_3x3(const uint8_t* src_data, size_t src_step, uint8_t* dst_data,
                         int width, int height, int full_width, int full_height,
                         int offset_x, int offset_y, const uint8_t* kernel_data, size_t kernel_step,
                         int anchor_x, int anchor_y, int borderType, const uint8_t* borderValue, int src_type)
{
    bool kernel[9];
    for (int i = 0; i < 9; i++)
    {
        kernel[i] = kernel_data[(i / 3) * kernel_step + i % 3] != 0;
    }

    constexpr int noval = std::numeric_limits<int>::max();
    auto access = [&](int x, int y) {
        int pi, pj;
        pi = borderInterpolate(offset_y + x - anchor_y, full_height, borderType);
        pj = borderInterpolate(offset_x + y - anchor_x, full_width, borderType);
        pi = pi < 0 ? noval : pi - offset_y;
        pj = pj < 0 ? noval : pj - offset_x;
        return std::make_pair(pi, pj);
    };

    const int left = anchor_x, right = width - (2 - anchor_x);

    for (int i = 0; i < height; i++)
    {
        if (left >= right)
        {
            for (int j = 0; j < width; j++)
            {
                if (src_type == CV_8UC1)
                {
                    uint8_t val = rvv_ops<op>::init();
                    for (int k = 0; k < 9; k++)
                    {
                        if (kernel[k])
                        {
                            auto p = access(i + k / 3, j + k % 3);
                            if (p.first != noval && p.second != noval)
                            {
                                val = rvv_ops<op>::mop(val, src_data[p.first * src_step + p.second]);
                            }
                            else
                            {
                                val = rvv_ops<op>::mop(val, borderValue[0]);
                            }
                        }
                    }
                    dst_data[i * width + j] = val;
                }
            }
        }
        else
        {
            for (int j = 0; j < left; j++)
            {
                if (src_type == CV_8UC1)
                {
                    uint8_t val = rvv_ops<op>::init();
                    for (int k = 0; k < 9; k++)
                    {
                        if (kernel[k])
                        {
                            auto p = access(i + k / 3, j + k % 3);
                            if (p.first != noval && p.second != noval)
                            {
                                val = rvv_ops<op>::mop(val, src_data[p.first * src_step + p.second]);
                            }
                            else
                            {
                                val = rvv_ops<op>::mop(val, borderValue[0]);
                            }
                        }
                    }
                    dst_data[i * width + j] = val;
                }
            }

            for (int j = right; j < width; j++)
            {
                if (src_type == CV_8UC1)
                {
                    uint8_t val = rvv_ops<op>::init();
                    for (int k = 0; k < 9; k++)
                    {
                        if (kernel[k])
                        {
                            auto p = access(i + k / 3, j + k % 3);
                            if (p.first != noval && p.second != noval)
                            {
                                val = rvv_ops<op>::mop(val, src_data[p.first * src_step + p.second]);
                            }
                            else
                            {
                                val = rvv_ops<op>::mop(val, borderValue[0]);
                            }
                        }
                    }
                    dst_data[i * width + j] = val;
                }
            }
            auto row0_pair = access(i, 0);
            auto row1_pair = access(i + 1, 0);
            auto row2_pair = access(i + 2, 0);

            const uint8_t* row0 = row0_pair.first == noval ? nullptr : src_data + row0_pair.first * src_step;
            const uint8_t* row1 = row1_pair.first == noval ? nullptr : src_data + row1_pair.first * src_step;
            const uint8_t* row2 = row2_pair.first == noval ? nullptr : src_data + row2_pair.first * src_step;

            if (src_type == CV_8UC1)
            {
                int vl;
                for (int j = left; j < right; j += vl)
                {
                    vl = __riscv_vsetvl_e8m4(right - j);
                    auto m0 = __riscv_vmv_v_x_u8m4(rvv_ops<op>::init(), vl);

                    auto loadsrc = [&](const uint8_t* row, bool k0, bool k1, bool k2) {
                        if (!row)
                        {
                            m0 = rvv_ops<op>::vop(m0, borderValue[0], vl);
                            return;
                        }

                        const uint8_t* extra = row + j - anchor_x;
                        auto v0 = __riscv_vle8_v_u8m4(extra, vl);

                        if (k0) m0 = rvv_ops<op>::vop(m0, v0, vl);
                        v0 = __riscv_vslide1down(v0, extra[vl], vl);
                        if (k1) m0 = rvv_ops<op>::vop(m0, v0, vl);
                        if (!k2) return;
                        v0 = __riscv_vslide1down(v0, extra[vl + 1], vl);
                        m0 = rvv_ops<op>::vop(m0, v0, vl);
                    };

                    loadsrc(row0, kernel[0], kernel[1], kernel[2]);
                    loadsrc(row1, kernel[3], kernel[4], kernel[5]);
                    loadsrc(row2, kernel[6], kernel[7], kernel[8]);
                    __riscv_vse8(dst_data + i * width + j, m0, vl);
                }
            }
        }
    }

    return CV_HAL_ERROR_OK;
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

    if (op == CV_HAL_MORPH_ERODE)
        return morph_rvv_3x3<CV_HAL_MORPH_ERODE>(src_data, src_step, dst_data,
                                                 width, height, width, height,
                                                 0, 0, kernel_data, 3,
                                                 anchor_x, anchor_y, borderType, &borderValue, CV_8UC1);
    else
        return morph_rvv_3x3<CV_HAL_MORPH_DILATE>(src_data, src_step, dst_data,
                                                  width, height, width, height,
                                                  0, 0, kernel_data, 3,
                                                  anchor_x, anchor_y, borderType, &borderValue, CV_8UC1);

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
