#include <arm_neon.h>
#include <stdint.h>

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <limits>
#include <vector>
#define MORPH_ERODE 0
#define MORPH_DILATE 1
#define BORDER_CONSTANT 0
#define BORDER_REPLICATE 4
inline void prefetch(const void* ptr) {
#if defined __GNUC__
  __builtin_prefetch(reinterpret_cast<const char*>(ptr));
#else
  (void)ptr;
#endif
}
template <typename T>
inline T* getRowPtr(T* base, ptrdiff_t stride, size_t row) {
  char* baseRaw = const_cast<char*>(reinterpret_cast<const char*>(base));
  return reinterpret_cast<T*>(baseRaw + ptrdiff_t(row) * stride);
}
struct ErodeVecOp {
  inline uint8x16_t operator()(uint8x16_t a, uint8x16_t b) const {
    return vminq_u8(a, b);
  }

  inline uint8x8_t operator()(uint8x8_t a, uint8x8_t b) const {
    return vmin_u8(a, b);
  }

  inline uint8_t operator()(uint8_t a, uint8_t b) const {
    return std::min(a, b);
  }
};
struct DilateVecOp {
  inline uint8x16_t operator()(uint8x16_t a, uint8x16_t b) const {
    return vmaxq_u8(a, b);
  }

  inline uint8x8_t operator()(uint8x8_t a, uint8x8_t b) const {
    return vmax_u8(a, b);
  }

  inline uint8_t operator()(uint8_t a, uint8_t b) const {
    return std::max(a, b);
  }
};

template <typename VecOp>
void morph_kernel_neon_impl(int width, int height, const uint8_t* srcBase,
                            ptrdiff_t srcStride, uint8_t* dstBase,
                            ptrdiff_t dstStride, const uint8_t* kernel,
                            int borderType, uint8_t borderValue,
                            const VecOp& op) {
  std::vector<std::pair<int, int>> kernel_positions;
  for (int ky = 0; ky < 3; ky++) {
    for (int kx = 0; kx < 3; kx++) {
      if (kernel[ky * 3 + kx] != 0) {
        kernel_positions.push_back({ky - 1, kx - 1});
      }
    }
  }

  if (kernel_positions.empty()) {
    for (int y = 0; y < height; y++) {
      uint8_t* drow = getRowPtr(dstBase, dstStride, y);
      memset(drow, borderValue, width);
    }
    return;
  }

  for (int y = 0; y < height; ++y) {
    const uint8_t* srow = getRowPtr(srcBase, srcStride, y);
    uint8_t* drow = getRowPtr(dstBase, dstStride, y);

    int x = 0;
    const int simd_width = 16;
    for (; x <= width - simd_width; x += simd_width) {
      uint8x16_t result;
      bool first = true;
      for (const auto& pos : kernel_positions) {
        int sy = y + pos.first;
        int sx_base = x + pos.second;
        if (sy < 0 || sy >= height) {
          if (borderType == BORDER_CONSTANT) {
            uint8x16_t border_vec = vdupq_n_u8(
                borderValue);
            if (first) {
              result = border_vec;
              first = false;
            } else {
              result = op(result, border_vec);
            }
            continue;
          } else if (borderType == BORDER_REPLICATE) {
            sy = std::max(0, std::min(sy, height - 1));
          }
        }

        const uint8_t* sptr = getRowPtr(srcBase, srcStride, sy);
        uint8x16_t data;
        bool all_in_bounds = (sx_base >= 0 && sx_base + simd_width <= width);

        if (all_in_bounds) {
          data =
              vld1q_u8(sptr + sx_base);
          prefetch(sptr + sx_base);
        } else {
          uint8_t temp[simd_width];
          for (int i = 0; i < simd_width; i++) {
            int sx = sx_base + i;
            if (sx < 0 || sx >= width) {
              if (borderType == BORDER_CONSTANT) {
                temp[i] = borderValue;
              } else if (borderType == BORDER_REPLICATE) {
                sx = std::max(0, std::min(sx, width - 1));
                temp[i] = sptr[sx];
              }
            } else {
              temp[i] = sptr[sx];
            }
          }
          data = vld1q_u8(temp);
        }

        if (first) {
          result = data;
          first = false;
        } else {
          result = op(result, data);
        }
      }

      vst1q_u8(drow + x, result);
    }
    for (; x < width; ++x) {
      uint8_t pixel_result;
      bool first = true;

      for (const auto& pos : kernel_positions) {
        int sy = y + pos.first;
        int sx = x + pos.second;

        uint8_t val;
        if (sy < 0 || sy >= height || sx < 0 || sx >= width) {
          if (borderType == BORDER_CONSTANT) {
            val = borderValue;
          } else if (borderType == BORDER_REPLICATE) {
            sy = std::max(0, std::min(sy, height - 1));
            sx = std::max(0, std::min(sx, width - 1));
            const uint8_t* sptr = getRowPtr(srcBase, srcStride, sy);
            val = sptr[sx];
          }
        } else {
          const uint8_t* sptr = getRowPtr(srcBase, srcStride, sy);
          val = sptr[sx];
        }

        if (first) {
          pixel_result = val;
          first = false;
        } else {
          pixel_result = op(pixel_result, val);
        }
      }

      drow[x] = pixel_result;
    }
  }
}
void erode_kernel_neon(int width, int height, const uint8_t* srcBase,
                       ptrdiff_t srcStride, uint8_t* dstBase,
                       ptrdiff_t dstStride, const uint8_t* kernel,
                       int borderType, uint8_t borderValue) {
  ErodeVecOp op;
  uint8_t effective_border_value = borderValue;
  if (borderType == BORDER_REPLICATE) {
    effective_border_value = std::numeric_limits<uint8_t>::max();
  }

  morph_kernel_neon_impl(width, height, srcBase, srcStride, dstBase, dstStride,
                         kernel, borderType, effective_border_value, op);
}
void dilate_kernel_neon(int width, int height, const uint8_t* srcBase,
                        ptrdiff_t srcStride, uint8_t* dstBase,
                        ptrdiff_t dstStride, const uint8_t* kernel,
                        int borderType, uint8_t borderValue) {
  DilateVecOp op;
  uint8_t effective_border_value = borderValue;
  if (borderType == BORDER_REPLICATE) {
    effective_border_value = std::numeric_limits<uint8_t>::min();
  }

  morph_kernel_neon_impl(width, height, srcBase, srcStride, dstBase, dstStride,
                         kernel, borderType, effective_border_value, op);
}
int morph(int op, int src_type, int dst_type, uint8_t* src_data,
          size_t src_step, uint8_t* dst_data, size_t dst_step, int width,
          int height, int roi_width, int roi_height, int roi_x, int roi_y,
          int roi_width2, int roi_height2, int roi_x2, int roi_y2,
          int kernel_type, uint8_t* kernel_data, size_t kernel_step,
          int kernel_width, int kernel_height, int anchor_x, int anchor_y,
          int borderType, const double borderValue[4], int iterations,
          bool isSubmatrix) {
  if (src_type != 0 || dst_type != 0)
    return 1;

  if (iterations != 1 || kernel_width != 3 || kernel_height != 3) return 1;

  if (borderType != BORDER_CONSTANT && borderType != BORDER_REPLICATE) return 1;

  if (op != MORPH_ERODE && op != MORPH_DILATE) return 1;

  uint8_t bValue = (uint8_t)(borderValue[0]);

  if (op == MORPH_ERODE) {
    erode_kernel_neon(width, height, src_data, src_step, dst_data, dst_step,
                      kernel_data, borderType, bValue);
  } else
  {
    dilate_kernel_neon(width, height, src_data, src_step, dst_data, dst_step,
                       kernel_data, borderType, bValue);
  }

  return 0;
}
