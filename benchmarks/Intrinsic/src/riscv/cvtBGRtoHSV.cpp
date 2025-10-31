#include <riscv_vector.h>
#include <algorithm>
#include <cstdint>
#include <climits>
#include <cfloat>
template<typename T>
static inline int cvtBGRtoHSV_impl(int start, int end, const T * src, size_t src_step, T * dst, size_t dst_step, int width, int scn, bool swapBlue, bool isFullRange, bool isHSV);

template<>
inline int cvtBGRtoHSV_impl<unsigned char>(int start, int end, const unsigned char * src, size_t src_step, unsigned char * dst, size_t dst_step, int width, int scn, bool swapBlue, bool isFullRange, bool isHSV)
{
    for (int i = start; i < end; i++)
    {
        int vl;
        for (int j = 0; j < width; j += vl)
        {
            vl = __riscv_vsetvl_e8m1(width - j);
            vint32m4_t b, g, r;
            if (scn == 3)
            {
                auto x = __riscv_vlseg3e8_v_u8m1x3(src + i * src_step + j * 3, vl);
                b = __riscv_vreinterpret_v_u32m4_i32m4(__riscv_vzext_vf4(__riscv_vget_v_u8m1x3_u8m1(x, 0), vl));
                g = __riscv_vreinterpret_v_u32m4_i32m4(__riscv_vzext_vf4(__riscv_vget_v_u8m1x3_u8m1(x, 1), vl));
                r = __riscv_vreinterpret_v_u32m4_i32m4(__riscv_vzext_vf4(__riscv_vget_v_u8m1x3_u8m1(x, 2), vl));
            }
            else
            {
                auto x = __riscv_vlseg4e8_v_u8m1x4(src + i * src_step + j * 4, vl);
                b = __riscv_vreinterpret_v_u32m4_i32m4(__riscv_vzext_vf4(__riscv_vget_v_u8m1x4_u8m1(x, 0), vl));
                g = __riscv_vreinterpret_v_u32m4_i32m4(__riscv_vzext_vf4(__riscv_vget_v_u8m1x4_u8m1(x, 1), vl));
                r = __riscv_vreinterpret_v_u32m4_i32m4(__riscv_vzext_vf4(__riscv_vget_v_u8m1x4_u8m1(x, 2), vl));
            }
            if (swapBlue)
            {
                auto t = b;
                b = r, r = t;
            }

            auto v = b, vmin = b;
            v = __riscv_vmax(v, g, vl);
            v = __riscv_vmax(v, r, vl);
            vmin = __riscv_vmin(vmin, g, vl);
            vmin = __riscv_vmin(vmin, r, vl);
            auto diff = __riscv_vsub(v, vmin, vl);

            vint32m4_t l, t;
            if (isHSV)
            {
                t = v;
            }
            else
            {
                l = __riscv_vdiv(__riscv_vadd(v, vmin, vl), 2, vl);
                t = __riscv_vmerge(__riscv_vrsub(__riscv_vadd(v, vmin, vl), UCHAR_MAX * 2, vl), __riscv_vadd(v, vmin, vl), __riscv_vmslt(l, UCHAR_MAX / 2 + 1, vl), vl);
            }
            auto s = __riscv_vssra(__riscv_vmul(diff, __riscv_vfcvt_x(__riscv_vfrdiv(__riscv_vfcvt_f(t, vl), 255 << 12, vl), vl), vl), 12, __RISCV_VXRM_RNU, vl);

            auto h = __riscv_vmadd(diff, 4, __riscv_vsub(r, g, vl), vl);
            h = __riscv_vmerge(h, __riscv_vmadd(diff, 2, __riscv_vsub(b, r, vl), vl), __riscv_vmseq(v, g, vl), vl);
            h = __riscv_vmerge(h, __riscv_vsub(g, b, vl), __riscv_vmseq(v, r, vl), vl);
            h = __riscv_vssra(__riscv_vmul(h, __riscv_vfcvt_x(__riscv_vfrdiv(__riscv_vfcvt_f(__riscv_vmul(diff, 6, vl), vl), isFullRange ? 256 << 12 : 180 << 12, vl), vl), vl), 12, __RISCV_VXRM_RNU, vl);
            h = __riscv_vadd_mu(__riscv_vmslt(h, 0, vl), h, h, isFullRange ? 256 : 180, vl);

            vuint8m1x3_t x{};
            x = __riscv_vset_v_u8m1_u8m1x3(x, 0, __riscv_vnclipu(__riscv_vnclipu(__riscv_vreinterpret_v_i32m4_u32m4(h), 0, __RISCV_VXRM_RNU, vl), 0, __RISCV_VXRM_RNU, vl));
            x = __riscv_vset_v_u8m1_u8m1x3(x, 1, __riscv_vncvt_x(__riscv_vncvt_x(__riscv_vreinterpret_v_i32m4_u32m4(isHSV ? s : l), vl), vl));
            x = __riscv_vset_v_u8m1_u8m1x3(x, 2, __riscv_vncvt_x(__riscv_vncvt_x(__riscv_vreinterpret_v_i32m4_u32m4(isHSV ? v : s), vl), vl));
            __riscv_vsseg3e8(dst + i * dst_step + j * 3, x, vl);
        }
    }

    return 0;
}

template<>
inline int cvtBGRtoHSV_impl<float>(int start, int end, const float * src, size_t src_step, float * dst, size_t dst_step, int width, int scn, bool swapBlue, bool, bool isHSV)
{
    src_step /= sizeof(float);
    dst_step /= sizeof(float);

    for (int i = start; i < end; i++)
    {
        int vl;
        for (int j = 0; j < width; j += vl)
        {
            vl = __riscv_vsetvl_e32m2(width - j);
            vfloat32m2_t b, g, r;
            if (scn == 3)
            {
                auto x = __riscv_vlseg3e32_v_f32m2x3(src + i * src_step + j * 3, vl);
                b = __riscv_vget_v_f32m2x3_f32m2(x, 0);
                g = __riscv_vget_v_f32m2x3_f32m2(x, 1);
                r = __riscv_vget_v_f32m2x3_f32m2(x, 2);
            }
            else
            {
                auto x = __riscv_vlseg4e32_v_f32m2x4(src + i * src_step + j * 4, vl);
                b = __riscv_vget_v_f32m2x4_f32m2(x, 0);
                g = __riscv_vget_v_f32m2x4_f32m2(x, 1);
                r = __riscv_vget_v_f32m2x4_f32m2(x, 2);
            }
            if (swapBlue)
            {
                auto t = b;
                b = r, r = t;
            }

            auto v = b, vmin = b;
            v = __riscv_vfmax(v, g, vl);
            v = __riscv_vfmax(v, r, vl);
            vmin = __riscv_vfmin(vmin, g, vl);
            vmin = __riscv_vfmin(vmin, r, vl);
            auto diff = __riscv_vfsub(v, vmin, vl);

            vfloat32m2_t l, t;
            if (isHSV)
            {
                t = __riscv_vfadd(__riscv_vfabs(v, vl), FLT_EPSILON, vl);
            }
            else
            {
                l = __riscv_vfmul(__riscv_vfadd(v, vmin, vl), 0.5f, vl);
                t = __riscv_vmerge(__riscv_vfrsub(__riscv_vfadd(v, vmin, vl), 2.0f, vl), __riscv_vfadd(v, vmin, vl), __riscv_vmflt(l, 0.5f, vl), vl);
            }
            auto s = __riscv_vfdiv(diff, t, vl);
            diff = __riscv_vfrdiv(__riscv_vfadd(diff, FLT_EPSILON, vl), 60.0f, vl);

            auto h = __riscv_vfmadd(__riscv_vfsub(r, g, vl), diff, __riscv_vfmv_v_f_f32m2(240.0f, vl), vl);
            h = __riscv_vmerge(h, __riscv_vfmadd(__riscv_vfsub(b, r, vl), diff, __riscv_vfmv_v_f_f32m2(120.0f, vl), vl), __riscv_vmfeq(v, g, vl), vl);
            h = __riscv_vmerge(h, __riscv_vfmul(__riscv_vfsub(g, b, vl), diff, vl), __riscv_vmfeq(v, r, vl), vl);
            h = __riscv_vfadd_mu(__riscv_vmflt(h, 0, vl), h, h, 360.0f, vl);

            vfloat32m2x3_t x{};
            x = __riscv_vset_v_f32m2_f32m2x3(x, 0, h);
            x = __riscv_vset_v_f32m2_f32m2x3(x, 1, isHSV ? s : l);
            x = __riscv_vset_v_f32m2_f32m2x3(x, 2, isHSV ? v : s);
            __riscv_vsseg3e32(dst + i * dst_step + j * 3, x, vl);
        }
    }

    return 0;
}
int cvtBGRtoHSV(int start, int end, const unsigned char * src, size_t src_step,
                unsigned char * dst, size_t dst_step, int width, int scn,
                bool swapBlue, bool isFullRange, bool isHSV)
{
    if (!src || !dst || width <= 0 || start < 0 || end <= start)
        return 1;

    if (scn != 3 && scn != 4)
        return 1;

    return cvtBGRtoHSV_impl<unsigned char>(start, end, src, src_step, dst, dst_step, width, scn, swapBlue, isFullRange, isHSV);
}
