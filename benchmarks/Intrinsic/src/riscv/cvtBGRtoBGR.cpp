#include <riscv_vector.h>
#include <string.h>

template<typename T> struct rvv;
template<> struct rvv<unsigned char>
{
    using T = vuint8m2_t;
    static inline size_t vsetvlmax() { return __riscv_vsetvlmax_e8m2(); }
    static inline size_t vsetvl(size_t a) { return __riscv_vsetvl_e8m2(a); }
    static inline void vlseg(const unsigned char* a, int b, T& c, T& d, T& e, T& f, size_t g)
    {
        if (b == 3)
        {
            auto x = __riscv_vlseg3e8_v_u8m2x3(a, g);
            c = __riscv_vget_v_u8m2x3_u8m2(x, 0), d = __riscv_vget_v_u8m2x3_u8m2(x, 1), e = __riscv_vget_v_u8m2x3_u8m2(x, 2);
        }
        else
        {
            auto x = __riscv_vlseg4e8_v_u8m2x4(a, g);
            c = __riscv_vget_v_u8m2x4_u8m2(x, 0), d = __riscv_vget_v_u8m2x4_u8m2(x, 1), e = __riscv_vget_v_u8m2x4_u8m2(x, 2), f = __riscv_vget_v_u8m2x4_u8m2(x, 3);
        }
    }
    static inline void vsseg(unsigned char* a, int b, T c, T d, T e, T f, size_t g)
    {
        if (b == 3)
        {
            vuint8m2x3_t x{};
            x = __riscv_vset_v_u8m2_u8m2x3(x, 0, c);
            x = __riscv_vset_v_u8m2_u8m2x3(x, 1, d);
            x = __riscv_vset_v_u8m2_u8m2x3(x, 2, e);
            __riscv_vsseg3e8(a, x, g);
        }
        else
        {
            vuint8m2x4_t x{};
            x = __riscv_vset_v_u8m2_u8m2x4(x, 0, c);
            x = __riscv_vset_v_u8m2_u8m2x4(x, 1, d);
            x = __riscv_vset_v_u8m2_u8m2x4(x, 2, e);
            x = __riscv_vset_v_u8m2_u8m2x4(x, 3, f);
            __riscv_vsseg4e8(a, x, g);
        }
    }
    static inline T vmv_v_x(unsigned char a, size_t b) { return __riscv_vmv_v_x_u8m2(a, b); }
};

template<typename T>
static inline int cvtBGRtoBGR_impl(int start, int end, const T * src, size_t src_step, T * dst, size_t dst_step, int width, int scn, int dcn, bool swapBlue)
{
    src_step /= sizeof(T);
    dst_step /= sizeof(T);

    if (scn == dcn && !swapBlue)
    {
        for (int i = start; i < end; i++)
            memcpy(dst + i * dst_step, src + i * src_step, sizeof(T) * width * scn);
    }
    else
    {
        auto alpha = rvv<T>::vmv_v_x(255, rvv<T>::vsetvlmax());
        for (int i = start; i < end; i++)
        {
            int vl;
            for (int j = 0; j < width; j += vl)
            {
                vl = rvv<T>::vsetvl(width - j);
                typename rvv<T>::T vec_srcB, vec_srcG, vec_srcR, vec_srcA{};
                rvv<T>::vlseg(src + i * src_step + j * scn, scn, vec_srcB, vec_srcG, vec_srcR, vec_srcA, vl);
                if (swapBlue)
                {
                    auto t = vec_srcB;
                    vec_srcB = vec_srcR, vec_srcR = t;
                }
                rvv<T>::vsseg(dst + i * dst_step + j * dcn, dcn, vec_srcB, vec_srcG, vec_srcR, scn == 3 && dcn == 4 ? alpha : vec_srcA, vl);
            }
        }
    }

    return 0;
}

int cvtBGRtoBGR(const unsigned char* src_data, size_t src_step,
                unsigned char* dst_data, size_t dst_step,
                int width, int height,
                int depth, int scn, int dcn)
{
    if (depth != 0)
        return 1;
    if (!((scn == 3 && dcn == 4) || (scn == 4 && dcn == 3) || (scn == dcn)))
        return 1;
    if (scn == dcn) {
        for (int i = 0; i < height; ++i) {
            const unsigned char* src = src_data + src_step * i;
            unsigned char* dst = dst_data + dst_step * i;
            memcpy(dst, src, width * scn);
        }
        return 0;
    }

    return cvtBGRtoBGR_impl<unsigned char>(0, height, reinterpret_cast<const unsigned char*>(src_data), src_step, reinterpret_cast<unsigned char*>(dst_data), dst_step, width, scn, dcn, false);
}
