#include <string.h>
#include <algorithm>

template<typename _Tp>
struct RGB2RGB_scalar
{
    typedef _Tp channel_type;

    RGB2RGB_scalar(int _srccn, int _dstcn, int _blueIdx) :
        srccn(_srccn), dstcn(_dstcn), blueIdx(_blueIdx)
    {}

    void operator()(const _Tp* src, _Tp* dst, int n) const
    {
        int scn = srccn, dcn = dstcn, bi = blueIdx;
        int i = 0;
        _Tp alphav = 255;

        for ( ; i < n; i++, src += scn, dst += dcn )
        {
            _Tp t0 = src[0], t1 = src[1], t2 = src[2];
            dst[bi  ] = t0;
            dst[1]    = t1;
            dst[bi^2] = t2;
            if(dcn == 4)
            {
                _Tp d = scn == 4 ? src[3] : alphav;
                dst[3] = d;
            }
        }
    }

    int srccn, dstcn, blueIdx;
};

template<typename _Tp>
static void cvtBGRtoBGR_impl(const _Tp* src_data, size_t src_step,
                             _Tp* dst_data, size_t dst_step,
                             int width, int height,
                             int scn, int dcn, bool swapBlue)
{
    int blueIdx = swapBlue ? 2 : 0;

    RGB2RGB_scalar<_Tp> converter(scn, dcn, blueIdx);
    for (int y = 0; y < height; ++y) {
        const _Tp* src_row = reinterpret_cast<const _Tp*>(
            reinterpret_cast<const unsigned char*>(src_data) + y * src_step);
        _Tp* dst_row = reinterpret_cast<_Tp*>(
            reinterpret_cast<unsigned char*>(dst_data) + y * dst_step);

        converter(src_row, dst_row, width);
    }
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
    cvtBGRtoBGR_impl<unsigned char>(src_data, src_step, dst_data, dst_step,
                                    width, height, scn, dcn, false);

    return 0;
}
