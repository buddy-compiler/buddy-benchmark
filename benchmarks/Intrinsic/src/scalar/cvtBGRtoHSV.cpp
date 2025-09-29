#include <algorithm>
#include <cstdint>
#include <climits>
#include <cfloat>
#include <cstdlib>
#include <cmath>
enum
{
    hsv_shift = 12
};
static int sdiv_table[256];
static int hdiv_table180[256];
static int hdiv_table256[256];
static bool tables_initialized = false;

static void initializeTables()
{
    if (tables_initialized) return;
    sdiv_table[0] = hdiv_table180[0] = hdiv_table256[0] = 0;
    for (int i = 1; i < 256; i++)
    {
        sdiv_table[i] = (255 << hsv_shift) / i;
        hdiv_table180[i] = (180 << hsv_shift) / (6 * i);
        hdiv_table256[i] = (256 << hsv_shift) / (6 * i);
    }
    tables_initialized = true;
}
#define CV_CALC_MAX_8U(a, b) if((a) < (b)) (a) = (b)
#define CV_CALC_MIN_8U(a, b) if((a) > (b)) (a) = (b)
static inline int cvtBGRtoHSV_scalar_8u(int start, int end, const unsigned char * src, size_t src_step,
                                        unsigned char * dst, size_t dst_step, int width, int scn,
                                        bool swapBlue, bool isFullRange, bool isHSV)
{
    initializeTables();

    int bidx = swapBlue ? 2 : 0;
    int* hdiv_table = isFullRange ? hdiv_table256 : hdiv_table180;

    for (int i = start; i < end; i++)
    {
        const unsigned char* src_row = src + i * src_step;
        unsigned char* dst_row = dst + i * dst_step;

        for (int j = 0; j < width; j++, src_row += scn, dst_row += 3)
        {
            int b = src_row[bidx], g = src_row[1], r = src_row[bidx^2];
            int h, s, v = b;
            int vmin = b;
            int vr, vg;
            CV_CALC_MAX_8U( v, g );
            CV_CALC_MAX_8U( v, r );
            CV_CALC_MIN_8U( vmin, g );
            CV_CALC_MIN_8U( vmin, r );

            unsigned char diff = (unsigned char)(v - vmin);
            vr = v == r ? -1 : 0;
            vg = v == g ? -1 : 0;
            s = (diff * sdiv_table[v] + (1 << (hsv_shift-1))) >> hsv_shift;
            h = (vr & (g - b)) +
                (~vr & ((vg & (b - r + 2 * diff)) + ((~vg) & (r - g + 4 * diff))));
            h = (h * hdiv_table[diff] + (1 << (hsv_shift-1))) >> hsv_shift;
            dst_row[0] = (unsigned char)h;
            dst_row[1] = (unsigned char)s;
            dst_row[2] = (unsigned char)v;
        }
    }

    return 0;
}
static inline int cvtBGRtoHSV_scalar_32f(int start, int end, const float * src, size_t src_step,
                                         float * dst, size_t dst_step, int width, int scn,
                                         bool swapBlue, bool)
{
    int bidx = swapBlue ? 2 : 0;
    src_step /= sizeof(float);
    dst_step /= sizeof(float);

    for (int i = start; i < end; i++)
    {
        const float* src_row = src + i * src_step;
        float* dst_row = dst + i * dst_step;

        for (int j = 0; j < width; j++, src_row += scn, dst_row += 3)
        {
            float b = src_row[bidx], g = src_row[1], r = src_row[bidx^2];
            float h, s, v;
            float vmin, diff;
            v = vmin = r;
            if( v < g ) v = g;
            if( v < b ) v = b;
            if( vmin > g ) vmin = g;
            if( vmin > b ) vmin = b;

            diff = v - vmin;
            s = diff/(float)(fabs(v) + FLT_EPSILON);
            diff = (float)(60./(diff + FLT_EPSILON));
            if( v == r )
                h = (g - b)*diff;
            else if( v == g )
                h = (b - r)*diff + 120.f;
            else
                h = (r - g)*diff + 240.f;

            if( h < 0 ) h += 360.f;
            dst_row[0] = h;
            dst_row[1] = s;
            dst_row[2] = v;
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

    return cvtBGRtoHSV_scalar_8u(start, end, src, src_step, dst, dst_step, width, scn, swapBlue, isFullRange, isHSV);
}
