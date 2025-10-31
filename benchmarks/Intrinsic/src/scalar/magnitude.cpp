#include <cmath>

void magnitude32f(const float* x, const float* y, float* mag, int len)
{
    for(int i = 0 ; i < len; i++ )
    {
        float x0 = x[i], y0 = y[i];
        mag[i] = std::sqrt(x0*x0 + y0*y0);
    }
}