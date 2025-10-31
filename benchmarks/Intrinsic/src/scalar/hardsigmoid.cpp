#include <cmath>

void hardsigmoid(float* data, int channels, int size, float alpha, float beta)
{
    float lower = -beta / alpha;
    float upper = (1.f / alpha) + lower;
    for (int q = 0; q < channels; q++)
    {
        float* ptr = data + q * size;

        for (int i = 0; i < size; i++)
        {
            if (ptr[i] < lower)
                ptr[i] = 0.f;
            else if (ptr[i] > upper)
                ptr[i] = 1.f;
            else
                ptr[i] = ptr[i] * alpha + beta;
        }
    }
}
