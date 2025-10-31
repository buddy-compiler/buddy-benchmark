#include <cmath>

void selu(float* data, int channels, int size, float alpha, float lambda)
{
    float alphaxlambda = alpha * lambda;
    for (int q = 0; q < channels; q++)
    {
        float* ptr = data + q * size;

        for (int i = 0; i < size; i++)
        {
            if (ptr[i] < 0.f)
                ptr[i] = (expf(ptr[i]) - 1.f) * alphaxlambda;
            else
                ptr[i] *= lambda;
        }
    }
}
