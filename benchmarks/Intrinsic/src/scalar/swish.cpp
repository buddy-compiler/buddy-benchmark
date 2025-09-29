#include <cmath>

void swish(float* data, int channels, int size)
{
    for (int q = 0; q < channels; q++)
    {
        float* ptr = data + q * size;

        for (int i = 0; i < size; i++)
        {
            float x = ptr[i];
            ptr[i] = x / (1.f + expf(-x));
        }
    }
}