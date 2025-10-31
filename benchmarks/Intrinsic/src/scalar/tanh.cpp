#include <cmath>

void tanh(float* data, int channels, int size)
{
    for (int q = 0; q < channels; q++)
    {
        float* ptr = data + q * size;

        for (int i = 0; i < size; i++)
        {
            ptr[i] = tanhf(ptr[i]);
        }
    }

}