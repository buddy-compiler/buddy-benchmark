#include <cmath>

void gelu(float* data, int channels, int size, bool fast_gelu)
{
    if (fast_gelu)
    {
        for (int q = 0; q < channels; q++)
        {
            float* ptr = data + q * size;

            for (int i = 0; i < size; i++)
            {

                ptr[i] = 0.5f * ptr[i] * (1.0f + tanhf(0.79788452f * (ptr[i] + 0.044715f * ptr[i] * ptr[i] * ptr[i])));
            }
        }
    }
    else
    {
        for (int q = 0; q < channels; q++)
        {
            float* ptr = data + q * size;

            for (int i = 0; i < size; i++)
            {

                ptr[i] = 0.5f * ptr[i] * erfcf(-0.70710678f * ptr[i]);
            }
        }
    }
}
