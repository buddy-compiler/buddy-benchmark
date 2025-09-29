#include <cmath>

void sigmoid(float* data, int channels, int size) {
    for (int q = 0; q < channels; q++)
    {
        float* ptr = data + q * size;
        for (int i = 0; i < size; i++)
        {
            float v = ptr[i];
            v = std::min(v, 88.3762626647949f);
            v = std::max(v, -88.3762626647949f);
            ptr[i] = 1.f / (1.f + expf(-v));
        }
    }
}