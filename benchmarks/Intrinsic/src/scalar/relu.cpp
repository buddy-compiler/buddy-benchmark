void relu(float* data, int channels, int size, float slope = 0.f)
{
    for (int q = 0; q < channels; q++)
    {
        float* ptr = data + q * size;
        for (int i = 0; i < size; i++)
        {
            if (ptr[i] < 0)
                ptr[i] = slope == 0.f ? 0.f : ptr[i] * slope;
        }
    }
}
