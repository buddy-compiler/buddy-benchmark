void hardswish(float* data, int channels, int size, float alpha, float beta)
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
                ;
            else
                ptr[i] = ptr[i] * (ptr[i] * alpha + beta);
        }
    }
}