#include <gtest/gtest.h>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <cmath>

extern void gelu(float* data, int channels, int size, bool fast_gelu);

std::vector<float> RandomData(int channels, int size)
{
    std::vector<float> data(channels * size);
    for (auto& v : data)
    {
        v = (float(rand()) / RAND_MAX - 0.5f) * 10.f;
    }
    return data;
}

inline float GELURef(float x)
{

    return 0.5f * x * (1.0f + std::erf(x * 0.70710678f));
}

inline float FastGELURef(float x)
{

    const float sqrt_2_over_pi = 0.79788456f;
    return 0.5f * x *
           (1.0f + std::tanh(sqrt_2_over_pi * (x + 0.044715f * x * x * x)));
}

std::vector<float> GELUReference(const std::vector<float>& input, bool fast)
{
    std::vector<float> output = input;
    for (auto& v : output)
    {
        v = fast ? FastGELURef(v) : GELURef(v);
    }
    return output;
}

void TestGELU(int channels, int size, bool fast)
{
    auto data = RandomData(channels, size);
    std::vector<float> expected = GELUReference(data, fast);

    gelu(data.data(), channels, size, fast);

    auto NearlyEqual = [](float a, float b, float epsilon) {
        if (a == b) return true;
        float diff = fabs(a - b);
        if (diff <= epsilon) return true;
        return diff < epsilon * std::max(fabs(a), fabs(b));
    };

    for (size_t i = 0; i < data.size(); ++i)
    {
        EXPECT_TRUE(NearlyEqual(data[i], expected[i], 0.001f))
            << "Mismatch at index " << i
            << ": got " << data[i] << ", expected " << expected[i];
    }
}

TEST(GELUTest, Channels1D)
{
    TestGELU(1, 128, false);
    TestGELU(1, 128, true);
    TestGELU(1, 124, false);
    TestGELU(1, 124, true);
    TestGELU(1, 127, false);
    TestGELU(1, 127, true);
    TestGELU(1, 120, false);
    TestGELU(1, 120, true);
}

TEST(GELUTest, Channels2D)
{
    TestGELU(13, 32, false);
    TestGELU(13, 32, true);
    TestGELU(15, 24, false);
    TestGELU(15, 24, true);
    TestGELU(17, 12, false);
    TestGELU(17, 12, true);
    TestGELU(19, 15, false);
    TestGELU(19, 15, true);
}

TEST(GELUTest, Channels3D)
{
    TestGELU(5, 7*24, false);
    TestGELU(5, 7*24, true);
    TestGELU(7, 9*12, false);
    TestGELU(7, 9*12, true);
    TestGELU(3, 5*13, false);
    TestGELU(3, 5*13, true);
}

TEST(GELUTest, Channels4D)
{
    TestGELU(24, 5*6*7, false);
    TestGELU(24, 5*6*7, true);
    TestGELU(12, 7*8*9, false);
    TestGELU(12, 7*8*9, true);
    TestGELU(13, 3*4*5, false);
    TestGELU(13, 3*4*5, true);
}
