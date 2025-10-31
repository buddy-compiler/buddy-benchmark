#include <gtest/gtest.h>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <cmath>

extern void swish(float* data, int channels, int size);

inline std::vector<float> RandomData(int channels, int size)
{
    std::vector<float> data(channels * size);
    for (auto& v : data)
    {
        v = (float(rand()) / RAND_MAX - 0.5f) * 10.f;
    }
    return data;
}

std::vector<float> SwishReference(const std::vector<float>& input)
{
    std::vector<float> output = input;
    for (auto& v : output)
    {
        float sigmoid = 1.f / (1.f + std::exp(-v));
        v = v * sigmoid;
    }
    return output;
}

void TestSwish(int channels, int size)
{
    auto data = RandomData(channels, size);
    std::vector<float> expected = SwishReference(data);

    swish(data.data(), channels, size);

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

TEST(SwishTest, Channels1D)
{
    TestSwish(1, 128);
    TestSwish(1, 124);
    TestSwish(1, 127);
}

TEST(SwishTest, Channels2D)
{
    TestSwish(15, 24);
    TestSwish(17, 12);
    TestSwish(19, 15);
}

TEST(SwishTest, Channels3D)
{
    TestSwish(5, 7*24);
    TestSwish(7, 9*12);
    TestSwish(3, 5*13);
}

TEST(SwishTest, Channels4D)
{
    TestSwish(24, 5*6*7);
    TestSwish(12, 7*8*9);
    TestSwish(13, 3*4*5);
}