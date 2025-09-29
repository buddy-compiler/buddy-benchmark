#include <gtest/gtest.h>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <ctime>

extern void relu(float* data, int channels, int size, float slope = 0.f);

inline std::vector<float> RandomData(int channels, int size)
{
    std::vector<float> data(channels * size);
    for (auto& v : data)
    {
        v = (float(rand()) / RAND_MAX - 0.5f) * 10.f;
    }
    return data;
}

std::vector<float> ReLUReference(const std::vector<float>& input, float slope)
{
    std::vector<float> output = input;
    for (auto& v : output)
    {
        if (v < 0)
            v *= slope;
    }
    return output;
}

void TestReLU(int channels, int size, float slope)
{
    auto data = RandomData(channels, size);
    std::vector<float> expected = ReLUReference(data, slope);

    relu(data.data(), channels, size, slope);

    for (size_t i = 0; i < data.size(); ++i)
    {
        EXPECT_FLOAT_EQ(data[i], expected[i]);
    }
}

TEST(ReLUTest, Channels1D)
{
    TestReLU(1, 128, 0.f);
    TestReLU(1, 128, 0.1f);
    TestReLU(1, 124, 0.f);
    TestReLU(1, 124, 0.1f);
    TestReLU(1, 127, 0.f);
    TestReLU(1, 127, 0.1f);
}

TEST(ReLUTest, Channels2D)
{
    TestReLU(15, 24, 0.f);
    TestReLU(15, 24, 0.1f);
    TestReLU(17, 12, 0.f);
    TestReLU(17, 12, 0.1f);
    TestReLU(19, 15, 0.f);
    TestReLU(19, 15, 0.1f);
}

TEST(ReLUTest, Channels3D)
{
    TestReLU(5, 7*24, 0.f);
    TestReLU(5, 7*24, 0.1f);
    TestReLU(7, 9*12, 0.f);
    TestReLU(7, 9*12, 0.1f);
    TestReLU(3, 5*13, 0.f);
    TestReLU(3, 5*13, 0.1f);
}

TEST(ReLUTest, Channels4D)
{
    TestReLU(24, 5*6*7, 0.f);
    TestReLU(24, 5*6*7, 0.1f);
    TestReLU(12, 7*8*9, 0.f);
    TestReLU(12, 7*8*9, 0.1f);
    TestReLU(13, 3*4*5, 0.f);
    TestReLU(13, 3*4*5, 0.1f);
}

