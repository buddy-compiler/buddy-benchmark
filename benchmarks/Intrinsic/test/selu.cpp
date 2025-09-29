#include <gtest/gtest.h>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <cmath>

extern void selu(float* data, int channels, int size, float alpha = 1.673264f, float lambda = 1.050700f);

inline std::vector<float> RandomData(int channels, int size)
{
    std::vector<float> data(channels * size);
    for (auto& v : data)
    {
        v = (float(rand()) / RAND_MAX - 0.5f) * 10.f;
    }
    return data;
}

inline std::vector<float> SELUReference(const std::vector<float>& input, float alpha, float lambda)
{
    std::vector<float> output = input;
    for (auto& v : output)
    {
        v = (v > 0.0f) ? lambda * v : lambda * alpha * (std::exp(v) - 1.0f);
    }
    return output;
}

void TestSELU(int channels, int size, float alpha = 1.673264f, float lambda = 1.050700f)
{
    auto data = RandomData(channels, size);
    std::vector<float> expected = SELUReference(data, alpha, lambda);

    selu(data.data(), channels, size, alpha, lambda);

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

TEST(SELUTest, Channels1D)
{
    TestSELU(128, 1);
    TestSELU(124, 1);
    TestSELU(127, 1);
    TestSELU(120, 1);
}

TEST(SELUTest, Channels2D)
{
    TestSELU(13, 32);
    TestSELU(15, 24);
    TestSELU(17, 12);
    TestSELU(19, 15);
}

TEST(SELUTest, Channels3D)
{
    TestSELU(4, 7*32);
    TestSELU(5, 7*24);
    TestSELU(7, 9*12);
    TestSELU(3, 5*13);
}

TEST(SELUTest, Channels4D)
{
    TestSELU(7, 6*5*32);
    TestSELU(5, 6*7*24);
    TestSELU(7, 8*9*12);
    TestSELU(3, 4*5*13);
}