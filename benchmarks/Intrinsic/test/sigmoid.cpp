#include <gtest/gtest.h>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <cmath>

extern void sigmoid(float* data, int channels, int size);

inline std::vector<float> RandomData(int channels, int size)
{
    std::vector<float> data(channels * size);
    for (auto& v : data)
    {
        v = (float(rand()) / RAND_MAX - 0.5f) * 10.f;
    }
    return data;
}

std::vector<float> SigmoidReference(const std::vector<float>& input)
{
    std::vector<float> output = input;
    for (auto& v : output)
    {
        v = 1.f / (1.f + std::exp(-v));
    }
    return output;
}

void TestSigmoid(int channels, int size)
{
    auto data = RandomData(channels, size);
    std::vector<float> expected = SigmoidReference(data);

    sigmoid(data.data(), channels, size);
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

TEST(SigmoidTest, Channels1D)
{
    TestSigmoid(1, 128);
    TestSigmoid(1, 124);
    TestSigmoid(1, 127);
}

TEST(SigmoidTest, Channels2D)
{
    TestSigmoid(15, 24);
    TestSigmoid(17, 12);
    TestSigmoid(19, 15);
}

TEST(SigmoidTest, Channels3D)
{
    TestSigmoid(5, 7*24);
    TestSigmoid(7, 9*12);
    TestSigmoid(3, 5*13);
}

TEST(SigmoidTest, Channels4D)
{
    TestSigmoid(24, 5*6*7);
    TestSigmoid(12, 7*8*9);
    TestSigmoid(13, 3*4*5);
}