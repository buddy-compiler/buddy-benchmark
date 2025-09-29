#include <gtest/gtest.h>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <cmath>

extern void hardsigmoid(float* data, int channels, int size, float alpha = 0.2f, float beta = 0.5f);

inline std::vector<float> RandomData(int channels, int size)
{
    std::vector<float> data(channels * size);
    for (auto& v : data)
    {
        v = (float(rand()) / RAND_MAX - 0.5f) * 10.f;
    }
    return data;
}

inline std::vector<float> HardSigmoidReference(const std::vector<float>& input, float alpha, float beta)
{
    std::vector<float> output = input;
    for (auto& v : output)
    {
        float y = alpha * v + beta;
        if (y < 0.0f) y = 0.0f;
        if (y > 1.0f) y = 1.0f;
        v = y;
    }
    return output;
}

void TestHardSigmoid(int channels, int size, float alpha = 0.2f, float beta = 0.5f)
{
    auto data = RandomData(channels, size);
    std::vector<float> expected = HardSigmoidReference(data, alpha, beta);

    hardsigmoid(data.data(), channels, size, alpha, beta);

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

TEST(HardSigmoidTest, Channels1D)
{
    TestHardSigmoid(128, 1);
    TestHardSigmoid(124, 1);
    TestHardSigmoid(127, 1);
}

TEST(HardSigmoidTest, Channels2D)
{
    TestHardSigmoid(15, 24);
    TestHardSigmoid(17, 12);
    TestHardSigmoid(19, 15);
}

TEST(HardSigmoidTest, Channels3D)
{
    TestHardSigmoid(5, 7*24);
    TestHardSigmoid(7, 9*12);
    TestHardSigmoid(3, 5*13);
}

TEST(HardSigmoidTest, Channels4D)
{
    TestHardSigmoid(5, 7*32);
    TestHardSigmoid(7, 8*9*12);
    TestHardSigmoid(3, 4*5*13);
}
