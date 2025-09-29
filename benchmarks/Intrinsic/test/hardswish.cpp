#include <gtest/gtest.h>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <cmath>

extern void hardswish(float* data, int channels, int size, float alpha = 0.2f, float beta = 0.5f);

inline std::vector<float> RandomData(int channels, int size)
{
    std::vector<float> data(channels * size);
    for (auto& v : data)
    {
        v = (float(rand()) / RAND_MAX - 0.5f) * 10.f;
    }
    return data;
}

inline std::vector<float> HardSwishReference(const std::vector<float>& input, float alpha, float beta)
{
    std::vector<float> output = input;
    for (auto& v : output)
    {
        float y = alpha * v + beta;
        if (y < 0.0f) y = 0.0f;
        if (y > 1.0f) y = 1.0f;
        v = v * y;
    }
    return output;
}

void TestHardSwish(int channels, int size, float alpha = 0.2f, float beta = 0.5f)
{
    auto data = RandomData(channels, size);
    std::vector<float> expected = HardSwishReference(data, alpha, beta);

    hardswish(data.data(), channels, size, alpha, beta);

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

TEST(HardSwishTest, Channels1D)
{
    TestHardSwish(128, 1);
    TestHardSwish(124, 1);
    TestHardSwish(127, 1);
}

TEST(HardSwishTest, Channels2D)
{
    TestHardSwish(15, 24);
    TestHardSwish(17, 12);
    TestHardSwish(19, 15);
}

TEST(HardSwishTest, Channels3D)
{
    TestHardSwish(5, 7*24);
    TestHardSwish(7, 9*12);
    TestHardSwish(3, 5*13);
}

TEST(HardSwishTest, Channels4D)
{
    TestHardSwish(5, 7*32);
    TestHardSwish(7, 8*9*12);
    TestHardSwish(3, 4*5*13);
}
