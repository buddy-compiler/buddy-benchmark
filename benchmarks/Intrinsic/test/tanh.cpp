#include <gtest/gtest.h>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <cmath>

extern void tanh(float* data, int channels, int size);

inline std::vector<float> RandomData(int channels, int size)
{
    std::vector<float> data(channels * size);
    for (auto& v : data)
    {
        v = (float(rand()) / RAND_MAX - 0.5f) * 10.f;
    }
    return data;
}

std::vector<float> TanhReference(const std::vector<float>& input)
{
    std::vector<float> output = input;
    for (auto& v : output)
    {
        v = std::tanh(v);
    }
    return output;
}

void TestTanh(int channels, int size)
{
    auto data = RandomData(channels, size);
    std::vector<float> expected = TanhReference(data);

    tanh(data.data(), channels, size);

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

TEST(TanhTest, Dim4)
{
    TestTanh(24, 5 * 6 * 7);
    TestTanh(12, 7 * 8 * 9);
    TestTanh(13, 3 * 4 * 5);
}

TEST(TanhTest, Dim3)
{
    TestTanh(24, 5 * 7);
    TestTanh(12, 7 * 9);
    TestTanh(13, 3 * 5);
}

TEST(TanhTest, Dim2)
{
    TestTanh(1, 15 * 24);
    TestTanh(1, 17 * 12);
    TestTanh(1, 19 * 15);
}

TEST(TanhTest, Dim1)
{
    TestTanh(1, 128);
    TestTanh(1, 124);
    TestTanh(1, 127);
}
