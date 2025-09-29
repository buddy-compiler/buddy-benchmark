#include <gtest/gtest.h>
#include <cmath>
#include <vector>

extern void magnitude32f(const float* x, const float* y, float* mag, int len);

TEST(Magnitude32fTest, HandlesEmptyArray) {
    float x[1] = {1.0f};
    float y[1] = {2.0f};
    float mag[1] = {0.0f};

    magnitude32f(x, y, mag, 0);
    EXPECT_FLOAT_EQ(mag[0], 0.0f);
}

TEST(Magnitude32fTest, HandlesSingleElement) {
    float x[1] = {3.0f};
    float y[1] = {4.0f};
    float mag[1];

    magnitude32f(x, y, mag, 1);
    EXPECT_FLOAT_EQ(mag[0], 5.0f);
}

TEST(Magnitude32fTest, HandlesZeroVectors) {
    float x[1] = {0.0f};
    float y[1] = {0.0f};
    float mag[1];

    magnitude32f(x, y, mag, 1);
    EXPECT_FLOAT_EQ(mag[0], 0.0f);
}

TEST(Magnitude32fTest, HandlesUnitVectors) {
    const int n = 4;
    float x[n] = {1.0f, 0.0f, -1.0f, 0.0f};
    float y[n] = {0.0f, 1.0f, 0.0f, -1.0f};
    float mag[n];

    magnitude32f(x, y, mag, n);

    for (int i = 0; i < n; ++i) {
        EXPECT_FLOAT_EQ(mag[i], 1.0f);
    }
}

TEST(Magnitude32fTest, HandlesMultipleElements) {
    const int n = 5;
    float x[n] = {3.0f, 5.0f, 0.0f, -3.0f, 1.0f};
    float y[n] = {4.0f, 12.0f, 7.0f, -4.0f, 1.0f};
    float expected[n];
    float mag[n];

    for (int i = 0; i < n; ++i) {
        expected[i] = std::sqrt(x[i]*x[i] + y[i]*y[i]);
    }

    magnitude32f(x, y, mag, n);

    for (int i = 0; i < n; ++i) {
        EXPECT_FLOAT_EQ(mag[i], expected[i]);
    }
}

TEST(Magnitude32fTest, HandlesNegativeValues) {
    const int n = 4;
    float x[n] = {-3.0f, 3.0f, -3.0f, 3.0f};
    float y[n] = {4.0f, 4.0f, -4.0f, -4.0f};
    float mag[n];

    magnitude32f(x, y, mag, n);

    for (int i = 0; i < n; ++i) {
        EXPECT_FLOAT_EQ(mag[i], 5.0f);
    }
}

TEST(Magnitude32fTest, HandlesPythagoreanTriples) {
    const int n = 3;

    float x[n] = {3.0f, 5.0f, 8.0f};
    float y[n] = {4.0f, 12.0f, 15.0f};
    float expected[n] = {5.0f, 13.0f, 17.0f};
    float mag[n];

    magnitude32f(x, y, mag, n);

    for (int i = 0; i < n; ++i) {
        EXPECT_FLOAT_EQ(mag[i], expected[i]);
    }
}

TEST(Magnitude32fTest, HandlesSmallValues) {
    const int n = 3;
    float x[n] = {0.3f, 0.0001f, 0.6f};
    float y[n] = {0.4f, 0.0001f, 0.8f};
    float expected[n];
    float mag[n];

    for (int i = 0; i < n; ++i) {
        expected[i] = std::sqrt(x[i]*x[i] + y[i]*y[i]);
    }

    magnitude32f(x, y, mag, n);

    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(mag[i], expected[i], 1e-6f);
    }
}

TEST(Magnitude32fTest, HandlesLargeArray) {
    const int n = 10000;
    float* x = new float[n];
    float* y = new float[n];
    float* mag = new float[n];
    float* expected = new float[n];

    for (int i = 0; i < n; ++i) {
        x[i] = static_cast<float>(i % 100) / 10.0f;
        y[i] = static_cast<float>((i + 1) % 100) / 10.0f;
        expected[i] = std::sqrt(x[i]*x[i] + y[i]*y[i]);
    }

    magnitude32f(x, y, mag, n);

    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(mag[i], expected[i], 1e-5f);
    }

    delete[] x;
    delete[] y;
    delete[] mag;
    delete[] expected;
}

TEST(Magnitude32fTest, HandlesIdenticalInputs) {
    const int n = 3;
    float x[n] = {2.0f, 5.0f, 7.0f};
    float y[n] = {2.0f, 5.0f, 7.0f};
    float expected[n];
    float mag[n];

    for (int i = 0; i < n; ++i) {
        expected[i] = x[i] * std::sqrt(2.0f);
    }

    magnitude32f(x, y, mag, n);

    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(mag[i], expected[i], 1e-6f);
    }
}

TEST(Magnitude32fTest, HandlesSpecialCases) {
    const int n = 5;
    float x[n] = {1.0f, -1.0f, 0.0f, 100.0f, 0.001f};
    float y[n] = {0.0f, 0.0f, 1.0f, 100.0f, 0.001f};
    float expected[n] = {1.0f, 1.0f, 1.0f, std::sqrt(20000.0f), std::sqrt(0.000002f)};
    float mag[n];

    magnitude32f(x, y, mag, n);

    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(mag[i], expected[i], 1e-5f);
    }
}
