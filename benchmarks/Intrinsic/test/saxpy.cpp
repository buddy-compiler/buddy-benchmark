#include <gtest/gtest.h>

extern void SAXPY(const int n, const float *x, float *y, const float a);

TEST(SAXPYTest, HandlesEmptyArray) {
    float x[1] = {1.0f};
    float y[1] = {2.0f};

    SAXPY(0, x, y, 3.0f);
    EXPECT_FLOAT_EQ(y[0], 2.0f);
}

TEST(SAXPYTest, HandlesSingleElement) {
    float x[1] = {2.0f};
    float y[1] = {3.0f};

    SAXPY(1, x, y, 4.0f);
    EXPECT_FLOAT_EQ(y[0], 4.0f * 2.0f + 3.0f);
}

TEST(SAXPYTest, HandlesMultipleElements) {
    const int n = 5;
    float x[n] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float y[n] = {5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
    float expected[n];

    float a = 2.5f;
    for (int i = 0; i < n; ++i) {
        expected[i] = a * x[i] + y[i];
    }

    SAXPY(n, x, y, a);

    for (int i = 0; i < n; ++i) {
        EXPECT_FLOAT_EQ(y[i], expected[i]);
    }
}

TEST(SAXPYTest, HandlesZeroAlpha) {
    const int n = 3;
    float x[n] = {1.0f, 2.0f, 3.0f};
    float y_original[n] = {4.0f, 5.0f, 6.0f};
    float y[n];

    std::copy(y_original, y_original + n, y);

    SAXPY(n, x, y, 0.0f);

    for (int i = 0; i < n; ++i) {
        EXPECT_FLOAT_EQ(y[i], y_original[i]);
    }
}

TEST(SAXPYTest, HandlesNegativeAlpha) {
    const int n = 3;
    float x[n] = {1.0f, -2.0f, 3.0f};
    float y[n] = {4.0f, 5.0f, -6.0f};
    float expected[n];

    float a = -1.5f;
    for (int i = 0; i < n; ++i) {
        expected[i] = a * x[i] + y[i];
    }

    SAXPY(n, x, y, a);

    for (int i = 0; i < n; ++i) {
        EXPECT_FLOAT_EQ(y[i], expected[i]);
    }
}

TEST(SAXPYTest, HandlesLargeArray) {
    const int n = 10000;
    float* x = new float[n];
    float* y = new float[n];
    float* expected = new float[n];

    float a = 0.5f;

    for (int i = 0; i < n; ++i) {
        x[i] = static_cast<float>(i);
        y[i] = static_cast<float>(n - i);
        expected[i] = a * x[i] + y[i];
    }

    SAXPY(n, x, y, a);

    for (int i = 0; i < n; ++i) {
        EXPECT_FLOAT_EQ(y[i], expected[i]);
    }

    delete[] x;
    delete[] y;
    delete[] expected;
}

TEST(SAXPYTest, HandlesInplaceOperation) {
    const int n = 3;
    float x[n] = {1.0f, 2.0f, 3.0f};
    float y[n];

    std::copy(x, x + n, y);

    float a = 2.0f;
    float expected[n];
    for (int i = 0; i < n; ++i) {
      expected[i] = a * y[i] + y[i];
    }

    SAXPY(n, y, y, a);

    for (int i = 0; i < n; ++i) {
        EXPECT_FLOAT_EQ(y[i], expected[i]);
    }
}
