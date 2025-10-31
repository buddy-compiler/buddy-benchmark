#include <gtest/gtest.h>
#include <cmath>
#include <vector>

extern void fastAtan32f(const float* y, const float* x, float* dst, size_t n, bool angle_in_deg);

static const float DEG_EPSILON = 0.01f;
static const float RAD_EPSILON = DEG_EPSILON * static_cast<float>(M_PI / 180.0);

static float referenceAtan2Deg(float y, float x) {
  float angle = std::atan2(y, x) * 180.0f / static_cast<float>(M_PI);
  if (angle < 0) angle += 360;
  return angle;
}

TEST(FastAtan32fTest, HandlesEmptyArray) {
    float Y[1] = {1.0f};
    float X[1] = {1.0f};
    float angle[1] = {0.0f};

    fastAtan32f(Y, X, angle, 0, true);
    EXPECT_FLOAT_EQ(angle[0], 0.0f);
}

struct TestCase {
  float y;
  float x;
  float expected;
};

TEST(FastAtan32fTest, HandlesSingleElementDeg) {

  TestCase testCases[] = {

      {0.0f, 1.0f, 0.0f},
      {1.0f, 0.0f, 90.0f},
      {0.0f, -1.0f, 180.0f},
      {-1.0f, 0.0f, 270.0f},
      {1.0f, 1.0f, 45.0f},
      {1.0f, -1.0f, 135.0f},
      {-1.0f, -1.0f, 225.0f},
      {-1.0f, 1.0f, 315.0f},
      {2.0f, 1.0f, 0.0f}
  };

  testCases[8].expected = referenceAtan2Deg(2.0f, 1.0f);

  for (const auto& testCase : testCases) {
    float angle;
    fastAtan32f(&testCase.y, &testCase.x, &angle, 1, true);
    EXPECT_NEAR(angle, testCase.expected, DEG_EPSILON)
        << "atan2(" << testCase.y << ", " << testCase.x << ")";
  }
}

TEST(FastAtan32fTest, HandlesSingleElementRad) {

  const float rad45 = static_cast<float>(M_PI / 4);
  const float rad90 = static_cast<float>(M_PI / 2);
  const float rad135 = static_cast<float>(3 * M_PI / 4);
  const float rad180 = static_cast<float>(M_PI);
  const float rad225 = static_cast<float>(5 * M_PI / 4);
  const float rad315 = static_cast<float>(7 * M_PI / 4);

  const TestCase testCases[] = {
      {1.0f, 1.0f, rad45},
      {1.0f, -1.0f, rad135},
      {-1.0f, -1.0f, rad225},
      {-1.0f, 1.0f, rad315}
  };

  for (const auto& testCase : testCases) {
    float angle;
    fastAtan32f(&testCase.y, &testCase.x, &angle, 1, false);
    EXPECT_NEAR(angle, testCase.expected, RAD_EPSILON)
        << "atan2(" << testCase.y << ", " << testCase.x << ")";
  }
}

TEST(FastAtan32fTest, HandlesMultipleElements) {
  const size_t n = 5;

  float Y[n] = {0.0f, 1.0f, 0.0f, -1.0f, 0.0f};
  float X[n] = {1.0f, 0.0f, -1.0f, 0.0f, 0.0f};
  float expectedDeg[n] = {0.0f, 90.0f, 180.0f, 270.0f, 0.0f};
  float actual[n];

  fastAtan32f(Y, X, actual, n, true);

  for (size_t i = 0; i < n; ++i) {
    EXPECT_NEAR(actual[i], expectedDeg[i], DEG_EPSILON)
        << "Index " << i << ": atan2(" << Y[i] << ", " << X[i] << ")";
  }
}

TEST(FastAtan32fTest, HandlesLargeArrayDeg) {
  const size_t n = 10000;
  std::vector<float> Y(n);
  std::vector<float> X(n);
  std::vector<float> actual(n);
  std::vector<float> expected(n);

  const float degToRad = static_cast<float>(M_PI / 180.0);

  for (size_t i = 0; i < n; ++i) {
    float angle_deg = static_cast<float>(i % 360);
    float rad = angle_deg * degToRad;
    X[i] = std::cos(rad);
    Y[i] = std::sin(rad);
    expected[i] = angle_deg;
  }

    fastAtan32f(Y.data(), X.data(), actual.data(), n, true);

    float maxError = 0.0f;
    size_t maxErrorIndex = 0;
    for (size_t i = 0; i < n; ++i) {
      float error = std::fabs(actual[i] - expected[i]);
      if (error > maxError) {
        maxError = error;
        maxErrorIndex = i;
      }
    }

    const float tolerance = 0.1f;
    for (size_t i = 0; i < n; ++i) {
      EXPECT_NEAR(actual[i], expected[i], tolerance)
          << "Index " << i << ": atan2(" << Y[i] << ", " << X[i] << ")";
    }
}

TEST(FastAtan32fTest, HandlesMixedRadDeg) {
  const size_t n = 4;

  float Y[n] = {1.0f, 1.0f, 1.0f, -1.0f};
  float X[n] = {1.0f, 0.0f, -1.0f, 1.0f};
  float degResults[n];
  float radResults[n];

  fastAtan32f(Y, X, degResults, n, true);
  const float deg45 = 45.0f;
  const float deg90 = 90.0f;
  const float deg135 = 135.0f;
  const float deg315 = 315.0f;

  EXPECT_NEAR(degResults[0], deg45, DEG_EPSILON);
  EXPECT_NEAR(degResults[1], deg90, DEG_EPSILON);
  EXPECT_NEAR(degResults[2], deg135, DEG_EPSILON);
  EXPECT_NEAR(degResults[3], deg315, DEG_EPSILON);

  fastAtan32f(Y, X, radResults, n, false);
  const float rad45 = static_cast<float>(M_PI / 4);
  const float rad90 = static_cast<float>(M_PI / 2);
  const float rad135 = static_cast<float>(3 * M_PI / 4);
  const float rad315 = static_cast<float>(7 * M_PI / 4);

  EXPECT_NEAR(radResults[0], rad45, RAD_EPSILON);
  EXPECT_NEAR(radResults[1], rad90, RAD_EPSILON);
  EXPECT_NEAR(radResults[2], rad135, RAD_EPSILON);
  EXPECT_NEAR(radResults[3], rad315, RAD_EPSILON);
}