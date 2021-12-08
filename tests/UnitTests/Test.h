#include <gtest/gtest.h>

template <typename T> void ASSERT_ARRAY_EQ(const T *x, const T *y) {
  size_t size = sizeof(x) / sizeof(T);
  size_t size_y = sizeof(y) / sizeof(T);
  ASSERT_EQ(size, size_y);

  if (std::is_integral<T>::value) {
    for (size_t i = 0; i < size; i++) {
      ASSERT_EQ(x[i], y[i]);
    }
  } else if (std::is_same<T, float>::value) {
    for (size_t i = 0; i < size; i++) {
      ASSERT_FLOAT_EQ(x[i], y[i]);
    }
  } else if (std::is_same<T, double>::value) {
    for (size_t i = 0; i < size; i++) {
      ASSERT_DOUBLE_EQ(x[i], y[i]);
    }
  }
}
