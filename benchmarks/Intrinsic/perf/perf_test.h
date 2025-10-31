#ifndef PERF_TEST_H
#define PERF_TEST_H

#include <benchmark/benchmark.h>
#include <stddef.h>

#include <vector>

template <typename T>
static std::vector<T> sequence_vector(size_t size) {
  std::vector<T> v;
  v.reserve(size);
  for (size_t i = 0; i < size; ++i) {
    v.push_back(static_cast<T>(1) + static_cast<T>(i));
  }
  return v;
}

#endif