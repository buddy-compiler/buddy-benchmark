#include <benchmark/benchmark.h>
#include <buddy/Core/Container.h>
#include <iostream>
#include <random>

// Define target layout.
#define BATCH_SIZE 3
#define M 128
#define N 128
#define K 128

namespace {
const std::string PASS = "\033[32mPASS\033[0m";
const std::string FAIL = "\033[31mFAIL\033[0m";

bool areArraysEqual(float array1[], float array2[], int size) {
  for (int i = 0; i < size; ++i) {
    if (array1[i] != array2[i]) {
      return false;
    }
  }
  return true;
}
} // namespace

namespace {
// Declare the batch matmul C interface.
extern "C" {
void _mlir_ciface_batch_matmul_scalar(MemRef<float, 3> *input1,
                                      MemRef<float, 3> *input2,
                                      MemRef<float, 3> *output);
void _mlir_ciface_batch_matmul_auto_vectorization(MemRef<float, 3> *input1,
                                                  MemRef<float, 3> *input2,
                                                  MemRef<float, 3> *output);
}

#define DEFINE_BATCH_MATMUL_BENCHMARK(name, func)                             \
  void BM_BATCH_MATMUL_##name(benchmark::State &state) {                      \
    intptr_t sizesInput1[3] = {BATCH_SIZE, M, K};                             \
    intptr_t sizesInput2[3] = {BATCH_SIZE, K, N};                             \
    intptr_t sizesOutput[3] = {BATCH_SIZE, M, N};                             \
                                                                              \
    MemRef<float, 3> input1(sizesInput1, 1.0);                                \
    MemRef<float, 3> input2(sizesInput2, 1.0);                                \
    MemRef<float, 3> output(sizesOutput, 0.0);                                \
                                                                              \
    for (auto _ : state) {                                                    \
      func(&input1, &input2, &output);                                        \
    }                                                                         \
  }

DEFINE_BATCH_MATMUL_BENCHMARK(SCALAR, _mlir_ciface_batch_matmul_scalar)
DEFINE_BATCH_MATMUL_BENCHMARK(AutoVectorization, _mlir_ciface_batch_matmul_auto_vectorization)
} // namespace

// Register benchmark cases.
BENCHMARK(BM_BATCH_MATMUL_SCALAR)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_BATCH_MATMUL_AutoVectorization)->Unit(benchmark::kMillisecond);

/// Correctness Verification
void verification() {
  std::random_device rd;
  std::mt19937 generator(rd());
  std::uniform_real_distribution<float> distribution(0.0, 1.0);

  intptr_t sizesInput1[3] = {BATCH_SIZE, M, K};
  intptr_t sizesInput2[3] = {BATCH_SIZE, K, N};
  intptr_t sizesOutput[3] = {BATCH_SIZE, M, N};

  const int input1Size = BATCH_SIZE * M * K;
  float input1Rand[input1Size];
  for (int i = 0; i < input1Size; ++i) {
    input1Rand[i] = distribution(generator);
  }
  MemRef<float, 3> input1MemRef(input1Rand, sizesInput1);

  const int input2Size = BATCH_SIZE * K * N;
  float input2Rand[input2Size];
  for (int i = 0; i < input2Size; ++i) {
    input2Rand[i] = distribution(generator);
  }
  MemRef<float, 3> input2MemRef(input2Rand, sizesInput2);

  const int outputSize = BATCH_SIZE * M * N;
  MemRef<float, 3> outputScalar(sizesOutput, 0.0);
  MemRef<float, 3> outputAutoVectorization(sizesOutput, 0.0);

  _mlir_ciface_batch_matmul_scalar(&input1MemRef, &input2MemRef, &outputScalar);
  _mlir_ciface_batch_matmul_auto_vectorization(&input1MemRef, &input2MemRef,
                                               &outputAutoVectorization);

  auto resultScalar = outputScalar.getData();
  auto resultAutoVectorization = outputAutoVectorization.getData();

  std::cout << "-----------------------------------------------------------"
            << std::endl;
  std::cout << "Correctness Verification:" << std::endl;
  std::cout << "Transform case: "
            << (areArraysEqual(resultScalar, resultAutoVectorization, outputSize)
                    ? PASS
                    : FAIL)
            << std::endl;
  std::cout << "-----------------------------------------------------------"
            << std::endl;
}

int main(int argc, char **argv) {
  ::benchmark::Initialize(&argc, argv);
  ::benchmark::RunSpecifiedBenchmarks();
  verification();
  return 0;
}
