#include <benchmark/benchmark.h>
#include <buddy/Core/Container.h>
#include <iostream>
#include <random>

// Define target layout.
#define BATCH_SIZE 3
#define M (128 * 4)
#define N (128 * 4)
#define K (128 * 4)

namespace {
const std::string PASS = "\033[32mPASS\033[0m";
const std::string FAIL = "\033[31mFAIL\033[0m";

bool areArraysEqual(int array1[], int array2[], int size) {
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
void _mlir_ciface_batch_matmul_scalar(MemRef<int, 3> *input1,
                                      MemRef<int, 3> *input2,
                                      MemRef<int, 3> *output);
void _mlir_ciface_batch_matmul_auto_vectorization(MemRef<int, 3> *input1,
                                                  MemRef<int, 3> *input2,
                                                  MemRef<int, 3> *output);
void _mlir_ciface_batch_matmul_rvv_vectorization(MemRef<int, 3> *input1,
                                                 MemRef<int, 3> *input2,
                                                 MemRef<int, 3> *output);
void _mlir_ciface_batch_matmul_vectorization(MemRef<int, 3> *input1,
                                             MemRef<int, 3> *input2,
                                             MemRef<int, 3> *output);
}

#define DEFINE_BATCH_MATMUL_BENCHMARK(name, func)                              \
  void BM_BATCH_MATMUL_##name(benchmark::State &state) {                       \
    intptr_t sizesInput1[3] = {BATCH_SIZE, M, K};                              \
    intptr_t sizesInput2[3] = {BATCH_SIZE, K, N};                              \
    intptr_t sizesOutput[3] = {BATCH_SIZE, M, N};                              \
                                                                               \
    MemRef<int, 3> input1(sizesInput1, 1.0);                                   \
    MemRef<int, 3> input2(sizesInput2, 1.0);                                   \
    MemRef<int, 3> output(sizesOutput, 0.0);                                   \
                                                                               \
    for (auto _ : state) {                                                     \
      func(&input1, &input2, &output);                                         \
    }                                                                          \
  }

DEFINE_BATCH_MATMUL_BENCHMARK(SCALAR, _mlir_ciface_batch_matmul_scalar)
DEFINE_BATCH_MATMUL_BENCHMARK(AutoVectorization,
                              _mlir_ciface_batch_matmul_auto_vectorization)
DEFINE_BATCH_MATMUL_BENCHMARK(Vectorization,
                              _mlir_ciface_batch_matmul_vectorization)
DEFINE_BATCH_MATMUL_BENCHMARK(RvvVectorization,
                              _mlir_ciface_batch_matmul_rvv_vectorization)
} // namespace

// Register benchmark cases.
BENCHMARK(BM_BATCH_MATMUL_SCALAR)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_BATCH_MATMUL_AutoVectorization)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_BATCH_MATMUL_Vectorization)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_BATCH_MATMUL_RvvVectorization)->Unit(benchmark::kMillisecond);

/// Correctness Verification
void verification() {
  unsigned int seed = time(NULL);

  intptr_t sizesInput1[3] = {BATCH_SIZE, M, K};
  intptr_t sizesInput2[3] = {BATCH_SIZE, K, N};
  intptr_t sizesOutput[3] = {BATCH_SIZE, M, N};

  const int input1Size = BATCH_SIZE * M * K;
  int input1Rand[input1Size];
  for (int i = 0; i < input1Size; ++i) {
    input1Rand[i] = rand_r(&seed) / 1000 - 500;
  }
  MemRef<int, 3> input1MemRef(input1Rand, sizesInput1);

  const int input2Size = BATCH_SIZE * K * N;
  int input2Rand[input2Size];
  for (int i = 0; i < input2Size; ++i) {
    input2Rand[i] = rand_r(&seed) / 1000 - 500;
  }
  MemRef<int, 3> input2MemRef(input2Rand, sizesInput2);

  const int outputSize = BATCH_SIZE * M * N;
  MemRef<int, 3> outputScalar(sizesOutput, 0.0);
  MemRef<int, 3> outputAutoVectorization(sizesOutput, 0.0);
  MemRef<int, 3> outputRVV(sizesOutput, 0.0);
  MemRef<int, 3> outputManual(sizesOutput, 0.0);

  _mlir_ciface_batch_matmul_scalar(&input1MemRef, &input2MemRef, &outputScalar);
  _mlir_ciface_batch_matmul_auto_vectorization(&input1MemRef, &input2MemRef,
                                               &outputAutoVectorization);
  _mlir_ciface_batch_matmul_vectorization(&input1MemRef, &input2MemRef,
                                          &outputManual);
  _mlir_ciface_batch_matmul_rvv_vectorization(&input1MemRef, &input2MemRef,
                                              &outputRVV);

  auto resultScalar = outputScalar.getData();
  auto resultAutoVectorization = outputAutoVectorization.getData();
  auto resultRVVVectorization = outputRVV.getData();

  std::cout << "-----------------------------------------------------------"
            << std::endl;
  std::cout << "Correctness Verification:" << std::endl;
  std::cout << "RVV case: "
            << (areArraysEqual(resultScalar, resultRVVVectorization, outputSize)
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
