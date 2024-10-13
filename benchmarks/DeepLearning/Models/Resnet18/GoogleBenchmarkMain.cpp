#include <benchmark/benchmark.h>
#include <buddy/Core/Container.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>

// Helper functions and variables.
namespace {
constexpr size_t ParamsSize = 11699112;
constexpr size_t OutputSize = 1000;
constexpr size_t InputChannels = 3;
constexpr size_t InputHeight = 224;
constexpr size_t InputWidth = 224;
constexpr size_t ExtraInputSize = 20;  
const std::string PASS = "\033[32mPASS\033[0m";
const std::string FAIL = "\033[31mFAIL\033[0m";

bool areArraysEqual(float array1[], float array2[], int size,
                    float epsilon = 0.0001) {
  for (int i = 0; i < size; ++i) {
    if (fabs(array1[i] - array2[i]) > epsilon) {
      return false;
    }
  }
  return true;
}
} // namespace

namespace {

// Declare the ResNet18 C interface.
extern "C" {
void _mlir_ciface_forward_auto_vectorization(
                                             MemRef<float, 2> *output,
                                             MemRef<float, 1> *params,
                                             MemRef<long long, 1> *extra_input,
                                             MemRef<float, 4> *input);

void _mlir_ciface_forward_buddy_vectorization(
                                              MemRef<float, 2> *output,
                                              MemRef<float, 1> *params,
                                              MemRef<long long, 1> *extra_input,
                                              MemRef<float, 4> *input);
}

template <typename Func>
void DL_MODEL_Resnet18(benchmark::State &state, Func func) {
  // Create containers for input, output, parameters, and extra_input.
  MemRef<float, 1> paramsContainerf32({ParamsSize}, 5);
  MemRef<float, 4> inputContainer({1, InputChannels, InputHeight, InputWidth}, 6);
  MemRef<float, 2> outputContainer({1, OutputSize}, 7);
  MemRef<long long, 1> extraInputContainer({ExtraInputSize}, 8);  

  for (auto _ : state) {
    func(&outputContainer, &paramsContainerf32, &extraInputContainer, &inputContainer);  
  }
}

} // namespace

// Register benchmarking function with different arguments.
BENCHMARK_CAPTURE(DL_MODEL_Resnet18, Auto_Vectorization,
                  _mlir_ciface_forward_auto_vectorization)
    ->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(DL_MODEL_Resnet18, Buddy_Vectorization,
                  _mlir_ciface_forward_buddy_vectorization)
    ->Unit(benchmark::kMillisecond);

/// Correctness Verification
/// The verification does not affect the performance.
/// - Set the scalar case as the criteria.
/// - Input elements are random numbers.
/// - Output elements are initialized to zero.
/// - Compare the output of various optimizations with the scalar version to
///   verify correctness.
void verification() {
  // Set the random number generator.
  std::random_device rd;
  std::mt19937 generator(rd());
  std::uniform_real_distribution<float> distribution(0.0, 1.0);

  // Create containers for input, output, parameters, and extra input.
  MemRef<float, 4> inputContainer({1, InputChannels, InputHeight, InputWidth}, 6);
  MemRef<float, 2> resultAutoVectorizationContainer({1, OutputSize}, 7);
  MemRef<float, 2> resultBuddyVectorizationContainer({1, OutputSize}, 7);
  MemRef<float, 1> paramsContainerf32({ParamsSize}, 5);
  MemRef<long long, 1> extraInputContainer({ExtraInputSize}, 8);  // 使用超参数控制维度

  // Populate input with random data
  for (int i = 0; i < InputChannels * InputHeight * InputWidth; ++i) {
    inputContainer.getData()[i] = distribution(generator);
  }

  // Populate the extra input with some data
  for (int i = 0; i < ExtraInputSize; ++i) {
    extraInputContainer.getData()[i] = rd();
  }

  // Call the forward function of the model.
  _mlir_ciface_forward_auto_vectorization(&resultAutoVectorizationContainer,
                                          &paramsContainerf32, &extraInputContainer, &inputContainer);
  _mlir_ciface_forward_buddy_vectorization(&resultBuddyVectorizationContainer,
                                           &paramsContainerf32, &extraInputContainer, &inputContainer);

  // Compare results.
  auto resultAutoVectorization = resultAutoVectorizationContainer.getData();
  auto resultBuddyVectorization = resultBuddyVectorizationContainer.getData();
  size_t resultSize = resultAutoVectorizationContainer.getSize();

  // Print the verification result.
  std::cout << "-----------------------------------------------------------"
            << std::endl;
  std::cout << "Correctness Verification: "
            << (areArraysEqual(resultAutoVectorization,
                               resultBuddyVectorization, resultSize)
                    ? PASS
                    : FAIL)
            << std::endl;
  std::cout << "-----------------------------------------------------------"
            << std::endl;
}

int main(int argc, char **argv) {
  // Run benchmark.
  ::benchmark::Initialize(&argc, argv);
  ::benchmark::RunSpecifiedBenchmarks();
  // Run correctness verification.
  verification();
  return 0;
}
