#include <benchmark/benchmark.h>
#include <buddy/Core/Container.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>

// Helper functions and variables.
namespace {
constexpr size_t ParamsSize = 99148800;  // Whisper params size from MLIR
constexpr size_t InputChannels = 80;     // Input feature channels
constexpr size_t InputTimesteps = 3000;  // Input feature timesteps
constexpr size_t DecoderInputSize = 448; // Decoder input size
constexpr size_t Output1Timestep = 1500; // Output1 timestep
constexpr size_t Output1Features = 512;  // Output1 feature dimension
constexpr size_t Output2DecoderSteps = 448; // Output2 decoder steps
constexpr size_t Output2VocabSize = 51865;  // Output2 vocabulary size
const std::string PASS = "\033[32mPASS\033[0m";
const std::string FAIL = "\033[31mFAIL\033[0m";

bool areArraysEqual(float array1[], float array2[], int size, float epsilon = 0.0001) {
  for (int i = 0; i < size; ++i) {
    if (fabs(array1[i] - array2[i]) > epsilon) {
      return false;
    }
  }
  return true;
}
} // namespace

namespace {

// Declare the Whisper C interface.
extern "C" {
void _mlir_ciface_forward_auto_vectorization(
                                             MemRef<float, 3> resultContainer[2],
                                             MemRef<float, 1> *params,
                                             MemRef<float, 3> *input,
                                             MemRef<size_t, 2> *decoder_input
                                             );

void _mlir_ciface_forward_buddy_vectorization(
                                              MemRef<float, 3> resultContainer[2],
                                              MemRef<float, 1> *params,
                                              MemRef<float, 3> *input,
                                              MemRef<size_t, 2> *decoder_input
                                              );
}

template <typename Func>
void DL_MODEL_Whisper(benchmark::State &state, Func func) {
  // Create containers for input, output, parameters, and decoder input.
  MemRef<float, 1> paramsContainerf32({ParamsSize}, 5);
  MemRef<float, 3> inputContainer({1, InputChannels, InputTimesteps}, 6);
  MemRef<size_t, 2> decoderInputContainer({1, DecoderInputSize}, 8);  

  // Create result container (merged into one array)
  MemRef<float, 3> resultContainer[2] = {
      MemRef<float, 3>({1, Output1Timestep, Output1Features}, false, 7),
      MemRef<float, 3>({1, Output2DecoderSteps, Output2VocabSize}, false, 9)
  };

  for (auto _ : state) {
    func(resultContainer, &paramsContainerf32, &inputContainer, &decoderInputContainer); 
  }
}

} // namespace

// Register benchmarking function with different arguments.
BENCHMARK_CAPTURE(DL_MODEL_Whisper, Auto_Vectorization,
                  _mlir_ciface_forward_auto_vectorization)
    ->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(DL_MODEL_Whisper, Buddy_Vectorization,
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

  // Create containers for input, output, parameters, and decoder input.
  MemRef<float, 3> inputContainer({1, InputChannels, InputTimesteps}, 6);
  MemRef<size_t, 2> decoderInputContainer({1, DecoderInputSize}, 8);
  MemRef<float, 1> paramsContainerf32({ParamsSize}, 5);

  // Create result containers for auto and buddy vectorization (using array form)
  MemRef<float, 3> resultAutoVectorization[2] = {
      MemRef<float, 3>({1, Output1Timestep, Output1Features}, 7),
      MemRef<float, 3>({1, Output2DecoderSteps, Output2VocabSize}, 9)
  };

  MemRef<float, 3> resultBuddyVectorization[2] = {
      MemRef<float, 3>({1, Output1Timestep, Output1Features}, 7),
      MemRef<float, 3>({1, Output2DecoderSteps, Output2VocabSize}, 9)
  };


  // Call the forward function of the model.
  _mlir_ciface_forward_auto_vectorization(resultAutoVectorization,
                                          &paramsContainerf32, &inputContainer, &decoderInputContainer);
  _mlir_ciface_forward_buddy_vectorization(resultBuddyVectorization,
                                           &paramsContainerf32, &inputContainer, &decoderInputContainer);

  // Compare results.
  auto resultAutoVectorization1 = resultAutoVectorization[0].getData();
  auto resultAutoVectorization2 = resultAutoVectorization[1].getData();
  auto resultBuddyVectorization1 = resultBuddyVectorization[0].getData();
  auto resultBuddyVectorization2 = resultBuddyVectorization[1].getData();

  size_t resultSize1 = resultAutoVectorization[0].getSize();
  size_t resultSize2 = resultAutoVectorization[1].getSize();

  // Print the verification result.
  std::cout << "-----------------------------------------------------------"
            << std::endl;
  std::cout << "Correctness Verification for Output1: "
            << (areArraysEqual(resultAutoVectorization1, resultBuddyVectorization1, resultSize1)
                    ? PASS
                    : FAIL)
            << std::endl;
  std::cout << "Correctness Verification for Output2: "
            << (areArraysEqual(resultAutoVectorization2, resultBuddyVectorization2, resultSize2)
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
