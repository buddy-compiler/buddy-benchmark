//===- Utils.hpp ----------------------------------------------------------===//
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
//
// This file implements helper functions for FIROp.
//
//===----------------------------------------------------------------------===//

#ifndef FIR_UTILS_HPP
#define FIR_UTILS_HPP

#include <benchmark/benchmark.h>
#include <buddy/Core/Container.h>
#include <buddy/DAP/DAP.h>
#include <fstream>
#include <iomanip>
#include <kfr/base.hpp>
#include <kfr/dft.hpp>
#include <kfr/dsp.hpp>
#include <kfr/io.hpp>

using namespace kfr;

// -----------------------------------------------------------------------------
// C Function Wrappers
// -----------------------------------------------------------------------------

extern "C" {
void _mlir_ciface_fir_scalar(MemRef<float, 1> *inputMLIRFIR,
                             MemRef<float, 1> *kernelMLIRFIR,
                             MemRef<float, 1> *outputMLIRFIR);
void _mlir_ciface_fir_vectorization(MemRef<float, 1> *inputMLIRFIR,
                                    MemRef<float, 1> *kernelMLIRFIR,
                                    MemRef<float, 1> *outputMLIRFIR);
void _mlir_ciface_fir_tiled_vectorization(MemRef<float, 1> *inputMLIRFIR,
                                          MemRef<float, 1> *kernelMLIRFIR,
                                          MemRef<float, 1> *outputMLIRFIR);
}

// -----------------------------------------------------------------------------
// Helper Functions
// -----------------------------------------------------------------------------

namespace firOp {

// Initialize input and kernel data.
template <typename T, size_t N1, size_t N2>
void initializeKFRFIR(univector<T, N1> &input, univector<T, N2> &kernel) {
  // Decode audio data.
  audio_reader_wav<T> reader(open_file_for_reading(
      "../../benchmarks/AudioProcessing/Audios/NASA_Mars.wav"));
  reader.read(input.data(), input.size());
  // Generate kernel data.
  expression_pointer<T> kaiser =
      to_pointer(window_kaiser<T>(kernel.size(), 3.0));
  fir_lowpass(kernel, 0.2, kaiser, true);
}

// Print KFR univector result.
template <typename T, size_t N>
void printUnivector(const univector<T, N> &result, bool doPrint = false) {
  if (!doPrint)
    return;
  std::ofstream file("KFRFIRResult.txt");
  if (file.is_open()) {
    file << "[ KFR FIR Result Information ]" << std::endl;
    for (size_t i = 0; i < result.size(); ++i) {
      file << result[i] << std::endl;
    }
    file.close();
  } else {
    std::cerr << "Unable to open file for printing KFR FIR result."
              << std::endl;
  }
}

// Print MLIR based MemRef result.
template <typename T, size_t N>
void printMemRef(const MemRef<T, N> &result, const std::string &name = "",
                 bool doPrint = false) {
  if (!doPrint)
    return;
  std::string fileName = name + "FIRResult.txt";
  std::ofstream file(fileName);
  if (file.is_open()) {
    file << "[ " << name << " FIR Result Information ]" << std::endl;
    for (size_t i = 0; i < result.getSize(); ++i) {
      file << result[i] << std::endl;
    }
    file.close();
  } else {
    std::cerr << "Unable to open file for printing " << name << " FIR result."
              << std::endl;
  }
}

// Verify correctness of KFR vs. MLIR results using direct error.
template <typename T, size_t N>
void verify(const univector<T, N> &A, const MemRef<float, 1> &B, size_t size,
            const std::string &name) {
  // Tolerance for floating point comparison
  const double epsilon = 1e-2;
  bool isPass = true;
  const std::string PASS = "\033[32mPASS\033[0m";
  const std::string FAIL = "\033[31mFAIL\033[0m";

  // Print verification result.
  std::cout << name << " ";
  for (int i = 0; i < size; ++i) {
    if (std::fabs(A[i] - B[i]) > epsilon) {
      std::cout << FAIL << std::endl;
      std::cout << std::setprecision(15);
      std::cout << "i=" << i << ":\tA[" << i << "]=" << A[i] << "\tB[" << i
                << "]=" << B[i] << std::endl;
      isPass = false;
      break;
    }
  }
  if (isPass) {
    std::cout << PASS << std::endl;
  }
}
} // namespace firOp

#endif // FIR_UTILS_HPP
