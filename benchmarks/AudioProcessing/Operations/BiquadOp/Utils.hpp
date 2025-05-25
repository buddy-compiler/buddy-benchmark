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
// This file implements helper functions for BiquadOp.
//
//===----------------------------------------------------------------------===//

#ifndef BIQUAD_UTILS_HPP
#define BIQUAD_UTILS_HPP

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
void _mlir_ciface_mlir_biquad(MemRef<float, 1> *inputBuddyBiquad,
                              MemRef<float, 1> *kernelBuddyBiquad,
                              MemRef<float, 1> *outputBuddyBiquad);
}

// -----------------------------------------------------------------------------
// Helper Functions
// -----------------------------------------------------------------------------

namespace biquadOp {

// Initialize input and kernel data.
template <typename T, size_t N>
void initializeKFRBiquad(univector<T, N> &input, biquad_params<T> &bq,
                         univector<T, 6> &kernel) {
  // Decode audio data.
  audio_reader_wav<T> reader(open_file_for_reading(
      "../../benchmarks/AudioProcessing/Audios/NASA_Mars.wav"));
  reader.read(input.data(), input.size());
  // Generate kernel data.
  bq = {biquad_lowpass(0.3, -1.0)};
  // Store kernel data in univector.
  kernel[0] = bq.b0;
  kernel[1] = bq.b1;
  kernel[2] = bq.b2;
  kernel[3] = bq.a0;
  kernel[4] = bq.a1;
  kernel[5] = bq.a2;
}

// Print KFR univector result.
template <typename T, size_t N>
void printUnivector(const univector<T, N> &result, bool doPrint = false) {
  if (!doPrint)
    return;
  std::ofstream file("KFRBiquadResult.txt");
  if (file.is_open()) {
    file << "[ KFR Biquad Result Information ]" << std::endl;
    for (size_t i = 0; i < result.size(); ++i) {
      file << result[i] << std::endl;
    }
    file.close();
  } else {
    std::cerr << "Unable to open file for printing KFR Biquad result."
              << std::endl;
  }
}

// Print MLIR based MemRef result.
template <typename T, size_t N>
void printMemRef(const MemRef<T, N> &result, const std::string &name = "",
                 bool doPrint = false) {
  if (!doPrint)
    return;
  std::string fileName = name + "BiquadResult.txt";
  std::ofstream file(fileName);
  if (file.is_open()) {
    file << "[ " << name << " Biquad Result Information ]" << std::endl;
    for (size_t i = 0; i < result.getSize(); ++i) {
      file << result[i] << std::endl;
    }
    file.close();
  } else {
    std::cerr << "Unable to open file for printing " << name
              << " Biquad result." << std::endl;
  }
}

// Verify correctness of KFR vs. MLIR results using relative error.
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
    if (std::fabs((A[i] - B[i]) / A[i]) > epsilon) {
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
} // namespace biquadOp

#endif // BIQUAD_UTILS_HPP
