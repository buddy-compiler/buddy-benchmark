//===- KFRFft.cpp ---------------------------------------------------------===//
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
// This file implements the benchmark for KFR Fft function.
//
//===----------------------------------------------------------------------===//

#include <benchmark/benchmark.h>
#include <kfr/base.hpp>
#include <kfr/dft.hpp>
#include <kfr/dsp.hpp>
#include <kfr/io.hpp>
#include <kfr/simd/complex.hpp>
#include <kfr/dft/reference_dft.hpp>
using namespace kfr;

dft_plan_real<float> plan(2000000);   // dft_plan_real for real transform
univector<float, 2000000> aud_fft;
univector<complex<float>, 2000000> freq;
univector<u8> temp(plan.temp_size);

// Initialize univector.
void initializeKFRFft() {
  audio_reader_wav<float> reader(open_file_for_reading(
      "../../benchmarks/AudioProcessing/Audios/NASA_Mars.wav"));
  reader.read(aud_fft.data(), aud_fft.size());
}

// Benchmarking function.
static void KFR_FFT(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      plan.execute(freq, aud_fft, temp);
    }
  }
}

// Register benchmarking function.
BENCHMARK(KFR_FFT)->Arg(1)->Unit(benchmark::kMillisecond);

void generateResultKFRFft() {
  println("-------------------------------------------------------");
  println("[ KFR FFT Result Information ]");
  println("FFT operation finished!");
}