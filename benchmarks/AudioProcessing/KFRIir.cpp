//===- KFRIir.cpp ---------------------------------------------------------===//
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
// This file implements the benchmark for KFR Iir function.
//
//===----------------------------------------------------------------------===//

#include <benchmark/benchmark.h>
#include <kfr/base.hpp>
#include <kfr/dft.hpp>
#include <kfr/dsp.hpp>
#include <kfr/io.hpp>

using namespace kfr;

constexpr size_t maxorder = 32;
univector<float, 2000000> aud_iir;
univector<fbase, 1024> output;

zpk<fbase> filt = iir_lowpass(bessel<fbase>(24), 1000, 48000);
std::vector<biquad_params<fbase>> bqs = to_sos(filt);

// Initialize univector.
void initializeKFRIir() {
  audio_reader_wav<float> reader(open_file_for_reading(
      "../../benchmarks/AudioProcessing/Audios/NASA_Mars.wav"));
  reader.read(aud_iir.data(), aud_iir.size());
}

// Benchmarking function.
static void KFR_IIR(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      output = biquad<maxorder>(bqs, unitimpulse());
    }
  }
}

// Register benchmarking function.
BENCHMARK(KFR_IIR)->Arg(1);

// Generate result wav file.
void generateResultKFRIir() {
  println("IIR operation finished!");
}
