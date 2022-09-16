//===- KFRBiquad.cpp
//---------------------------------------------------------===//
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
// This file implements the benchmark for KFR Biquad function.
//
//===----------------------------------------------------------------------===//

#include <benchmark/benchmark.h>
#include <kfr/all.hpp>
#include <kfr/base.hpp>
#include <kfr/dft.hpp>
#include <kfr/dsp.hpp>
#include <kfr/dsp/biquad.hpp>
#include <kfr/io.hpp>
#include <kfr/math.hpp>

using namespace kfr;

univector<float, 2000000> aud_biquad;
biquad_params<float> bq = {biquad_lowpass(0.3, -1.0)};
univector<float, 2000000> result_biquad;
// Initialize univector.
void initializeKFRBiquad() {
  audio_reader_wav<float> reader(open_file_for_reading(
      "../../benchmarks/AudioProcessing/Audios/NASA_Mars.wav"));
  reader.read(aud_biquad.data(), aud_biquad.size());
}

// Benchmarking function.
static void KFR_Biquad(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      result_biquad = biquad(bq, aud_biquad);
    }
  }
}

// Register benchmarking function.
BENCHMARK(KFR_Biquad)->Arg(1);

void generateResultKFRBiquad() {
  println("-------------------------------------------------------");
  println("[ KFR Biquad Result Information ]");
  result_biquad = biquad(bq, aud_biquad);
  audio_writer_wav<float> writer(open_file_for_writing("./ResultKFRBiqaud.wav"),
                                 audio_format{1 /* channel */,
                                              audio_sample_type::i24,
                                              100000 /* sample rate */});
  writer.write(result_biquad.data(), result_biquad.size());
  println("Biquad operation finished!");
}
