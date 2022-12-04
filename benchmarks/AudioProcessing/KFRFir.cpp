//===- KFRFir.cpp ---------------------------------------------------------===//
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
// This file implements the benchmark for KFR Fir function.
//
//===----------------------------------------------------------------------===//

#include <benchmark/benchmark.h>
#include <kfr/base.hpp>
#include <kfr/dft.hpp>
#include <kfr/dsp.hpp>
#include <kfr/io.hpp>

using namespace kfr;

univector<fbase, 1023> taps127;
univector<float, 2000000> aud;
univector<float> result;

// Initialize univector.
void initializeKFRFir() {
  expression_pointer<fbase> kaiser =
      to_pointer(window_kaiser(taps127.size(), 3.0));
  fir_lowpass(taps127, 0.2, kaiser, true);
  audio_reader_wav<float> reader(open_file_for_reading(
      "../../benchmarks/AudioProcessing/Audios/NASA_Mars.wav"));
  reader.read(aud.data(), aud.size());
}

// Benchmarking function.
static void KFR_FIR(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      result = kfr::fir(aud, taps127);
    }
  }
}

// Register benchmarking function.
BENCHMARK(KFR_FIR)->Arg(1)->Unit(benchmark::kMillisecond);

// Generate result wav file.
void generateResultKFRFir() {
  println("-------------------------------------------------------");
  println("[ KFR FIR Result Information ]");
  univector<float> generateResult = kfr::fir(aud, taps127);

  audio_writer_wav<float> writer(open_file_for_writing("./ResultKFRFir.wav"),
                                 audio_format{1 /* channel */,
                                              audio_sample_type::i24,
                                              100000 /* sample rate */});
  writer.write(generateResult.data(), generateResult.size());
  println("Sample Rate  = ", writer.format().samplerate);
  println("Channels     = ", writer.format().channels);
  println("Length       = ", writer.format().length);
  println("Duration (s) = ",
          writer.format().length / writer.format().samplerate);
  println("Bit depth    = ", audio_sample_bit_depth(writer.format().type));
}
