//===- BuddyBiquadBenchmark.cpp ---------------------------------------------------------===//
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
// This file implements the benchmark for Buddy Biquad IIR function.
//
//===----------------------------------------------------------------------===//

#include "Utils/Container.h"
#include <benchmark/benchmark.h>
#include <kfr/base.hpp>
#include <kfr/dft.hpp>
#include <kfr/dsp.hpp>
#include <kfr/io.hpp>

using namespace kfr;

extern "C" {
void _mlir_ciface_buddy_biquad(MemRef<float, 1> *inputBuddyConv1D,
                               MemRef<float, 1> *kernelBuddyConv1D,
                               MemRef<float, 1> *outputBuddyConv1D);

}
namespace {
univector<float,5>kernel = {0.226053, 0.452106, 0.226053, -0.404376, 0.308588}; 
univector<float, 2000000> aud;
univector<float> result;
int sizeofKernel = kernel.size();
int sizeofAud{aud.size()};

// MemRef copys all data, so data here are actually not accessed.
MemRef<float,1> kernelRef(reinterpret_cast<intptr_t *>(&sizeofKernel));
MemRef<float,1> audRef(reinterpret_cast<intptr_t *>(&sizeofAud));
MemRef<float,1> resRef(reinterpret_cast<intptr_t *>(&sizeofAud));
}

// Initialize univector.
void initializeBuddyBiquad() {
  audio_reader_wav<float> reader(open_file_for_reading(
      "../../benchmarks/AudioProcessing/Audios/NASA_Mars.wav"));
  reader.read(aud.data(), aud.size());
  kernelRef=std::move(MemRef<float,1>(kernel.data(),
                              reinterpret_cast<intptr_t *>(&sizeofKernel)));
  audRef=std::move(MemRef<float,1>(aud.data(),
                          reinterpret_cast<intptr_t *>(&sizeofAud)));
  resRef=std::move(MemRef<float,1>(reinterpret_cast<intptr_t *>(&sizeofAud)));
}

// Benchmarking function.
static void BUDDY_BIQUAD(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      // result = kfr::fir(aud, taps127);
      _mlir_ciface_buddy_biquad(&audRef,&kernelRef,&resRef);
    }
  }
}

// Register benchmarking function.
BENCHMARK(BUDDY_BIQUAD)->Arg(1);

// Generate result wav file.
void generateResultBuddyBiquad() {
  MemRef<float,1> generateResult(reinterpret_cast<intptr_t *>(&sizeofAud));
  _mlir_ciface_buddy_biquad(&audRef,&kernelRef,&generateResult);

  audio_writer_wav<float> writer(open_file_for_writing("./ResultBuddyBiqaud.wav"),
                                 audio_format{1 /* channel */,
                                              audio_sample_type::i24,
                                              100000 /* sample rate */});
  writer.write(generateResult.getData(), generateResult.getSize());
  println("Sample Rate  = ", writer.format().samplerate);
  println("Channels     = ", writer.format().channels);
  println("Length       = ", writer.format().length);
  println("Duration (s) = ",
          writer.format().length / writer.format().samplerate);
  println("Bit depth    = ", audio_sample_bit_depth(writer.format().type));
}