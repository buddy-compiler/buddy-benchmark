//===- BuddyFirBenchmark.cpp ----------------------------------------------===//
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
// This file implements the benchmark for Buddy FIR function.
//
//===----------------------------------------------------------------------===//

#include <buddy/Core/Container.h>
#include <benchmark/benchmark.h>
#include <kfr/base.hpp>
#include <kfr/dft.hpp>
#include <kfr/dsp.hpp>
#include <kfr/io.hpp>


using namespace kfr;

// Declare the FIR C interface.
extern "C" {
void _mlir_ciface_conv1d_buddy(MemRef<float, 1> *inputBuddyConv1D,
                               MemRef<float, 1> *kernelBuddyConv1D,
                               MemRef<float, 1> *outputBuddyConv1D);

void _mlir_ciface_conv1d_linalg(MemRef<float, 1> *inputBuddyConv1D,
                                MemRef<float, 1> *kernelBuddyConv1D,
                                MemRef<float, 1> *outputBuddyConv1D);
}

namespace {

univector<float, 2000000> aud_buddy_fir;
univector<float, 2000000> result_buddy_fir;
univector<float, 127> taps127;
intptr_t sizeofKernel{taps127.size()};
intptr_t sizeofAud{aud_buddy_fir.size()};

// MemRef copys all data, so data here are actually not accessed.
MemRef<float, 1> audRef(&sizeofAud);
MemRef<float, 1> resRef(&sizeofAud);
MemRef<float, 1> kernelRef(&sizeofKernel);
} // namespace

// Initialize univector.
void initializeBuddyFir() {
  audio_reader_wav<float> reader(open_file_for_reading(
      "../../benchmarks/AudioProcessing/Audios/NASA_Mars.wav"));
  reader.read(aud_buddy_fir.data(), aud_buddy_fir.size());

  expression_handle<float> kaiser =
      to_handle(window_kaiser<float>(taps127.size(), 3.0));
  fir_lowpass(taps127, 0.2, kaiser, true);

  kernelRef = std::move(MemRef<float, 1>(taps127.data(), &sizeofKernel));
  audRef = std::move(MemRef<float, 1>(aud_buddy_fir.data(), &sizeofAud));
  resRef = std::move(MemRef<float, 1>(&sizeofAud));
}

// Benchmarking function.
static void Linalg_FIR(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      _mlir_ciface_conv1d_linalg(&audRef, &kernelRef, &resRef);
    }
  }
}

// Benchmarking function.
static void BUDDY_FIR(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      _mlir_ciface_conv1d_buddy(&audRef, &kernelRef, &resRef);
    }
  }
}

// Register benchmarking function.
BENCHMARK(Linalg_FIR)->Arg(1)->Unit(benchmark::kMillisecond);
BENCHMARK(BUDDY_FIR)->Arg(1)->Unit(benchmark::kMillisecond);

// Generate result_buddy_fir wav file.
void generateResultBuddyFir() {
  println("-------------------------------------------------------");
  println("[ Buddy FIR Result Information ]");
  MemRef<float, 1> generateResult(&sizeofAud);
  _mlir_ciface_conv1d_buddy(&audRef, &kernelRef, &generateResult);

  audio_writer_wav<float> writer(open_file_for_writing("./ResultBuddyFir.wav"),
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
  writer.close();
}