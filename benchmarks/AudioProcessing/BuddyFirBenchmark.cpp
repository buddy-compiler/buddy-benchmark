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
#include "Utils/Container.h"
#include "Utils/Correctness.h"
#include <benchmark/benchmark.h>
#include <kfr/base.hpp>
#include <kfr/dft.hpp>
#include <kfr/dsp.hpp>
#include <kfr/io.hpp>

using namespace kfr;

extern "C" {
void _mlir_ciface_conv1d_buddy(MemRef<float, 1> *inputBuddyConv1D,
                               MemRef<float, 1> *kernelBuddyConv1D,
                               MemRef<float, 1> *outputBuddyConv1D);

void _mlir_ciface_conv1d_linalg(MemRef<float, 1> *inputBuddyConv1D,
                                MemRef<float, 1> *kernelBuddyConv1D,
                                MemRef<float, 1> *outputBuddyConv1D);
}
namespace {
univector<float, 1023> taps127;
univector<float, 2000000> aud;
univector<float> result;
int sizeofTaps{taps127.size()};
int sizeofAud{aud.size()};

// MemRef copys all data, so data here are actually not accessed.
MemRef<float, 1> taps127Ref(reinterpret_cast<intptr_t *>(&sizeofTaps));
MemRef<float, 1> audRef(reinterpret_cast<intptr_t *>(&sizeofAud));
MemRef<float, 1> resRef(reinterpret_cast<intptr_t *>(&sizeofAud));
}

// Initialize univector.
void initializeBuddyFir() {
  expression_pointer<float> kaiser =
      to_pointer(window_kaiser<float>(taps127.size(), 3.0));
  fir_lowpass(taps127, 0.2, kaiser, true);
  audio_reader_wav<float> reader(open_file_for_reading(
      "/Users/zircon/Works/Github/buddy-benchmark/benchmarks/AudioProcessing/Audios/NASA_Mars.wav"));
  reader.read(aud.data(), aud.size());
  taps127Ref = std::move(MemRef<float, 1>(taps127.data(),
                                          reinterpret_cast<intptr_t *>(&sizeofTaps)));
  audRef = std::move(MemRef<float, 1>(aud.data(),
                                      reinterpret_cast<intptr_t *>(&sizeofAud)));
  resRef = std::move(MemRef<float, 1>(reinterpret_cast<intptr_t *>(&sizeofAud)));
}

// Benchmarking function.
static void BUDDY_FIR(benchmark::State &state) {
  for (auto _: state) {
    for (int i = 0; i < state.range(0); ++i) {
      // result = kfr::fir(aud, taps127);
      _mlir_ciface_conv1d_buddy(&audRef, &taps127Ref, &resRef);
    }
  }
}

// Register benchmarking function.
BENCHMARK(BUDDY_FIR)->Arg(1);

// Generate result wav file.
void generateResultBuddyFir() {
  MemRef<float, 1> generateResult(reinterpret_cast<intptr_t *>(&sizeofAud));
  _mlir_ciface_conv1d_buddy(&audRef, &taps127Ref, &generateResult);
  univector<float> resultKFR = kfr::fir(aud, taps127);
  std::function<float(int)> memRefAcc = [&generateResult](int num) { return generateResult[num]; };
  std::function<float(int)> uniVecAcc = [&resultKFR](int num) { return resultKFR[num]; };
  auto dist = euclideanDistMono(uniVecAcc, memRefAcc, resultKFR.size());
  println("Euclidean dist:", dist);
  auto num = equivCheckMono(uniVecAcc, memRefAcc, resultKFR.size(), 0.02f);
  println("Different points bigger than 0.02:", num);

  // Notice: dtw will construct an array with sizeof(count*count)
  // It could be very big when comparing the whole sequence.
  auto dtw = DTWCheckMono(uniVecAcc, memRefAcc, 10000, 2);
  println("DTW euclidean dist:", dtw);
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
}
