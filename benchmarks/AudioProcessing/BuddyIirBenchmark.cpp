//===- BuddyIirBenchmark.cpp ----------------------------------------------===//
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
// This file implements the benchmark for Buddy IIR function.
//
//===----------------------------------------------------------------------===//

#include <benchmark/benchmark.h>
#include <buddy/Core/Container.h>
#include <kfr/base.hpp>
#include <kfr/dft.hpp>
#include <kfr/dsp.hpp>
#include <kfr/io.hpp>

using namespace kfr;

// Declare the IIR C interface.
extern "C" {
void _mlir_ciface_mlir_iir(MemRef<double, 1> *inputBuddyConv1D,
                           MemRef<double, 2> *kernelBuddyConv1D,
                           MemRef<double, 1> *outputBuddyConv1D);

void _mlir_ciface_mlir_iir_simd(MemRef<double, 1> *inputBuddyConv1D,
                                MemRef<double, 2> *kernelBuddyConv1D,
                                MemRef<double, 1> *outputBuddyConv1D);

// TODO : Debug buddy_iir lowering pass. The final PR will use version with type
// f64. void _mlir_ciface_buddy_iir(MemRef<double, 1> *inputBuddyConv1D,
//                             MemRef<double, 2> *kernelBuddyConv1D,
//                             MemRef<double, 1> *outputBuddyConv1D);

void _mlir_ciface_buddy_iir_simd(MemRef<double, 1> *inputBuddyConv1D,
                                 MemRef<double, 2> *kernelBuddyConv1D,
                                 MemRef<double, 1> *outputBuddyConv1D);
}

namespace {
univector<double> kernel;
zpk<double> filt = iir_lowpass(bessel<double>(24), 1000, 48000);
std::vector<biquad_params<double>> bqs = to_sos(filt);

univector<double, 2000000> aud_buddy_iir;
univector<double, 2000000> result_buddy_iir;
intptr_t size_f = bqs.size();
intptr_t sizeofKernel[2] = {size_f, 6};
intptr_t sizeofAud{aud_buddy_iir.size()};

// MemRef copys all data, so data here are actually not accessed.
MemRef<double, 2> kernelRef(sizeofKernel);
MemRef<double, 1> audRef(&sizeofAud);
MemRef<double, 1> resRef(&sizeofAud);
} // namespace

// Initialize univector.
void initializeBuddyIir() {
  audio_reader_wav<double> reader(open_file_for_reading(
      "../../benchmarks/AudioProcessing/Audios/NASA_Mars.wav"));
  reader.read(aud_buddy_iir.data(), aud_buddy_iir.size());

  for (int i = 0; i < bqs.size(); ++i) {
    auto bq = bqs[i];
    kernel.push_back(bq.b0);
    kernel.push_back(bq.b1);
    kernel.push_back(bq.b2);
    kernel.push_back(bq.a0);
    kernel.push_back(bq.a1);
    kernel.push_back(bq.a2);
  }

  kernelRef = std::move(MemRef<double, 2>(kernel.data(), sizeofKernel));
  audRef = std::move(MemRef<double, 1>(aud_buddy_iir.data(), &sizeofAud));
  resRef = std::move(MemRef<double, 1>(&sizeofAud));
}

// Benchmarking function.
static void MLIR_IIR(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      _mlir_ciface_mlir_iir(&audRef, &kernelRef, &resRef);
    }
  }
}

// Benchmarking function.
static void MLIR_IIR_SIMD(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      _mlir_ciface_mlir_iir_simd(&audRef, &kernelRef, &resRef);
    }
  }
}

// // Benchmarking function.
// static void BUDDY_IIR(benchmark::State &state) {
//   for (auto _ : state) {
//     for (int i = 0; i < state.range(0); ++i) {
//       _mlir_ciface_buddy_iir(&audRef, &kernelRef, &resRef);
//     }
//   }
// }

// Benchmarking function.
static void BUDDY_IIR_SIMD(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      _mlir_ciface_buddy_iir_simd(&audRef, &kernelRef, &resRef);
    }
  }
}

// Register benchmarking function.
BENCHMARK(MLIR_IIR)->Arg(1)->Unit(benchmark::kMillisecond);
BENCHMARK(MLIR_IIR_SIMD)->Arg(1)->Unit(benchmark::kMillisecond);
// BENCHMARK(BUDDY_IIR)->Arg(1)->Unit(benchmark::kMillisecond);
BENCHMARK(BUDDY_IIR_SIMD)->Arg(1)->Unit(benchmark::kMillisecond);

// Generate result_buddy_iir wav file.
void generateResultBuddyIir() {
  println("-------------------------------------------------------");
  println("[ Buddy IIR Result Information ]");
  MemRef<double, 1> generateResult(&sizeofAud);

  _mlir_ciface_buddy_iir_simd(&audRef, &kernelRef, &generateResult);

  // print output for the first 100 ouputs
  for (int i = 0; i < 100; ++i) {
    std::cout << " " << generateResult[i];
    if (i % 10 == 9)
      std::cout << "\n";
  }

  audio_writer_wav<double> writer(open_file_for_writing("./ResultBuddyIir.wav"),
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
