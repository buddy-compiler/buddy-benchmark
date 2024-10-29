//===- BuddyFIRBenchmark.cpp ----------------------------------------------===//
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
void _mlir_ciface_mlir_fir(MemRef<float, 1> *inputBuddyConv1D,
                           MemRef<float, 1> *kernelBuddyConv1D,
                           MemRef<float, 1> *outputBuddyConv1D);
}

// TODO: Add FIR benchmark
// namespace {
// univector<float, 6> kernel;
// biquad_params<float> bq = {biquad_lowpass(0.3, -1.0)};

// univector<float, 2000000> aud_buddy_fir;
// univector<float, 2000000> result_buddy_fir;
// intptr_t sizeofKernel{kernel.size()};
// intptr_t sizeofAud{aud_buddy_fir.size()};

// // MemRef copys all data, so data here are actually not accessed.
// MemRef<float, 1> kernelRef(&sizeofKernel);
// MemRef<float, 1> audRef(&sizeofAud);
// MemRef<float, 1> resRef(&sizeofAud);
// } // namespace

// // Initialize univector.
// void initializeBuddyFIR() {
//   audio_reader_wav<float> reader_buddy_fir(open_file_for_reading(
//       "../../benchmarks/AudioProcessing/Audios/NASA_Mars.wav"));
//   reader_buddy_fir.read(aud_buddy_fir.data(), aud_buddy_fir.size());

//   kernel[0] = bq.b0;
//   kernel[1] = bq.b1;
//   kernel[2] = bq.b2;
//   kernel[3] = bq.a0;
//   kernel[4] = bq.a1;
//   kernel[5] = bq.a2;

//   kernelRef = std::move(MemRef<float, 1>(kernel.data(), &sizeofKernel));
//   audRef = std::move(MemRef<float, 1>(aud_buddy_fir.data(), &sizeofAud));
//   resRef = std::move(MemRef<float, 1>(&sizeofAud));
// }

// // Benchmarking function.
// static void MLIR_FIR(benchmark::State &state) {
//   for (auto _ : state) {
//     for (int i = 0; i < state.range(0); ++i) {
//       _mlir_ciface_mlir_fir(&audRef, &kernelRef, &resRef);
//     }
//   }
// }

// // Register benchmarking function.
// BENCHMARK(MLIR_FIR)->Arg(1)->Unit(benchmark::kMillisecond);

// // Generate result_buddy_fir wav file.
// void generateResultBuddyFIR() {
//   println("-------------------------------------------------------");
//   println("[ Buddy Biquad Result Information ]");
//   MemRef<float, 1> generateResult(&sizeofAud);
//   _mlir_ciface_mlir_fir(&audRef, &kernelRef, &generateResult);

//   audio_writer_wav<float> writer(
//       open_file_for_writing("./ResultBuddyFIR.wav"),
//       audio_format{1 /* channel */, audio_sample_type::i24,
//                    100000 /* sample rate */});
//   writer.write(generateResult.getData(), generateResult.getSize());
//   println("Sample Rate  = ", writer.format().samplerate);
//   println("Channels     = ", writer.format().channels);
//   println("Length       = ", writer.format().length);
//   println("Duration (s) = ",
//           writer.format().length / writer.format().samplerate);
//   println("Bit depth    = ", audio_sample_bit_depth(writer.format().type));
//   writer.close();
// }
