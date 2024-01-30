//===- BuddyMorph2DBenchmark.cpp ------------------------------------------===//
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
// This file implements the benchmark for Morph2D operation.
//
//===----------------------------------------------------------------------===//

#include "Kernels.h"
#include <benchmark/benchmark.h>
#include <buddy/Core/Container.h>
#include <buddy/DIP/DIP.h>
#include <buddy/DIP/ImageContainer.h>
#include <buddy/DIP/imgcodecs/loadsave.h>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

// Declare input image and kernel.
Mat inputImageBuddyMorph2D;

// Name of the input image to be read.
std::string inputNameBuddyMorph2D;

// Define the kernel size.
float *kernelDataBuddyMorph2D;
int kernelRowsBuddyMorph2D, kernelColsBuddyMorph2D;

// Define the output size.
int outputRowsBuddyMorph2D, outputColsBuddyMorph2D;

// Define sizes of input, kernel, and output.
intptr_t sizesInputBuddyMorph2D[2];
intptr_t sizesKernelBuddyMorph2D[2];
intptr_t sizesOutputBuddyMorph2D[2];

// Declare Boundary Options supported.
enum BoundaryOption { constant_padding, replicate_padding };

// Define Boundary option selected.
BoundaryOption BoundaryType1;

void initializeBuddyMorph2D(char **argv, Img<float, 2> inputImageBuddyMorph2D) {
  inputNameBuddyMorph2D = argv[1];

  kernelDataBuddyMorph2D = get<0>(kernelMap[argv[2]]);
  kernelRowsBuddyMorph2D = get<1>(kernelMap[argv[2]]);
  kernelColsBuddyMorph2D = get<2>(kernelMap[argv[2]]);

  outputRowsBuddyMorph2D = inputImageBuddyMorph2D.getSizes()[0];
  outputColsBuddyMorph2D = inputImageBuddyMorph2D.getSizes()[1];

  sizesInputBuddyMorph2D[0] = inputImageBuddyMorph2D.getSizes()[0];
  sizesInputBuddyMorph2D[1] = inputImageBuddyMorph2D.getSizes()[1];

  sizesKernelBuddyMorph2D[0] = kernelRowsBuddyMorph2D;
  sizesKernelBuddyMorph2D[1] = kernelColsBuddyMorph2D;

  sizesOutputBuddyMorph2D[0] = outputRowsBuddyMorph2D;
  sizesOutputBuddyMorph2D[1] = outputColsBuddyMorph2D;

  if (static_cast<string>(argv[3]) == "REPLICATE_PADDING") {
    BoundaryType1 = replicate_padding;
  } else {
    BoundaryType1 = constant_padding;
  }
}

MemRef<float, 2> kernelBuddyMorph2D(kernelDataBuddyMorph2D,
                                    sizesKernelBuddyMorph2D);
static void Buddy_Erosion2D_Constant_Padding(benchmark::State &state) {
  // Define the MemRef descriptor for input, kernel, and output.
  Img<float, 2> inputBuddyMorph2D =
      dip::imread<float, 2>(inputNameBuddyMorph2D, dip::IMGRD_GRAYSCALE);
  MemRef<float, 2> outputBuddyErosion2D(sizesOutputBuddyMorph2D);
  MemRef<float, 2> copyMemRefErosion2D(sizesOutputBuddyMorph2D, 256.f);

  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      dip::detail::_mlir_ciface_erosion_2d_constant_padding(
          inputBuddyMorph2D, &kernelBuddyMorph2D, &outputBuddyErosion2D,
          &copyMemRefErosion2D, 1 /* Center X */, 1 /* Center Y */, 1,
          0.0f /* Constant Value */);
    }
  }
}

static void Buddy_Erosion2D_Replicate_Padding(benchmark::State &state) {
  // Define the MemRef descriptor for input, kernel, and output.
  Img<float, 2> inputBuddyMorph2D =
      dip::imread<float, 2>(inputNameBuddyMorph2D, dip::IMGRD_GRAYSCALE);
  MemRef<float, 2> outputBuddyErosion2D(sizesOutputBuddyMorph2D);
  MemRef<float, 2> copyMemRefErosion2D(sizesOutputBuddyMorph2D, 256.f);

  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      dip::detail::_mlir_ciface_erosion_2d_replicate_padding(
          inputBuddyMorph2D, &kernelBuddyMorph2D, &outputBuddyErosion2D,
          &copyMemRefErosion2D, 1 /* Center X */, 1 /* Center Y */, 1,
          0.0f /* Constant Value */);
    }
  }
}

static void Buddy_Dilation2D_Constant_Padding(benchmark::State &state) {
  // Define the MemRef descriptor for input, kernel, and output.
  Img<float, 2> inputBuddyMorph2D =
      dip::imread<float, 2>(inputNameBuddyMorph2D, dip::IMGRD_GRAYSCALE);
  MemRef<float, 2> outputBuddyDilation2D(sizesOutputBuddyMorph2D);
  MemRef<float, 2> copyMemRefDilation2D(sizesOutputBuddyMorph2D, -1.f);

  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      dip::detail::_mlir_ciface_dilation_2d_constant_padding(
          inputBuddyMorph2D, &kernelBuddyMorph2D, &outputBuddyDilation2D,
          &copyMemRefDilation2D, 1 /* Center X */, 1 /* Center Y */, 1,
          0.0f /* Constant Value */);
    }
  }
}

static void Buddy_Dilation2D_Replicate_Padding(benchmark::State &state) {
  // Define the MemRef descriptor for input, kernel, and output.
  Img<float, 2> inputBuddyMorph2D =
      dip::imread<float, 2>(inputNameBuddyMorph2D, dip::IMGRD_GRAYSCALE);
  MemRef<float, 2> outputBuddyDilation2D(sizesOutputBuddyMorph2D);
  MemRef<float, 2> copyMemRefDilation2D(sizesOutputBuddyMorph2D, -1.f);

  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      dip::detail::_mlir_ciface_dilation_2d_replicate_padding(
          inputBuddyMorph2D, &kernelBuddyMorph2D, &outputBuddyDilation2D,
          &copyMemRefDilation2D, 1 /* Center X */, 1 /* Center Y */, 1,
          0.0f /* Constant Value */);
    }
  }
}

static void Buddy_Opening2D_Constant_Padding(benchmark::State &state) {
  // Define the MemRef descriptor for input, kernel, and output.
  Img<float, 2> inputBuddyMorph2D =
      dip::imread<float, 2>(inputNameBuddyMorph2D, dip::IMGRD_GRAYSCALE);
  MemRef<float, 2> outputBuddyOpening2D(sizesOutputBuddyMorph2D);
  MemRef<float, 2> outputBuddyOpening2D1(sizesOutputBuddyMorph2D);
  MemRef<float, 2> copyMemRefOpening2D(sizesOutputBuddyMorph2D, -1.f);
  MemRef<float, 2> copyMemRefOpening2D1(sizesOutputBuddyMorph2D, 256.f);

  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      dip::detail::_mlir_ciface_opening_2d_constant_padding(
          inputBuddyMorph2D, &kernelBuddyMorph2D, &outputBuddyOpening2D,
          &outputBuddyOpening2D1, &copyMemRefOpening2D, &copyMemRefOpening2D1,
          1 /* Center X */, 1 /* Center Y */, 1, 0.0f /* Constant Value */);
    }
  }
}

static void Buddy_Opening2D_Replicate_Padding(benchmark::State &state) {
  // Define the MemRef descriptor for input, kernel, and output.
  Img<float, 2> inputBuddyMorph2D =
      dip::imread<float, 2>(inputNameBuddyMorph2D, dip::IMGRD_GRAYSCALE);
  MemRef<float, 2> outputBuddyOpening2D(sizesOutputBuddyMorph2D);
  MemRef<float, 2> outputBuddyOpening2D1(sizesOutputBuddyMorph2D);
  MemRef<float, 2> copyMemRefOpening2D(sizesOutputBuddyMorph2D, -1.f);
  MemRef<float, 2> copyMemRefOpening2D1(sizesOutputBuddyMorph2D, 256.f);

  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      dip::detail::_mlir_ciface_opening_2d_replicate_padding(
          inputBuddyMorph2D, &kernelBuddyMorph2D, &outputBuddyOpening2D,
          &outputBuddyOpening2D1, &copyMemRefOpening2D, &copyMemRefOpening2D1,
          1 /* Center X */, 1 /* Center Y */, 1, 0.0f /* Constant Value */);
    }
  }
}

static void Buddy_Closing2D_Constant_Padding(benchmark::State &state) {
  // Define the MemRef descriptor for input, kernel, and output.
  Img<float, 2> inputBuddyMorph2D =
      dip::imread<float, 2>(inputNameBuddyMorph2D, dip::IMGRD_GRAYSCALE);
  MemRef<float, 2> outputBuddyClosing2D(sizesOutputBuddyMorph2D);
  MemRef<float, 2> outputBuddyClosing2D1(sizesOutputBuddyMorph2D);
  MemRef<float, 2> copyMemRefClosing2D(sizesOutputBuddyMorph2D, -1.f);
  MemRef<float, 2> copyMemRefClosing2D1(sizesOutputBuddyMorph2D, 256.f);

  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      dip::detail::_mlir_ciface_closing_2d_constant_padding(
          inputBuddyMorph2D, &kernelBuddyMorph2D, &outputBuddyClosing2D,
          &outputBuddyClosing2D1, &copyMemRefClosing2D, &copyMemRefClosing2D1,
          1 /* Center X */, 1 /* Center Y */, 1, 0.0f /* Constant Value */);
    }
  }
}

static void Buddy_Closing2D_Replicate_Padding(benchmark::State &state) {
  // Define the MemRef descriptor for input, kernel, and output.
  Img<float, 2> inputBuddyMorph2D =
      dip::imread<float, 2>(inputNameBuddyMorph2D, dip::IMGRD_GRAYSCALE);
  MemRef<float, 2> outputBuddyClosing2D(sizesOutputBuddyMorph2D);
  MemRef<float, 2> outputBuddyClosing2D1(sizesOutputBuddyMorph2D);
  MemRef<float, 2> copyMemRefClosing2D(sizesOutputBuddyMorph2D, -1.f);
  MemRef<float, 2> copyMemRefClosing2D1(sizesOutputBuddyMorph2D, 256.f);

  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      dip::detail::_mlir_ciface_closing_2d_replicate_padding(
          inputBuddyMorph2D, &kernelBuddyMorph2D, &outputBuddyClosing2D,
          &outputBuddyClosing2D1, &copyMemRefClosing2D, &copyMemRefClosing2D1,
          1 /* Center X */, 1 /* Center Y */, 1, 0.0f /* Constant Value */);
    }
  }
}

static void Buddy_TopHat2D_Constant_Padding(benchmark::State &state) {
  // Define the MemRef descriptor for input, kernel, and output.
  Img<float, 2> inputBuddyMorph2D =
      dip::imread<float, 2>(inputNameBuddyMorph2D, dip::IMGRD_GRAYSCALE);
  MemRef<float, 2> inputBuddyTopHat2D1(sizesInputBuddyMorph2D);
  MemRef<float, 2> outputBuddyTopHat2D(sizesOutputBuddyMorph2D);
  MemRef<float, 2> outputBuddyTopHat2D1(sizesOutputBuddyMorph2D);
  MemRef<float, 2> outputBuddyTopHat2D2(sizesOutputBuddyMorph2D);
  MemRef<float, 2> copyMemRefTopHat2D(sizesOutputBuddyMorph2D, 256.f);
  MemRef<float, 2> copyMemRefTopHat2D1(sizesOutputBuddyMorph2D, -1.f);

  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      dip::detail::_mlir_ciface_tophat_2d_constant_padding(
          inputBuddyMorph2D, &kernelBuddyMorph2D, &outputBuddyTopHat2D,
          &outputBuddyTopHat2D1, &outputBuddyTopHat2D2, &inputBuddyTopHat2D1,
          &copyMemRefTopHat2D, &copyMemRefTopHat2D1, 1 /* Center X */,
          1 /* Center Y */, 1, 0.0f /* Constant Value */);
    }
  }
}

static void Buddy_TopHat2D_Replicate_Padding(benchmark::State &state) {
  // Define the MemRef descriptor for input, kernel, and output.
  Img<float, 2> inputBuddyMorph2D =
      dip::imread<float, 2>(inputNameBuddyMorph2D, dip::IMGRD_GRAYSCALE);
  MemRef<float, 2> inputBuddyTopHat2D1(sizesInputBuddyMorph2D);
  MemRef<float, 2> outputBuddyTopHat2D(sizesOutputBuddyMorph2D);
  MemRef<float, 2> outputBuddyTopHat2D1(sizesOutputBuddyMorph2D);
  MemRef<float, 2> outputBuddyTopHat2D2(sizesOutputBuddyMorph2D);
  MemRef<float, 2> copyMemRefTopHat2D(sizesOutputBuddyMorph2D, 256.f);
  MemRef<float, 2> copyMemRefTopHat2D1(sizesOutputBuddyMorph2D, -1.f);

  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      dip::detail::_mlir_ciface_tophat_2d_replicate_padding(
          inputBuddyMorph2D, &kernelBuddyMorph2D, &outputBuddyTopHat2D,
          &outputBuddyTopHat2D1, &outputBuddyTopHat2D2, &inputBuddyTopHat2D1,
          &copyMemRefTopHat2D, &copyMemRefTopHat2D1, 1 /* Center X */,
          1 /* Center Y */, 1, 0.0f /* Constant Value */);
    }
  }
}

static void Buddy_BottomHat2D_Constant_Padding(benchmark::State &state) {
  // Define the MemRef descriptor for input, kernel, and output.
  Img<float, 2> inputBuddyMorph2D =
      dip::imread<float, 2>(inputNameBuddyMorph2D, dip::IMGRD_GRAYSCALE);
  MemRef<float, 2> inputBuddyBottomHat2D1(sizesInputBuddyMorph2D);
  MemRef<float, 2> outputBuddyBottomHat2D(sizesOutputBuddyMorph2D);
  MemRef<float, 2> outputBuddyBottomHat2D1(sizesOutputBuddyMorph2D);
  MemRef<float, 2> outputBuddyBottomHat2D2(sizesOutputBuddyMorph2D);
  MemRef<float, 2> copyMemRefBottomHat2D(sizesOutputBuddyMorph2D, 256.f);
  MemRef<float, 2> copyMemRefBottomHat2D1(sizesOutputBuddyMorph2D, -1.f);

  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      dip::detail::_mlir_ciface_bottomhat_2d_constant_padding(
          inputBuddyMorph2D, &kernelBuddyMorph2D, &outputBuddyBottomHat2D,
          &outputBuddyBottomHat2D1, &outputBuddyBottomHat2D2,
          &inputBuddyBottomHat2D1, &copyMemRefBottomHat2D,
          &copyMemRefBottomHat2D1, 1 /* Center X */, 1 /* Center Y */, 1,
          0.0f /* Constant Value */);
    }
  }
}

static void Buddy_BottomHat2D_Replicate_Padding(benchmark::State &state) {
  // Define the MemRef descriptor for input, kernel, and output.
  Img<float, 2> inputBuddyMorph2D =
      dip::imread<float, 2>(inputNameBuddyMorph2D, dip::IMGRD_GRAYSCALE);
  MemRef<float, 2> inputBuddyBottomHat2D1(sizesInputBuddyMorph2D);
  MemRef<float, 2> outputBuddyBottomHat2D(sizesOutputBuddyMorph2D);
  MemRef<float, 2> outputBuddyBottomHat2D1(sizesOutputBuddyMorph2D);
  MemRef<float, 2> outputBuddyBottomHat2D2(sizesOutputBuddyMorph2D);
  MemRef<float, 2> copyMemRefBottomHat2D(sizesOutputBuddyMorph2D, 256.f);
  MemRef<float, 2> copyMemRefBottomHat2D1(sizesOutputBuddyMorph2D, -1.f);

  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      dip::detail::_mlir_ciface_bottomhat_2d_replicate_padding(
          inputBuddyMorph2D, &kernelBuddyMorph2D, &outputBuddyBottomHat2D,
          &outputBuddyBottomHat2D1, &outputBuddyBottomHat2D2,
          &inputBuddyBottomHat2D1, &copyMemRefBottomHat2D,
          &copyMemRefBottomHat2D1, 1 /* Center X */, 1 /* Center Y */, 1,
          0.0f /* Constant Value */);
    }
  }
}

static void Buddy_MorphGrad2D_Constant_Padding(benchmark::State &state) {
  // Define the MemRef descriptor for input, kernel, and output.
  Img<float, 2> inputBuddyMorph2D =
      dip::imread<float, 2>(inputNameBuddyMorph2D, dip::IMGRD_GRAYSCALE);
  MemRef<float, 2> inputBuddyMorphGrad2D1(sizesInputBuddyMorph2D);
  MemRef<float, 2> outputBuddyMorphGrad2D(sizesOutputBuddyMorph2D);
  MemRef<float, 2> outputBuddyMorphGrad2D1(sizesOutputBuddyMorph2D);
  MemRef<float, 2> outputBuddyMorphGrad2D2(sizesOutputBuddyMorph2D);
  MemRef<float, 2> copyMemRefMorphGrad2D(sizesOutputBuddyMorph2D, 256.f);
  MemRef<float, 2> copyMemRefMorphGrad2D1(sizesOutputBuddyMorph2D, -1.f);

  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      dip::detail::_mlir_ciface_morphgrad_2d_constant_padding(
          inputBuddyMorph2D, &kernelBuddyMorph2D, &outputBuddyMorphGrad2D,
          &outputBuddyMorphGrad2D1, &outputBuddyMorphGrad2D2,
          &inputBuddyMorphGrad2D1, &copyMemRefMorphGrad2D,
          &copyMemRefMorphGrad2D1, 1 /* Center X */, 1 /* Center Y */, 1,
          0.0f /* Constant Value */);
    }
  }
}

static void Buddy_MorphGrad2D_Replicate_Padding(benchmark::State &state) {
  // Define the MemRef descriptor for input, kernel, and output.
  Img<float, 2> inputBuddyMorph2D =
      dip::imread<float, 2>(inputNameBuddyMorph2D, dip::IMGRD_GRAYSCALE);
  MemRef<float, 2> inputBuddyMorphGrad2D1(sizesInputBuddyMorph2D);
  MemRef<float, 2> outputBuddyMorphGrad2D(sizesOutputBuddyMorph2D);
  MemRef<float, 2> outputBuddyMorphGrad2D1(sizesOutputBuddyMorph2D);
  MemRef<float, 2> outputBuddyMorphGrad2D2(sizesOutputBuddyMorph2D);
  MemRef<float, 2> copyMemRefMorphGrad2D(sizesOutputBuddyMorph2D, 256.f);
  MemRef<float, 2> copyMemRefMorphGrad2D1(sizesOutputBuddyMorph2D, -1.f);

  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      dip::detail::_mlir_ciface_morphgrad_2d_replicate_padding(
          inputBuddyMorph2D, &kernelBuddyMorph2D, &outputBuddyMorphGrad2D,
          &outputBuddyMorphGrad2D1, &outputBuddyMorphGrad2D2,
          &inputBuddyMorphGrad2D1, &copyMemRefMorphGrad2D,
          &copyMemRefMorphGrad2D1, 1 /* Center X */, 1 /* Center Y */, 1,
          0.0f /* Constant Value */);
    }
  }
}

// Register benchmarking function.
void registerBenchmarkBuddyErosion2D() {
  if (BoundaryType1 == replicate_padding) {
    BENCHMARK(Buddy_Erosion2D_Replicate_Padding)
        ->Arg(1)
        ->Unit(benchmark::kMillisecond);
  } else {
    BENCHMARK(Buddy_Erosion2D_Constant_Padding)
        ->Arg(1)
        ->Unit(benchmark::kMillisecond);
  }
}

// Register benchmarking function.
void registerBenchmarkBuddyDilation2D() {
  if (BoundaryType1 == replicate_padding) {
    BENCHMARK(Buddy_Dilation2D_Replicate_Padding)
        ->Arg(1)
        ->Unit(benchmark::kMillisecond);
  } else {
    BENCHMARK(Buddy_Dilation2D_Constant_Padding)
        ->Arg(1)
        ->Unit(benchmark::kMillisecond);
  }
}

// Register benchmarking function.
void registerBenchmarkBuddyOpening2D() {
  if (BoundaryType1 == replicate_padding) {
    BENCHMARK(Buddy_Opening2D_Replicate_Padding)
        ->Arg(1)
        ->Unit(benchmark::kMillisecond);
  } else {
    BENCHMARK(Buddy_Opening2D_Constant_Padding)
        ->Arg(1)
        ->Unit(benchmark::kMillisecond);
  }
}

// Register benchmarking function.
void registerBenchmarkBuddyClosing2D() {
  if (BoundaryType1 == replicate_padding) {
    BENCHMARK(Buddy_Closing2D_Replicate_Padding)
        ->Arg(1)
        ->Unit(benchmark::kMillisecond);
  } else {
    BENCHMARK(Buddy_Closing2D_Constant_Padding)
        ->Arg(1)
        ->Unit(benchmark::kMillisecond);
  }
}

// Register benchmarking function.
void registerBenchmarkBuddyTopHat2D() {
  if (BoundaryType1 == replicate_padding) {
    BENCHMARK(Buddy_TopHat2D_Replicate_Padding)
        ->Arg(1)
        ->Unit(benchmark::kMillisecond);
  } else {
    BENCHMARK(Buddy_TopHat2D_Constant_Padding)
        ->Arg(1)
        ->Unit(benchmark::kMillisecond);
  }
}

// Register benchmarking function.
void registerBenchmarkBuddyBottomHat2D() {
  if (BoundaryType1 == replicate_padding) {
    BENCHMARK(Buddy_BottomHat2D_Replicate_Padding)
        ->Arg(1)
        ->Unit(benchmark::kMillisecond);
  } else {
    BENCHMARK(Buddy_BottomHat2D_Constant_Padding)
        ->Arg(1)
        ->Unit(benchmark::kMillisecond);
  }
}

// Register benchmarking function.
void registerBenchmarkBuddyMorphGrad2D() {
  if (BoundaryType1 == replicate_padding) {
    BENCHMARK(Buddy_MorphGrad2D_Replicate_Padding)
        ->Arg(1)
        ->Unit(benchmark::kMillisecond);
  } else {
    BENCHMARK(Buddy_MorphGrad2D_Constant_Padding)
        ->Arg(1)
        ->Unit(benchmark::kMillisecond);
  }
}

// Generate result image.
void generateResultBuddyErosion2D(char **argv, Img<float, 2> input) {
  // Define the MemRef descriptor for kernel and output.
  MemRef<float, 2> kernel = kernelBuddyMorph2D;
  MemRef<float, 2> output(sizesOutputBuddyMorph2D);
  MemRef<float, 2> copyMemRef(sizesOutputBuddyMorph2D, 256.f);
  // Run the 2D Erosionelation.
  if (static_cast<string>(argv[3]) == "REPLICATE_PADDING") {
    dip::detail::_mlir_ciface_erosion_2d_replicate_padding(
        input, &kernel, &output, &copyMemRef, 1 /* Center X */,
        1 /* Center Y */, 1, 0.0f /* Constant Value */);
  } else {
    dip::detail::_mlir_ciface_erosion_2d_constant_padding(
        input, &kernel, &output, &copyMemRef, 1 /* Center X */,
        1 /* Center Y */, 1, 0.0f /* Constant Value */);
  }

  // Define an Img container for the output image.
  intptr_t outputSizes[2] = {outputRowsBuddyMorph2D, outputColsBuddyMorph2D};
  Img<float, 2> outputImage(output.getData(), outputSizes);

  // Write output to PNG.
  bool result = dip::imwrite("ResultBuddyErosion2D.png", outputImage);

  if (!result) {
    fprintf(stderr, "Exception converting image to PNG format. \n");
  }
  if (result)
    cout << "Saved PNG file." << endl;
  else
    cout << "ERROR: Can't save PNG file." << endl;
}

// Generate result image.
void generateResultBuddyDilation2D(char **argv, Img<float, 2> input) {
  // Define the MemRef descriptor kernel and output.
  MemRef<float, 2> kernel = kernelBuddyMorph2D;
  MemRef<float, 2> output(sizesOutputBuddyMorph2D);
  MemRef<float, 2> copymemref(sizesOutputBuddyMorph2D, -1.f);
  // Run the 2D Dilationelation.
  if (static_cast<string>(argv[3]) == "REPLICATE_PADDING") {
    dip::detail::_mlir_ciface_dilation_2d_replicate_padding(
        input, &kernel, &output, &copymemref, 1 /* Center X */,
        1 /* Center Y */, 1, 0.0f /* Constant Value */);
  } else {
    dip::detail::_mlir_ciface_dilation_2d_constant_padding(
        input, &kernel, &output, &copymemref, 1 /* Center X */,
        1 /* Center Y */, 1, 0.0f /* Constant Value */);
  }

  // Define an Img container for the output image.
  intptr_t outputSizes[2] = {outputRowsBuddyMorph2D, outputColsBuddyMorph2D};
  Img<float, 2> outputImage(output.getData(), outputSizes);

  // Write output to PNG.
  bool result = dip::imwrite("ResultBuddyDilation2D.png", outputImage);

  if (!result) {
    fprintf(stderr, "Exception converting image to PNG format. \n");
  }
  if (result)
    cout << "Saved PNG file." << endl;
  else
    cout << "ERROR: Can't save PNG file." << endl;
}

// Generate result image.
void generateResultBuddyOpening2D(char **argv, Img<float, 2> input) {
  // Define the MemRef descriptor for kernel and output.
  MemRef<float, 2> kernel = kernelBuddyMorph2D;
  MemRef<float, 2> output(sizesOutputBuddyMorph2D);
  MemRef<float, 2> output1(sizesOutputBuddyMorph2D);
  MemRef<float, 2> copymemref(sizesOutputBuddyMorph2D, -1.f);
  MemRef<float, 2> copymemref1(sizesOutputBuddyMorph2D, 256.f);
  // Run the 2D Opening operation
  if (static_cast<string>(argv[3]) == "REPLICATE_PADDING") {
    dip::detail::_mlir_ciface_opening_2d_replicate_padding(
        input, &kernel, &output, &output1, &copymemref, &copymemref1,
        1 /* Center X */, 1 /* Center Y */, 1, 0.0f /* Constant Value */);
  } else {
    dip::detail::_mlir_ciface_opening_2d_constant_padding(
        input, &kernel, &output, &output1, &copymemref, &copymemref1,
        1 /* Center X */, 1 /* Center Y */, 1, 0.0f /* Constant Value */);
  }

  // Define an Img container for the output image.
  intptr_t outputSizes[2] = {outputRowsBuddyMorph2D, outputColsBuddyMorph2D};
  Img<float, 2> outputImage(output.getData(), outputSizes);

  // Write output to PNG.
  bool result = dip::imwrite("ResultBuddyOpening2D.png", outputImage);

  if (!result) {
    fprintf(stderr, "Exception converting image to PNG format. \n");
  }
  if (result)
    cout << "Saved PNG file." << endl;
  else
    cout << "ERROR: Can't save PNG file." << endl;
}

// Generate result image.
void generateResultBuddyClosing2D(char **argv, Img<float, 2> input) {
  // Define the MemRef descriptor for kernel and output.
  MemRef<float, 2> kernel = kernelBuddyMorph2D;
  MemRef<float, 2> output(sizesOutputBuddyMorph2D);
  MemRef<float, 2> output1(sizesOutputBuddyMorph2D);
  MemRef<float, 2> copymemref(sizesOutputBuddyMorph2D, -1.f);
  MemRef<float, 2> copymemref1(sizesOutputBuddyMorph2D, 256.f);
  // Run the 2D Closing operation
  if (static_cast<string>(argv[3]) == "REPLICATE_PADDING") {
    dip::detail::_mlir_ciface_closing_2d_replicate_padding(
        input, &kernel, &output, &output1, &copymemref, &copymemref1,
        1 /* Center X */, 1 /* Center Y */, 1, 0.0f /* Constant Value */);
  } else {
    dip::detail::_mlir_ciface_closing_2d_constant_padding(
        input, &kernel, &output, &output1, &copymemref, &copymemref1,
        1 /* Center X */, 1 /* Center Y */, 1, 0.0f /* Constant Value */);
  }

  // Define an Img container for the output image.
  intptr_t outputSizes[2] = {outputRowsBuddyMorph2D, outputColsBuddyMorph2D};
  Img<float, 2> outputImage(output.getData(), outputSizes);

  // Write output to PNG.
  bool result = dip::imwrite("ResultBuddyClosing2D.png", outputImage);

  if (!result) {
    fprintf(stderr, "Exception converting image to PNG format. \n");
  }
  if (result)
    cout << "Saved PNG file." << endl;
  else
    cout << "ERROR: Can't save PNG file." << endl;
}

// Generate result image.
void generateResultBuddyTopHat2D(char **argv, Img<float, 2> input) {
  // Define the MemRef descriptor for kernel and output.
  MemRef<float, 2> kernel = kernelBuddyMorph2D;
  MemRef<float, 2> output(sizesOutputBuddyMorph2D);
  MemRef<float, 2> output1(sizesOutputBuddyMorph2D);
  MemRef<float, 2> output2(sizesOutputBuddyMorph2D);
  MemRef<float, 2> input1(sizesOutputBuddyMorph2D);
  MemRef<float, 2> copymemref(sizesOutputBuddyMorph2D, 256.f);
  MemRef<float, 2> copymemref1(sizesOutputBuddyMorph2D, -1.f);
  // Run the 2D TopHat operation
  if (static_cast<string>(argv[3]) == "REPLICATE_PADDING") {
    dip::detail::_mlir_ciface_tophat_2d_replicate_padding(
        input, &kernel, &output, &output1, &output2, &input1, &copymemref,
        &copymemref1, 1 /* Center X */, 1 /* Center Y */, 1,
        0.0f /* Constant Value */);
  } else {
    dip::detail::_mlir_ciface_tophat_2d_constant_padding(
        input, &kernel, &output, &output1, &output2, &input1, &copymemref,
        &copymemref1, 1 /* Center X */, 1 /* Center Y */, 1,
        0.0f /* Constant Value */);
  }

  // Define an Img container for the output image.
  intptr_t outputSizes[2] = {outputRowsBuddyMorph2D, outputColsBuddyMorph2D};
  Img<float, 2> outputImage(output.getData(), outputSizes);

  // Write output to PNG.
  bool result = dip::imwrite("ResultBuddyTopHat2D.png", outputImage);

  if (!result) {
    fprintf(stderr, "Exception converting image to PNG format. \n");
  }
  if (result)
    cout << "Saved PNG file." << endl;
  else
    cout << "ERROR: Can't save PNG file." << endl;
}

// Generate result image.
void generateResultBuddyBottomHat2D(char **argv, Img<float, 2> input) {
  // Define the MemRef descriptor for kernel and output.
  MemRef<float, 2> kernel = kernelBuddyMorph2D;
  MemRef<float, 2> output(sizesOutputBuddyMorph2D);
  MemRef<float, 2> output1(sizesOutputBuddyMorph2D);
  MemRef<float, 2> output2(sizesOutputBuddyMorph2D);
  MemRef<float, 2> input1(sizesOutputBuddyMorph2D);
  MemRef<float, 2> copymemref(sizesOutputBuddyMorph2D, 256.f);
  MemRef<float, 2> copymemref1(sizesOutputBuddyMorph2D, -1.f);
  // Run the 2D BottomHat operation
  if (static_cast<string>(argv[3]) == "REPLICATE_PADDING") {
    dip::detail::_mlir_ciface_bottomhat_2d_replicate_padding(
        input, &kernel, &output, &output1, &output2, &input1, &copymemref,
        &copymemref1, 1 /* Center X */, 1 /* Center Y */, 1,
        0.0f /* Constant Value */);
  } else {
    dip::detail::_mlir_ciface_bottomhat_2d_constant_padding(
        input, &kernel, &output, &output1, &output2, &input1, &copymemref,
        &copymemref1, 1 /* Center X */, 1 /* Center Y */, 1,
        0.0f /* Constant Value */);
  }

  // Define an Img container for the output image.
  intptr_t outputSizes[2] = {outputRowsBuddyMorph2D, outputColsBuddyMorph2D};
  Img<float, 2> outputImage(output.getData(), outputSizes);

  // Write output to PNG.
  bool result = dip::imwrite("ResultBuddyBottomHat2D.png", outputImage);

  if (!result) {
    fprintf(stderr, "Exception converting image to PNG format. \n");
  }
  if (result)
    cout << "Saved PNG file." << endl;
  else
    cout << "ERROR: Can't save PNG file." << endl;
}

// Generate result image.
void generateResultBuddyMorphGrad2D(char **argv, Img<float, 2> input) {
  // Define the MemRef descriptor for kernel and output.
  MemRef<float, 2> kernel = kernelBuddyMorph2D;
  MemRef<float, 2> output(sizesOutputBuddyMorph2D);
  MemRef<float, 2> output1(sizesOutputBuddyMorph2D);
  MemRef<float, 2> output2(sizesOutputBuddyMorph2D);
  MemRef<float, 2> input1(sizesOutputBuddyMorph2D);
  MemRef<float, 2> copymemref(sizesOutputBuddyMorph2D, 256.f);
  MemRef<float, 2> copymemref1(sizesOutputBuddyMorph2D, -1.f);
  // Run the 2D MorphGrad operation
  if (static_cast<string>(argv[3]) == "REPLICATE_PADDING") {
    dip::detail::_mlir_ciface_morphgrad_2d_replicate_padding(
        input, &kernel, &output, &output1, &output2, &input1, &copymemref,
        &copymemref1, 1 /* Center X */, 1 /* Center Y */, 1,
        0.0f /* Constant Value */);
  } else {
    dip::detail::_mlir_ciface_morphgrad_2d_constant_padding(
        input, &kernel, &output, &output1, &output2, &input1, &copymemref,
        &copymemref1, 1 /* Center X */, 1 /* Center Y */, 1,
        0.0f /* Constant Value */);
  }

  // Define an Img container for the output image.
  intptr_t outputSizes[2] = {outputRowsBuddyMorph2D, outputColsBuddyMorph2D};
  Img<float, 2> outputImage(output.getData(), outputSizes);

  // Write output to PNG.
  bool result = dip::imwrite("ResultBuddyMorphGrad2D.png", outputImage);

  if (!result) {
    fprintf(stderr, "Exception converting image to PNG format. \n");
  }
  if (result)
    cout << "Saved PNG file." << endl;
  else
    cout << "ERROR: Can't save PNG file." << endl;
}
