//===- BuddySepCorr2DBenchmark.cpp -------------------------------------------===//
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
// This file implements the benchmark for SepCorr2D operation.
//
//===----------------------------------------------------------------------===//

#include "Kernels.h"
#include <benchmark/benchmark.h>
#include <buddy/Core/Container.h>
#include <buddy/DIP/ImageContainer.h>
#include <buddy/DIP/DIP.h>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

Mat inputImageBuddySepCorr2D;

float *kernelDataBuddySepCorr2DX;
float *kernelDataBuddySepCorr2DY;

int kernelRowsBuddySepCorr2DX, kernelColsBuddySepCorr2DX, kernelRowsBuddySepCorr2DY, kernelColsBuddySepCorr2DY;

// Define the output size.
int outputRowsBuddySepCorr2D, outputColsBuddySepCorr2D;

// Define sizes of input, kernel, and output.
intptr_t sizesInputBuddySepCorr2D[2];
intptr_t sizesKernelBuddySepCorr2DX[2];
intptr_t sizesKernelBuddySepCorr2DY[2];
intptr_t sizesOutputBuddySepCorr2D[2];

// Declare Boundary Options supported.
enum BoundaryOption { constant_padding, replicate_padding };

// Define Boundary option selected.
BoundaryOption BoundaryType2;

void initializeBuddySepCorr2D(char **argv) {
  inputImageBuddySepCorr2D = imread(argv[1], IMREAD_GRAYSCALE);

  kernelDataBuddySepCorr2DX = get<0>(kernelMap[argv[5]]);
  kernelRowsBuddySepCorr2DX = get<1>(kernelMap[argv[5]]);
  kernelColsBuddySepCorr2DX = get<2>(kernelMap[argv[5]]);

  kernelDataBuddySepCorr2DY = get<0>(kernelMap[argv[6]]);
  kernelRowsBuddySepCorr2DY = get<1>(kernelMap[argv[6]]);
  kernelColsBuddySepCorr2DY = get<2>(kernelMap[argv[6]]);

  outputRowsBuddySepCorr2D = inputImageBuddySepCorr2D.rows;
  outputColsBuddySepCorr2D = inputImageBuddySepCorr2D.cols;

  sizesInputBuddySepCorr2D[0] = inputImageBuddySepCorr2D.rows;
  sizesInputBuddySepCorr2D[1] = inputImageBuddySepCorr2D.cols;

  sizesKernelBuddySepCorr2DX[0] = kernelRowsBuddySepCorr2DX;
  sizesKernelBuddySepCorr2DX[1] = kernelColsBuddySepCorr2DX;

  sizesKernelBuddySepCorr2DY[0] = kernelRowsBuddySepCorr2DY;
  sizesKernelBuddySepCorr2DY[1] = kernelColsBuddySepCorr2DY;  

  sizesOutputBuddySepCorr2D[0] = outputRowsBuddySepCorr2D;
  sizesOutputBuddySepCorr2D[1] = outputColsBuddySepCorr2D;

  if (static_cast<string>(argv[3]) == "REPLICATE_PADDING") {
    BoundaryType2 = replicate_padding;
  } else {
    BoundaryType2 = constant_padding;
  }
}

static void Buddy_SepCorr2D_Constant_Padding(benchmark::State &state) {
  // Define the MemRef descriptor for input, kernel, and output.
  Img<float, 2> inputBuddySepCorr2D(inputImageBuddySepCorr2D);
  MemRef<float, 2> kernelBuddySepCorr2DX(kernelDataBuddySepCorr2DX,
                                     sizesKernelBuddySepCorr2DX);
  MemRef<float, 2> kernelBuddySepCorr2DY(kernelDataBuddySepCorr2DY,
                                     sizesKernelBuddySepCorr2DY);                                   
  MemRef<float, 2> outputBuddySepCorr2D(sizesOutputBuddySepCorr2D);

  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      // Call the MLIR Corr2D function.
      dip::Sep_Corr2D(&inputBuddySepCorr2D, &kernelBuddySepCorr2DX, &kernelBuddySepCorr2DY, &outputBuddySepCorr2D,
                  0 /* Center X */, 0 /* Center Y */,
                  dip::BOUNDARY_OPTION::CONSTANT_PADDING,
                  0.0f /* Constant Value*/);
    }
  }
}

static void Buddy_SepCorr2D_Replicate_Padding(benchmark::State &state) {
  // Define the MemRef descriptor for input, kernel, and output.
  Img<float, 2> inputBuddySepCorr2D(inputImageBuddySepCorr2D);
  MemRef<float, 2> kernelBuddySepCorr2DX(kernelDataBuddySepCorr2DX,
                                     sizesKernelBuddySepCorr2DX);
  MemRef<float, 2> kernelBuddySepCorr2DY(kernelDataBuddySepCorr2DY,
                                     sizesKernelBuddySepCorr2DY);                                   
  MemRef<float, 2> outputBuddySepCorr2D(sizesOutputBuddySepCorr2D);

  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      // Call the MLIR Sep_Corr2D function.
      dip::Sep_Corr2D(&inputBuddySepCorr2D, &kernelBuddySepCorr2DX, &kernelBuddySepCorr2DY, &outputBuddySepCorr2D,
                  0 /* Center X */, 0 /* Center Y */,
                  dip::BOUNDARY_OPTION::REPLICATE_PADDING,
                  0.0f /* Constant Value*/);
    }
  }
}

// Register benchmarking function.
void registerBenchmarkBuddySepCorr2D() {
  if (BoundaryType2 == replicate_padding) {
    BENCHMARK(Buddy_SepCorr2D_Replicate_Padding)
        ->Arg(1)
        ->Unit(benchmark::kMillisecond);
  } else {
    BENCHMARK(Buddy_SepCorr2D_Constant_Padding)
        ->Arg(1)
        ->Unit(benchmark::kMillisecond);
  }
}

// Generate result image.
void generateResultBuddySepCorr2D(char **argv) {
  // Define the MemRef descriptor for input, kernel, and output.
  Img<float, 2> input(inputImageBuddySepCorr2D);
  MemRef<float, 2> kernelX(kernelDataBuddySepCorr2DX, sizesKernelBuddySepCorr2DX);
  MemRef<float, 2> kernelY(kernelDataBuddySepCorr2DY, sizesKernelBuddySepCorr2DY);  
  MemRef<float, 2> output(sizesOutputBuddySepCorr2D);
  // Run the 2D correlation.
  if (static_cast<string>(argv[3]) == "REPLICATE_PADDING") {
    // Call the MLIR Corr2D function.
    dip::Sep_Corr2D(&input, &kernelX, &kernelY, &output, 0 /* Center X */, 0/* Center Y */,
                dip::BOUNDARY_OPTION::REPLICATE_PADDING,
                0.0f /* Constant Value*/);
  } else {
    // Call the MLIR Corr2D function.
    dip::Sep_Corr2D(&input, &kernelX, &kernelY, &output, 0 /* Center X */, 0 /* Center Y */,
                dip::BOUNDARY_OPTION::CONSTANT_PADDING,
                0.0f /* Constant Value*/);
  }

Mat outputImage(outputRowsBuddySepCorr2D, outputColsBuddySepCorr2D, CV_32FC1,
                  output.getData());

  // Choose a PNG compression level
  vector<int> compressionParams;
  compressionParams.push_back(IMWRITE_PNG_COMPRESSION);
  compressionParams.push_back(9);

  // Write output to PNG.
  bool result = false;
  try {
    result = imwrite("ResultBuddySepCorr2D.png", outputImage, compressionParams);
  } catch (const cv::Exception &ex) {
    fprintf(stderr, "Exception converting image to PNG format: %s\n",
            ex.what());
  }
  if (result)
    cout << "Saved PNG file." << endl;
  else
    cout << "ERROR: Can't save PNG file." << endl;
}                  
