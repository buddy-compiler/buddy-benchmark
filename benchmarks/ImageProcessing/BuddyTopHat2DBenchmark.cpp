//===- BuddyTopHat2DBenchmark.cpp -------------------------------------------===//
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
// This file implements the benchmark for TopHat2D operation.
//
//===----------------------------------------------------------------------===//

#include "ImageProcessing/Kernels.h"
#include "Utils/Container.h"
#include <benchmark/benchmark.h>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

// Declare the conv2d C interface.
extern "C" {
void _mlir_ciface_tophat_2d_constant_padding(MemRef<float, 2> *inputBuddyTopHat2D,
                                           MemRef<float, 2> *kernelBuddyTopHat2D,
                                           MemRef<float, 2> *outputBuddyTopHat2D,
                                           MemRef<float, 2> *outputBuddyTopHat2D1,
                                           MemRef<float, 2> *outputBuddyTopHat2D2,
                                           MemRef<float, 2> *inputBuddyTopHat2D1,
                                           MemRef<float, 2> *copyMemRefTopHat2D,
                                           MemRef<float, 2>* copyMemRefTopHat2D1,
                                           unsigned int centerX,
                                           unsigned int centerY,
                                           unsigned int iterations,
                                           float constantValue
    );

void _mlir_ciface_tophat_2d_replicate_padding(MemRef<float, 2> *inputBuddyTopHat2D,
                                           MemRef<float, 2> *kernelBuddyTopHat2D,
                                           MemRef<float, 2> *outputBuddyTopHat2D,
                                           MemRef<float, 2> *outputBuddyTopHat2D1,
                                           MemRef<float, 2> *outputBuddyTopHat2D2,
                                           MemRef<float, 2> *inputBuddyTopHat2D1,
                                           MemRef<float, 2> *copyMemRefTopHat2D,
                                           MemRef<float, 2>* copyMemRefTopHat2D1,
                                           unsigned int centerX,
                                           unsigned int centerY,
                                           unsigned int iterations,
                                           float constantValue
    );
}

// Declare input image and kernel.
Mat inputImageBuddyTopHat2D, kernelBuddyTopHat2DMat;

// Define the kernel size.
int kernelRowsBuddyTopHat2D, kernelColsBuddyTopHat2D;

// Define the output size.
int outputRowsBuddyTopHat2D, outputColsBuddyTopHat2D;

// Define sizes of input, kernel, and output.
intptr_t sizesInputBuddyTopHat2D[2];
intptr_t sizesKernelBuddyTopHat2D[2];
intptr_t sizesOutputBuddyTopHat2D[2];

// Declare Boundary Options supported.
enum BoundaryOption { constant_padding, replicate_padding };

// Define Boundary option selected.
BoundaryOption BoundaryType5;

void initializeTopHat2D(char **argv) {
  inputImageBuddyTopHat2D = imread(argv[1], IMREAD_GRAYSCALE);
  kernelBuddyTopHat2DMat =
      Mat(get<1>(kernelMap[argv[2]]), get<2>(kernelMap[argv[2]]), CV_32FC1,
          get<0>(kernelMap[argv[2]]));

  kernelRowsBuddyTopHat2D = kernelBuddyTopHat2DMat.rows;
  kernelColsBuddyTopHat2D = kernelBuddyTopHat2DMat.cols;

  outputRowsBuddyTopHat2D = inputImageBuddyTopHat2D.rows;
  outputColsBuddyTopHat2D = inputImageBuddyTopHat2D.cols;

  sizesInputBuddyTopHat2D[0] = inputImageBuddyTopHat2D.rows;
  sizesInputBuddyTopHat2D[1] = inputImageBuddyTopHat2D.cols;

  sizesKernelBuddyTopHat2D[0] = kernelRowsBuddyTopHat2D;
  sizesKernelBuddyTopHat2D[1] = kernelColsBuddyTopHat2D;

  sizesOutputBuddyTopHat2D[0] = outputRowsBuddyTopHat2D;
  sizesOutputBuddyTopHat2D[1] = outputColsBuddyTopHat2D;

  if (static_cast<string>(argv[3]) == "REPLICATE_PADDING") {
    BoundaryType5 = replicate_padding;
  } else {
    BoundaryType5 = constant_padding;
  }
}

static void Buddy_TopHat2D_Constant_Padding(benchmark::State &state) {
  // Define the MemRef descriptor for input, kernel, and output.
  MemRef<float, 2> inputBuddyTopHat2D(inputImageBuddyTopHat2D,
                                    sizesInputBuddyTopHat2D);
    MemRef<float, 2> inputBuddyTopHat2D1(inputImageBuddyTopHat2D,
                                    sizesInputBuddyTopHat2D);                                  
  MemRef<float, 2> kernelBuddyTopHat2D(kernelBuddyTopHat2DMat,
                                     sizesKernelBuddyTopHat2D);
  MemRef<float, 2> outputBuddyTopHat2D(sizesOutputBuddyTopHat2D);
  MemRef<float, 2> outputBuddyTopHat2D1(sizesOutputBuddyTopHat2D);
  MemRef<float, 2> outputBuddyTopHat2D2(sizesOutputBuddyTopHat2D);  
  MemRef<float, 2> copyMemRefTopHat2D(sizesOutputBuddyTopHat2D, 256.f);
  MemRef<float, 2> copyMemRefTopHat2D1(sizesOutputBuddyTopHat2D, -1.f);

  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      _mlir_ciface_tophat_2d_constant_padding(
          &inputBuddyTopHat2D, &kernelBuddyTopHat2D, &outputBuddyTopHat2D, &outputBuddyTopHat2D1, &outputBuddyTopHat2D2, &inputBuddyTopHat2D1, &copyMemRefTopHat2D, &copyMemRefTopHat2D1,
          1 /* Center X */, 1 /* Center Y */,5, 0.0f /* Constant Value */);
    }
  }
}

static void Buddy_TopHat2D_Replicate_Padding(benchmark::State &state) {
  // Define the MemRef descriptor for input, kernel, and output.
  MemRef<float, 2> inputBuddyTopHat2D(inputImageBuddyTopHat2D,
                                    sizesInputBuddyTopHat2D);
    MemRef<float, 2> inputBuddyTopHat2D1(inputImageBuddyTopHat2D,
                                    sizesInputBuddyTopHat2D);                                  
  MemRef<float, 2> kernelBuddyTopHat2D(kernelBuddyTopHat2DMat,
                                     sizesKernelBuddyTopHat2D);
  MemRef<float, 2> outputBuddyTopHat2D(sizesOutputBuddyTopHat2D);
  MemRef<float, 2> outputBuddyTopHat2D1(sizesOutputBuddyTopHat2D);
  MemRef<float, 2> outputBuddyTopHat2D2(sizesOutputBuddyTopHat2D);  
  MemRef<float, 2> copyMemRefTopHat2D(sizesOutputBuddyTopHat2D, 256.f);
  MemRef<float, 2> copyMemRefTopHat2D1(sizesOutputBuddyTopHat2D, -1.f);

  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      _mlir_ciface_tophat_2d_replicate_padding(
          &inputBuddyTopHat2D, &kernelBuddyTopHat2D, &outputBuddyTopHat2D, &outputBuddyTopHat2D1, &outputBuddyTopHat2D2, &inputBuddyTopHat2D1, &copyMemRefTopHat2D, &copyMemRefTopHat2D1,
          1 /* Center X */, 1 /* Center Y */,5, 0.0f /* Constant Value */);
    }
  }
}

// Register benchmarking function.
void registerBenchmarkBuddyTopHat2D() {
  if (BoundaryType5 == replicate_padding) {
    BENCHMARK(Buddy_TopHat2D_Replicate_Padding)->Arg(1);
  } else {
    BENCHMARK(Buddy_TopHat2D_Constant_Padding)->Arg(1);
  }
}

// Generate result image.
void generateResultBuddyTopHat2D(char **argv) {
  // Define the MemRef descriptor for input, kernel, and output.
  MemRef<float, 2> input(inputImageBuddyTopHat2D, sizesInputBuddyTopHat2D);
  MemRef<float, 2> kernel(get<0>(kernelMap[argv[2]]), sizesKernelBuddyTopHat2D);
  MemRef<float, 2> output(sizesOutputBuddyTopHat2D);
  MemRef<float, 2> output1(sizesOutputBuddyTopHat2D);
  MemRef<float, 2> output2(sizesOutputBuddyTopHat2D);
  MemRef<float, 2> input1(sizesOutputBuddyTopHat2D);    
  MemRef<float, 2> copymemref(sizesOutputBuddyTopHat2D, 256.f);
  MemRef<float, 2> copymemref1(sizesOutputBuddyTopHat2D, -1.f);
  // Run the 2D TopHat operation
  if (static_cast<string>(argv[3]) == "REPLICATE_PADDING") {
    _mlir_ciface_tophat_2d_replicate_padding(&input, &kernel, &output, &output1, &output2, &input1, &copymemref, &copymemref1,
                                           1 /* Center X */, 1 /* Center Y */, 5,
                                           0.0f /* Constant Value */);
  } else {
    _mlir_ciface_tophat_2d_constant_padding(&input, &kernel, &output, &output1, &output2, &input1, &copymemref, &copymemref1,
                                          1 /* Center X */, 1 /* Center Y */, 5,
                                          0.0f /* Constant Value */);
  }

  // Define a cv::Mat with the output of the TopHat
  Mat outputImage(outputRowsBuddyTopHat2D, outputColsBuddyTopHat2D, CV_32FC1,
                  output.getData());

  // Choose a PNG compression level
  vector<int> compressionParams;
  compressionParams.push_back(IMWRITE_PNG_COMPRESSION);
  compressionParams.push_back(9);

  // Write output to PNG.
  bool result = false;
  try {
    result = imwrite("ResultBuddyTopHat2D.png", outputImage, compressionParams);
  } catch (const cv::Exception &ex) {
    fprintf(stderr, "Exception converting image to PNG format: %s\n",
            ex.what());
  }
  if (result)
    cout << "Saved PNG file." << endl;
  else
    cout << "ERROR: Can't save PNG file." << endl;
}
