//===- BuddyBottomHat2DBenchmark.cpp -------------------------------------------===//
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
// This file implements the benchmark for BottomHat2D operation.
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
void _mlir_ciface_bottomhat_2d_constant_padding(MemRef<float, 2> *inputBuddyBottomHat2D,
                                           MemRef<float, 2> *kernelBuddyBottomHat2D,
                                           MemRef<float, 2> *outputBuddyBottomHat2D,
                                           MemRef<float, 2> *outputBuddyBottomHat2D1,
                                           MemRef<float, 2> *outputBuddyBottomHat2D2,
                                           MemRef<float, 2> *inputBuddyBottomHat2D1,
                                           MemRef<float, 2> *copyMemRefBottomHat2D,
                                           MemRef<float, 2>* copyMemRefBottomHat2D1,
                                           unsigned int centerX,
                                           unsigned int centerY,
                                           unsigned int iterations,
                                           float constantValue
    );

void _mlir_ciface_bottomhat_2d_replicate_padding(MemRef<float, 2> *inputBuddyBottomHat2D,
                                           MemRef<float, 2> *kernelBuddyBottomHat2D,
                                           MemRef<float, 2> *outputBuddyBottomHat2D,
                                           MemRef<float, 2> *outputBuddyBottomHat2D1,
                                           MemRef<float, 2> *outputBuddyBottomHat2D2,
                                           MemRef<float, 2> *inputBuddyBottomHat2D1,
                                           MemRef<float, 2> *copyMemRefBottomHat2D,
                                           MemRef<float, 2>* copyMemRefBottomHat2D1,
                                           unsigned int centerX,
                                           unsigned int centerY,
                                           unsigned int iterations,
                                           float constantValue
    );
}

// Declare input image and kernel.
Mat inputImageBuddyBottomHat2D, kernelBuddyBottomHat2DMat;

// Define the kernel size.
int kernelRowsBuddyBottomHat2D, kernelColsBuddyBottomHat2D;

// Define the output size.
int outputRowsBuddyBottomHat2D, outputColsBuddyBottomHat2D;

// Define sizes of input, kernel, and output.
intptr_t sizesInputBuddyBottomHat2D[2];
intptr_t sizesKernelBuddyBottomHat2D[2];
intptr_t sizesOutputBuddyBottomHat2D[2];

// Declare Boundary Options supported.
enum BoundaryOption { constant_padding, replicate_padding };

// Define Boundary option selected.
BoundaryOption BoundaryType6;

void initializeBottomHat2D(char **argv) {
  inputImageBuddyBottomHat2D = imread(argv[1], IMREAD_GRAYSCALE);
  kernelBuddyBottomHat2DMat =
      Mat(get<1>(kernelMap[argv[2]]), get<2>(kernelMap[argv[2]]), CV_32FC1,
          get<0>(kernelMap[argv[2]]));

  kernelRowsBuddyBottomHat2D = kernelBuddyBottomHat2DMat.rows;
  kernelColsBuddyBottomHat2D = kernelBuddyBottomHat2DMat.cols;

  outputRowsBuddyBottomHat2D = inputImageBuddyBottomHat2D.rows;
  outputColsBuddyBottomHat2D = inputImageBuddyBottomHat2D.cols;

  sizesInputBuddyBottomHat2D[0] = inputImageBuddyBottomHat2D.rows;
  sizesInputBuddyBottomHat2D[1] = inputImageBuddyBottomHat2D.cols;

  sizesKernelBuddyBottomHat2D[0] = kernelRowsBuddyBottomHat2D;
  sizesKernelBuddyBottomHat2D[1] = kernelColsBuddyBottomHat2D;

  sizesOutputBuddyBottomHat2D[0] = outputRowsBuddyBottomHat2D;
  sizesOutputBuddyBottomHat2D[1] = outputColsBuddyBottomHat2D;

  if (static_cast<string>(argv[3]) == "REPLICATE_PADDING") {
    BoundaryType6 = replicate_padding;
  } else {
    BoundaryType6 = constant_padding;
  }
}

static void Buddy_BottomHat2D_Constant_Padding(benchmark::State &state) {
  // Define the MemRef descriptor for input, kernel, and output.
  MemRef<float, 2> inputBuddyBottomHat2D(inputImageBuddyBottomHat2D,
                                    sizesInputBuddyBottomHat2D);
    MemRef<float, 2> inputBuddyBottomHat2D1(inputImageBuddyBottomHat2D,
                                    sizesInputBuddyBottomHat2D);                                  
  MemRef<float, 2> kernelBuddyBottomHat2D(kernelBuddyBottomHat2DMat,
                                     sizesKernelBuddyBottomHat2D);
  MemRef<float, 2> outputBuddyBottomHat2D(sizesOutputBuddyBottomHat2D);
  MemRef<float, 2> outputBuddyBottomHat2D1(sizesOutputBuddyBottomHat2D);
  MemRef<float, 2> outputBuddyBottomHat2D2(sizesOutputBuddyBottomHat2D);  
  MemRef<float, 2> copyMemRefBottomHat2D(sizesOutputBuddyBottomHat2D, 256.f);
  MemRef<float, 2> copyMemRefBottomHat2D1(sizesOutputBuddyBottomHat2D, -1.f);

  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      _mlir_ciface_bottomhat_2d_constant_padding(
          &inputBuddyBottomHat2D, &kernelBuddyBottomHat2D, &outputBuddyBottomHat2D, &outputBuddyBottomHat2D1, &outputBuddyBottomHat2D2, &inputBuddyBottomHat2D1, &copyMemRefBottomHat2D, &copyMemRefBottomHat2D1,
          1 /* Center X */, 1 /* Center Y */,5, 0.0f /* Constant Value */);
    }
  }
}

static void Buddy_BottomHat2D_Replicate_Padding(benchmark::State &state) {
  // Define the MemRef descriptor for input, kernel, and output.
  MemRef<float, 2> inputBuddyBottomHat2D(inputImageBuddyBottomHat2D,
                                    sizesInputBuddyBottomHat2D);
    MemRef<float, 2> inputBuddyBottomHat2D1(inputImageBuddyBottomHat2D,
                                    sizesInputBuddyBottomHat2D);                                  
  MemRef<float, 2> kernelBuddyBottomHat2D(kernelBuddyBottomHat2DMat,
                                     sizesKernelBuddyBottomHat2D);
  MemRef<float, 2> outputBuddyBottomHat2D(sizesOutputBuddyBottomHat2D);
  MemRef<float, 2> outputBuddyBottomHat2D1(sizesOutputBuddyBottomHat2D);
  MemRef<float, 2> outputBuddyBottomHat2D2(sizesOutputBuddyBottomHat2D);  
  MemRef<float, 2> copyMemRefBottomHat2D(sizesOutputBuddyBottomHat2D, 256.f);
  MemRef<float, 2> copyMemRefBottomHat2D1(sizesOutputBuddyBottomHat2D, -1.f);

  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      _mlir_ciface_bottomhat_2d_replicate_padding(
          &inputBuddyBottomHat2D, &kernelBuddyBottomHat2D, &outputBuddyBottomHat2D, &outputBuddyBottomHat2D1, &outputBuddyBottomHat2D2, &inputBuddyBottomHat2D1, &copyMemRefBottomHat2D, &copyMemRefBottomHat2D1,
          1 /* Center X */, 1 /* Center Y */,5, 0.0f /* Constant Value */);
    }
  }
}

// Register benchmarking function.
void registerBenchmarkBuddyBottomHat2D() {
  if (BoundaryType6 == replicate_padding) {
    BENCHMARK(Buddy_BottomHat2D_Replicate_Padding)->Arg(1);
  } else {
    BENCHMARK(Buddy_BottomHat2D_Constant_Padding)->Arg(1);
  }
}

// Generate result image.
void generateResultBuddyBottomHat2D(char **argv) {
  // Define the MemRef descriptor for input, kernel, and output.
  MemRef<float, 2> input(inputImageBuddyBottomHat2D, sizesInputBuddyBottomHat2D);
  MemRef<float, 2> kernel(get<0>(kernelMap[argv[2]]), sizesKernelBuddyBottomHat2D);
  MemRef<float, 2> output(sizesOutputBuddyBottomHat2D);
  MemRef<float, 2> output1(sizesOutputBuddyBottomHat2D);
  MemRef<float, 2> output2(sizesOutputBuddyBottomHat2D);
  MemRef<float, 2> input1(sizesOutputBuddyBottomHat2D);    
  MemRef<float, 2> copymemref(sizesOutputBuddyBottomHat2D, 256.f);
  MemRef<float, 2> copymemref1(sizesOutputBuddyBottomHat2D, -1.f);
  // Run the 2D BottomHat operation
  if (static_cast<string>(argv[3]) == "REPLICATE_PADDING") {
    _mlir_ciface_bottomhat_2d_replicate_padding(&input, &kernel, &output, &output1, &output2, &input1, &copymemref, &copymemref1,
                                           1 /* Center X */, 1 /* Center Y */, 5,
                                           0.0f /* Constant Value */);
  } else {
    _mlir_ciface_bottomhat_2d_constant_padding(&input, &kernel, &output, &output1, &output2, &input1, &copymemref, &copymemref1,
                                          1 /* Center X */, 1 /* Center Y */, 5,
                                          0.0f /* Constant Value */);
  }

  // Define a cv::Mat with the output of the BottomHat
  Mat outputImage(outputRowsBuddyBottomHat2D, outputColsBuddyBottomHat2D, CV_32FC1,
                  output.getData());

  // Choose a PNG compression level
  vector<int> compressionParams;
  compressionParams.push_back(IMWRITE_PNG_COMPRESSION);
  compressionParams.push_back(9);

  // Write output to PNG.
  bool result = false;
  try {
    result = imwrite("ResultBuddyBottomHat2D.png", outputImage, compressionParams);
  } catch (const cv::Exception &ex) {
    fprintf(stderr, "Exception converting image to PNG format: %s\n",
            ex.what());
  }
  if (result)
    cout << "Saved PNG file." << endl;
  else
    cout << "ERROR: Can't save PNG file." << endl;
}
