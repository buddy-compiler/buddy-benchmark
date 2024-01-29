//===- EigenConvolve2DBenchmark.cpp ---------------------------------------===//
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
// This file implements the benchmark for Eigen convolve.
//
//===----------------------------------------------------------------------===//

#include "Kernels.h"
#include <Eigen/CXX11/Tensor>
#include <Eigen/Dense>
#include <benchmark/benchmark.h>
#include <buddy/Core/Container.h>
#include <buddy/DIP/ImageContainer.h>
#include <buddy/DIP/imgcodecs/loadsave.h>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

using namespace std;

// Declare the input name.
std::string inputNameEigenConvolve2D;
float *kernelDataEigenConvolve2D;

// Declare the input, kernel and output Eigen Tensor.
Eigen::Tensor<float, 2> input;
Eigen::Tensor<float, 2> kernel;
Eigen::Tensor<float, 2> output;
// Set the dimensions that need to perform the convolution.
Eigen::array<int, 2> dims = {0, 1};

// Declare rows and cols of the input, kernel and output.
int inputRowsEigenConvolve2D, inputColsEigenConvolve2D;
int kernelRowsEigenConvolve2D, kernelColsEigenConvolve2D;
int outputRowsEigenConvolve2D, outputColsEigenConvolve2D;

// Initialize tensors for the convolution.
// - Read image and kernel as OpenCV Mat.
// - Get the sizes of the input, kernel and output.
// - Define the input, kernel and output MemRef container.
// - Initialize the input, kernel and output Eigen tensor.
void initializeEigenConvolve2D(char **argv, Img<float, 2> inputImage) {
  // Get the sizes of the input, kernel and output.
  inputRowsEigenConvolve2D = inputImage.getSizes()[0];
  inputColsEigenConvolve2D = inputImage.getSizes()[1];
  kernelDataEigenConvolve2D = get<0>(kernelMap[argv[2]]);
  kernelRowsEigenConvolve2D = get<1>(kernelMap[argv[2]]);
  kernelColsEigenConvolve2D = get<2>(kernelMap[argv[2]]);
  outputRowsEigenConvolve2D =
      inputRowsEigenConvolve2D - kernelRowsEigenConvolve2D + 1;
  outputColsEigenConvolve2D =
      inputColsEigenConvolve2D - kernelColsEigenConvolve2D + 1;

  intptr_t sizesInputEigenConvolve2D[2] = {inputRowsEigenConvolve2D,
                                           inputColsEigenConvolve2D};
  intptr_t sizesKernelEigenConvolve2D[2] = {kernelRowsEigenConvolve2D,
                                            kernelColsEigenConvolve2D};
  intptr_t sizesOutputEigenConvolve2D[2] = {outputRowsEigenConvolve2D,
                                            outputColsEigenConvolve2D};

  // Define the kernel and output MemRef container.
  MemRef<float, 2> kernelMemRef(kernelDataEigenConvolve2D,
                                sizesKernelEigenConvolve2D);
  MemRef<float, 2> outputMemRef(sizesOutputEigenConvolve2D);

  // Initialize the input, kernel and output Eigen tensor.
  Eigen::TensorMap inputTensorMap = Eigen::TensorMap<Eigen::Tensor<float, 2>>(
      inputImage.getData(), inputRowsEigenConvolve2D, inputColsEigenConvolve2D);
  Eigen::TensorMap kernelTensorMap = Eigen::TensorMap<Eigen::Tensor<float, 2>>(
      kernelMemRef.getData(), kernelRowsEigenConvolve2D,
      kernelColsEigenConvolve2D);
  Eigen::TensorMap outputTensorMap = Eigen::TensorMap<Eigen::Tensor<float, 2>>(
      outputMemRef.getData(), outputRowsEigenConvolve2D,
      outputColsEigenConvolve2D);
  input = Eigen::Tensor<float, 2>(inputTensorMap);
  kernel = Eigen::Tensor<float, 2>(kernelTensorMap);
  output = Eigen::Tensor<float, 2>(outputTensorMap);
}

// Benchmarking function.
static void Eigen_Convolve2D(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      output = input.convolve(kernel, dims);
    }
  }
}

// Register benchmarking function.
BENCHMARK(Eigen_Convolve2D)->Arg(1)->Unit(benchmark::kMillisecond);

// Generate result image.
void generateResultEigenConvolve2D() {
  // Define an Img container for the output image.
  intptr_t outputSizes[2] = {outputRowsEigenConvolve2D,
                             outputColsEigenConvolve2D};
  Img<float, 2> outputImage(output.data(), outputSizes);

  // Write output to PNG.
  bool result = dip::imwrite("ResultEigenConvolve2D.png", outputImage);

  if (!result) {
    fprintf(stderr, "Exception converting image to PNG format. \n");
  }
  if (result)
    cout << "Saved PNG file." << endl;
  else
    cout << "ERROR: Can't save PNG file." << endl;
}
