#include <benchmark/benchmark.h>
#include <opencv2/opencv.hpp>
#include "../../../kernels.h"
#include "../../include/ImageProcessing/BuddyContainer.hpp"

using namespace cv;
using namespace std;

// Read input image and specify kernel
Mat inputImage = imread("../../examples/conv-opt/images/YuTu.png", IMREAD_GRAYSCALE);
Mat kernel_opencv = Mat(3, 3, CV_32FC1, laplacianKernelAlign);
Mat output_opencv;

// Define the kernel.
float *kernelAlign = laplacianKernelAlign;
int kernelRows = laplacianKernelRows;
int kernelCols = laplacianKernelCols;

// Define output for buddy mlir implementation.
int outputRows = inputImage.rows - kernelRows + 1;
int outputCols = inputImage.cols - kernelCols + 1;
float *outputAlign = (float *)malloc(outputRows * outputCols * sizeof(float));

// Define allocated, sizes, and strides.
float *allocated = (float *)malloc(1 * sizeof(float));
intptr_t sizesInput[2] = {inputImage.rows, inputImage.cols};
intptr_t sizesKernel[2] = {kernelRows, kernelCols};
intptr_t sizesOutput[2] = {outputRows, outputCols};
intptr_t stridesInput[2] = {inputImage.rows, inputImage.cols};
intptr_t stridesKernel[2] = {kernelRows, kernelCols};
intptr_t stridesOutput[2] = {outputRows, outputCols};

float* fill_align(Mat image)
{
  int k = 0;
  float *inputAlign = (float *)malloc(image.rows * image.cols * sizeof(float));
  for (int i = 0; i < image.rows; i++) {
    for (int j = 0; j < image.cols; j++) {
      float pixelValue = (float)image.at<uchar>(i, j);
      inputAlign[k] = pixelValue;
      k++;
    }
  }
  return inputAlign;
}

float *inputAlign = fill_align(inputImage);

// Define memref descriptors.
MemRef_descriptor input =
    MemRef_Descriptor(allocated, inputAlign, 0, sizesInput, stridesInput);
MemRef_descriptor kernel =
    MemRef_Descriptor(allocated, kernelAlign, 0, sizesKernel, stridesKernel);
MemRef_descriptor output =
    MemRef_Descriptor(allocated, outputAlign, 0, sizesOutput, stridesOutput);

// Benchmarking function
static void BM_OpenCV(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      filter2D(inputImage, output_opencv, CV_32FC1, kernel_opencv);
    }
  }
}

// Benchmarking function
static void BM_Buddy(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      _mlir_ciface_conv_2d(input, kernel, output);
    }
  }
}

// Register above functions as benchmarks with different arguments
BENCHMARK(BM_OpenCV)->Arg(1);
BENCHMARK(BM_OpenCV)->Arg(2);
BENCHMARK(BM_OpenCV)->Arg(4);
BENCHMARK(BM_OpenCV)->Arg(8);
BENCHMARK(BM_OpenCV)->Arg(16);

BENCHMARK(BM_Buddy)->Arg(1);
BENCHMARK(BM_Buddy)->Arg(2);
BENCHMARK(BM_Buddy)->Arg(4);
BENCHMARK(BM_Buddy)->Arg(8);
BENCHMARK(BM_Buddy)->Arg(16);

// Run benchmarks
int main(int argc, char** argv)
{
  ::benchmark::Initialize(&argc, argv);
  ::benchmark::RunSpecifiedBenchmarks();

  free(inputAlign);
  free(outputAlign);
  free(input);
  free(kernel);
  free(output);
  free(allocated);
}
