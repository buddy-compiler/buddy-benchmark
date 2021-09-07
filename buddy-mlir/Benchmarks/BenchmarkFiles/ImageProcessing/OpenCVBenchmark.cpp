#include <benchmark/benchmark.h>
#include <opencv2/opencv.hpp>
#include "../../../kernels.h"

using namespace cv;
using namespace std;

// Read input image and specify kernel
Mat inputImage = imread("../../examples/conv-opt/images/YuTu.png", IMREAD_GRAYSCALE);
Mat kernel_opencv = Mat(3, 3, CV_32FC1, laplacianKernelAlign);

// Declare output image
Mat output_opencv;

// Benchmarking function
static void BM_OpenCV(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      filter2D(inputImage, output_opencv, CV_32FC1, kernel_opencv);
    }
  }
}

// Register benchmarking function with different arguments
BENCHMARK(BM_OpenCV)->Arg(1);
BENCHMARK(BM_OpenCV)->Arg(2);
BENCHMARK(BM_OpenCV)->Arg(4);
BENCHMARK(BM_OpenCV)->Arg(8);
BENCHMARK(BM_OpenCV)->Arg(16);

// Run benchmarks
int main(int argc, char** argv)
{
  ::benchmark::Initialize(&argc, argv);
  ::benchmark::RunSpecifiedBenchmarks();
}
