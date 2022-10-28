#include "ImageProcessing/Kernels.h"
#include "Utils/Container.h"
#include <benchmark/benchmark.h>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

// Declare the conv2d C interface.
extern "C" {
void _mlir_ciface_morphgrad_2d_constant_padding(MemRef<float, 2> *inputBuddyMorphGrad2D,
                                           MemRef<float, 2> *kernelBuddyMorphGrad2D,
                                           MemRef<float, 2> *outputBuddyMorphGrad2D,
                                           MemRef<float, 2> *outputBuddyMorphGrad2D1,
                                           MemRef<float, 2> *outputBuddyMorphGrad2D2,
                                           MemRef<float, 2> *inputBuddyMorphGrad2D1,
                                           MemRef<float, 2> *copyMemRefMorphGrad2D,
                                           MemRef<float, 2>* copyMemRefMorphGrad2D1,
                                           unsigned int centerX,
                                           unsigned int centerY,
                                           unsigned int iterations,
                                           float constantValue
    );

void _mlir_ciface_morphgrad_2d_replicate_padding(MemRef<float, 2> *inputBuddyMorphGrad2D,
                                           MemRef<float, 2> *kernelBuddyMorphGrad2D,
                                           MemRef<float, 2> *outputBuddyMorphGrad2D,
                                           MemRef<float, 2> *outputBuddyMorphGrad2D1,
                                           MemRef<float, 2> *outputBuddyMorphGrad2D2,
                                           MemRef<float, 2> *inputBuddyMorphGrad2D1,
                                           MemRef<float, 2> *copyMemRefMorphGrad2D,
                                           MemRef<float, 2>* copyMemRefMorphGrad2D1,
                                           unsigned int centerX,
                                           unsigned int centerY,
                                           unsigned int iterations,
                                           float constantValue
    );
}

// Declare input image and kernel.
Mat inputImageBuddyMorphGrad2D, kernelBuddyMorphGrad2DMat;

// Define the kernel size.
int kernelRowsBuddyMorphGrad2D, kernelColsBuddyMorphGrad2D;

// Define the output size.
int outputRowsBuddyMorphGrad2D, outputColsBuddyMorphGrad2D;

// Define sizes of input, kernel, and output.
intptr_t sizesInputBuddyMorphGrad2D[2];
intptr_t sizesKernelBuddyMorphGrad2D[2];
intptr_t sizesOutputBuddyMorphGrad2D[2];

// Declare Boundary Options supported.
enum BoundaryOption { constant_padding, replicate_padding };

// Define Boundary option selected.
BoundaryOption BoundaryType7;

void initializeMorphGrad2D(char **argv) {
  inputImageBuddyMorphGrad2D = imread(argv[1], IMREAD_GRAYSCALE);
  kernelBuddyMorphGrad2DMat =
      Mat(get<1>(kernelMap[argv[2]]), get<2>(kernelMap[argv[2]]), CV_32FC1,
          get<0>(kernelMap[argv[2]]));

  kernelRowsBuddyMorphGrad2D = kernelBuddyMorphGrad2DMat.rows;
  kernelColsBuddyMorphGrad2D = kernelBuddyMorphGrad2DMat.cols;

  outputRowsBuddyMorphGrad2D = inputImageBuddyMorphGrad2D.rows;
  outputColsBuddyMorphGrad2D = inputImageBuddyMorphGrad2D.cols;

  sizesInputBuddyMorphGrad2D[0] = inputImageBuddyMorphGrad2D.rows;
  sizesInputBuddyMorphGrad2D[1] = inputImageBuddyMorphGrad2D.cols;

  sizesKernelBuddyMorphGrad2D[0] = kernelRowsBuddyMorphGrad2D;
  sizesKernelBuddyMorphGrad2D[1] = kernelColsBuddyMorphGrad2D;

  sizesOutputBuddyMorphGrad2D[0] = outputRowsBuddyMorphGrad2D;
  sizesOutputBuddyMorphGrad2D[1] = outputColsBuddyMorphGrad2D;

  if (static_cast<string>(argv[3]) == "REPLICATE_PADDING") {
    BoundaryType7 = replicate_padding;
  } else {
    BoundaryType7 = constant_padding;
  }
}

static void Buddy_MorphGrad2D_Constant_Padding(benchmark::State &state) {
  // Define the MemRef descriptor for input, kernel, and output.
  MemRef<float, 2> inputBuddyMorphGrad2D(inputImageBuddyMorphGrad2D,
                                    sizesInputBuddyMorphGrad2D);
    MemRef<float, 2> inputBuddyMorphGrad2D1(inputImageBuddyMorphGrad2D,
                                    sizesInputBuddyMorphGrad2D);                                  
  MemRef<float, 2> kernelBuddyMorphGrad2D(kernelBuddyMorphGrad2DMat,
                                     sizesKernelBuddyMorphGrad2D);
  MemRef<float, 2> outputBuddyMorphGrad2D(sizesOutputBuddyMorphGrad2D);
  MemRef<float, 2> outputBuddyMorphGrad2D1(sizesOutputBuddyMorphGrad2D);
  MemRef<float, 2> outputBuddyMorphGrad2D2(sizesOutputBuddyMorphGrad2D);  
  MemRef<float, 2> copyMemRefMorphGrad2D(sizesOutputBuddyMorphGrad2D, 256.f);
  MemRef<float, 2> copyMemRefMorphGrad2D1(sizesOutputBuddyMorphGrad2D, -1.f);

  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      _mlir_ciface_morphgrad_2d_constant_padding(
          &inputBuddyMorphGrad2D, &kernelBuddyMorphGrad2D, &outputBuddyMorphGrad2D, &outputBuddyMorphGrad2D1, &outputBuddyMorphGrad2D2, &inputBuddyMorphGrad2D1, &copyMemRefMorphGrad2D, &copyMemRefMorphGrad2D1,
          1 /* Center X */, 1 /* Center Y */,5, 0.0f /* Constant Value */);
    }
  }
}

static void Buddy_MorphGrad2D_Replicate_Padding(benchmark::State &state) {
  // Define the MemRef descriptor for input, kernel, and output.
  MemRef<float, 2> inputBuddyMorphGrad2D(inputImageBuddyMorphGrad2D,
                                    sizesInputBuddyMorphGrad2D);
    MemRef<float, 2> inputBuddyMorphGrad2D1(inputImageBuddyMorphGrad2D,
                                    sizesInputBuddyMorphGrad2D);                                  
  MemRef<float, 2> kernelBuddyMorphGrad2D(kernelBuddyMorphGrad2DMat,
                                     sizesKernelBuddyMorphGrad2D);
  MemRef<float, 2> outputBuddyMorphGrad2D(sizesOutputBuddyMorphGrad2D);
  MemRef<float, 2> outputBuddyMorphGrad2D1(sizesOutputBuddyMorphGrad2D);
  MemRef<float, 2> outputBuddyMorphGrad2D2(sizesOutputBuddyMorphGrad2D);  
  MemRef<float, 2> copyMemRefMorphGrad2D(sizesOutputBuddyMorphGrad2D, 256.f);
  MemRef<float, 2> copyMemRefMorphGrad2D1(sizesOutputBuddyMorphGrad2D, -1.f);

  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      _mlir_ciface_morphgrad_2d_replicate_padding(
          &inputBuddyMorphGrad2D, &kernelBuddyMorphGrad2D, &outputBuddyMorphGrad2D, &outputBuddyMorphGrad2D1, &outputBuddyMorphGrad2D2, &inputBuddyMorphGrad2D1, &copyMemRefMorphGrad2D, &copyMemRefMorphGrad2D1,
          1 /* Center X */, 1 /* Center Y */,5, 0.0f /* Constant Value */);
    }
  }
}

// Register benchmarking function.
void registerBenchmarkBuddyMorphGrad2D() {
  if (BoundaryType7 == replicate_padding) {
    BENCHMARK(Buddy_MorphGrad2D_Replicate_Padding)->Arg(1);
  } else {
    BENCHMARK(Buddy_MorphGrad2D_Constant_Padding)->Arg(1);
  }
}

// Generate result image.
void generateResultBuddyMorphGrad2D(char **argv) {
  // Define the MemRef descriptor for input, kernel, and output.
  MemRef<float, 2> input(inputImageBuddyMorphGrad2D, sizesInputBuddyMorphGrad2D);
  MemRef<float, 2> kernel(get<0>(kernelMap[argv[2]]), sizesKernelBuddyMorphGrad2D);
  MemRef<float, 2> output(sizesOutputBuddyMorphGrad2D);
  MemRef<float, 2> output1(sizesOutputBuddyMorphGrad2D);
  MemRef<float, 2> output2(sizesOutputBuddyMorphGrad2D);
  MemRef<float, 2> input1(sizesOutputBuddyMorphGrad2D);    
  MemRef<float, 2> copymemref(sizesOutputBuddyMorphGrad2D, 256.f);
  MemRef<float, 2> copymemref1(sizesOutputBuddyMorphGrad2D, -1.f);
  // Run the 2D MorphGrad operation
  if (static_cast<string>(argv[3]) == "REPLICATE_PADDING") {
    _mlir_ciface_morphgrad_2d_replicate_padding(&input, &kernel, &output, &output1, &output2, &input1, &copymemref, &copymemref1,
                                           1 /* Center X */, 1 /* Center Y */, 5,
                                           0.0f /* Constant Value */);
  } else {
    _mlir_ciface_morphgrad_2d_constant_padding(&input, &kernel, &output, &output1, &output2, &input1, &copymemref, &copymemref1,
                                          1 /* Center X */, 1 /* Center Y */, 5,
                                          0.0f /* Constant Value */);
  }

  // Define a cv::Mat with the output of the MorphGrad
  Mat outputImage(outputRowsBuddyMorphGrad2D, outputRowsBuddyMorphGrad2D, CV_32FC1,
                  output.getData());

  // Choose a PNG compression level
  vector<int> compressionParams;
  compressionParams.push_back(IMWRITE_PNG_COMPRESSION);
  compressionParams.push_back(9);

  // Write output to PNG.
  bool result = false;
  try {
    result = imwrite("ResultBuddyMorphGrad2D.png", outputImage, compressionParams);
  } catch (const cv::Exception &ex) {
    fprintf(stderr, "Exception converting image to PNG format: %s\n",
            ex.what());
  }
  if (result)
    cout << "Saved PNG file." << endl;
  else
    cout << "ERROR: Can't save PNG file." << endl;
}
