//===- Main.cpp -----------------------------------------------------------===//
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
// This is the main file of the image processing benchmark.
//
//===----------------------------------------------------------------------===//

#include <benchmark/benchmark.h>
#include <stdexcept>

void initializeMLIRConv2D(char **);
void initializeBuddyConv2D(char **);
void initializeBuddyCorr2D(char **);
void initializeBuddyMorph2D(char **);
void initializeOpenCVMorph2D(char **);
void initializeOpenCVFilter2D(char **);
void initializeEigenConvolve2D(char **);

void generateResultBuddyConv2D(char **);
void generateResultBuddyCorr2D(char **);
void generateResultBuddyErosion2D(char **);
void generateResultBuddyOpening2D(char **);
void generateResultBuddyClosing2D(char **);
void generateResultBuddyTopHat2D(char **);
void generateResultBuddyBottomHat2D(char **);
void generateResultBuddyMorphGrad2D(char **);
void generateResultBuddyDilation2D(char **);
void generateResultOpenCVErode2D();
void generateResultOpenCVDilate2D();
void generateResultOpenCVFilter2D();
void generateResultOpenCVOpening2D();
void generateResultOpenCVClosing2D();
void generateResultOpenCVTopHat2D();
void generateResultOpenCVBottomHat2D();
void generateResultOpenCVMorphGrad2D();
void generateResultEigenConvolve2D();

void registerBenchmarkBuddyCorr2D();
void registerBenchmarkBuddyErosion2D();
void registerBenchmarkBuddyDilation2D();
void registerBenchmarkBuddyOpening2D();
void registerBenchmarkBuddyClosing2D();
void registerBenchmarkBuddyTopHat2D();
void registerBenchmarkBuddyBottomHat2D();
void registerBenchmarkBuddyMorphGrad2D();
void registerBenchmarkOpenCVErode2D();
void registerBenchmarkOpenCVDilate2D();
void registerBenchmarkOpenCVOpening2D();
void registerBenchmarkOpenCVClosing2D();
void registerBenchmarkOpenCVTopHat2D();
void registerBenchmarkOpenCVBottomHat2D();
void registerBenchmarkOpenCVMorphGrad2D();
void registerBenchmarkOpenCVFilter2D();

// Run benchmarks.
int main(int argc, char **argv) {
  if (argc != 5) {
    throw std::invalid_argument(
        "Wrong format of command line arguments.\n"
        "Correct format is ./image-processing-benchmark <image path> <kernel "
        "name> <kernelmorph> <Boundary Option>\n where "
        "image path provides path of the image to be processed, kernel name "
        "denotes the name "
        "of desired kernel as specified in "
        "kernelmorph denotes the kernel to be used for morphological operations"
        "include/ImageProcessing/Kernels.h and Boundary options available "
        "are CONSTANT_PADDING, REPLICATE_PADDING.\n");
  }

  initializeMLIRConv2D(argv);
  initializeBuddyConv2D(argv);
  initializeBuddyCorr2D(argv);
  initializeBuddyMorph2D(argv);
  initializeOpenCVMorph2D(argv);
  initializeOpenCVFilter2D(argv);
  initializeEigenConvolve2D(argv);

  registerBenchmarkBuddyCorr2D();
  registerBenchmarkOpenCVFilter2D();
  registerBenchmarkBuddyErosion2D();
  registerBenchmarkBuddyDilation2D();
  registerBenchmarkBuddyOpening2D();
  registerBenchmarkBuddyClosing2D();
  registerBenchmarkBuddyTopHat2D();
  registerBenchmarkBuddyBottomHat2D();
  registerBenchmarkOpenCVErode2D();
  registerBenchmarkOpenCVOpening2D();
  registerBenchmarkOpenCVClosing2D();
  registerBenchmarkOpenCVTopHat2D();
  registerBenchmarkOpenCVBottomHat2D();
  registerBenchmarkOpenCVMorphGrad2D();
  registerBenchmarkOpenCVDilate2D();

  ::benchmark::Initialize(&argc, argv);
  ::benchmark::RunSpecifiedBenchmarks();

  // Generate result image.
  generateResultOpenCVFilter2D();
  generateResultBuddyConv2D(argv);
  generateResultBuddyCorr2D(argv);
  generateResultBuddyErosion2D(argv);
  generateResultBuddyDilation2D(argv);
  generateResultBuddyOpening2D(argv);
  generateResultBuddyClosing2D(argv);
  generateResultBuddyTopHat2D(argv);
  generateResultBuddyBottomHat2D(argv);
  generateResultBuddyMorphGrad2D(argv);
  generateResultOpenCVTopHat2D();
  generateResultOpenCVBottomHat2D();
  generateResultOpenCVMorphGrad2D();
  generateResultOpenCVErode2D();
  generateResultOpenCVDilate2D();
  generateResultOpenCVOpening2D();
  generateResultOpenCVClosing2D();
  generateResultEigenConvolve2D();

  return 0;
}
