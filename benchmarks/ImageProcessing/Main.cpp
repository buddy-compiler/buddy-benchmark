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

#include <iostream>

#include <benchmark/benchmark.h>
#include <stdexcept>

#include <buddy/Core/Container.h>
#include <buddy/DIP/ImageContainer.h>
#include <buddy/DIP/imgcodecs/loadsave.h>

void initializeMLIRConv2D(char **, Img<float, 2>);
void initializeBuddyConv2D(char **, Img<float, 2>);
void initializeBuddyCorr2D(char **, Img<float, 2>);
void initializeBuddyMorph2D(char **, Img<float, 2>);
void initializeOpenCVMorph2D(char **);
void initializeOpenCVFilter2D(char **);
void initializeEigenConvolve2D(char **, Img<float, 2>);
void initializeBuddyResize2D(char **, Img<float, 2>);
void initializeBuddyRotate2D(char **, Img<float, 2>);
void initializeOpenCVResize2D(char **);
void initializeOpenCVRotate2D(char **);

void generateResultMLIRConv2D(Img<float, 2>);
void generateResultBuddyConv2D(Img<float, 2>);
void generateResultBuddyCorr2D(char **argv, Img<float, 2>);
void generateResultBuddyResize2D(char **argv, Img<float, 2>);
void generateResultBuddyRotate2D(char **argv, Img<float, 2>);
void generateResultBuddyErosion2D(char **, Img<float, 2>);
void generateResultBuddyOpening2D(char **, Img<float, 2>);
void generateResultBuddyClosing2D(char **, Img<float, 2>);
void generateResultBuddyTopHat2D(char **, Img<float, 2>);
void generateResultBuddyBottomHat2D(char **, Img<float, 2>);
void generateResultBuddyMorphGrad2D(char **, Img<float, 2>);
void generateResultBuddyDilation2D(char **, Img<float, 2>);
void generateResultOpenCVErode2D();
void generateResultOpenCVDilate2D();
void generateResultOpenCVFilter2D();
void generateResultOpenCVOpening2D();
void generateResultOpenCVClosing2D();
void generateResultOpenCVTopHat2D();
void generateResultOpenCVBottomHat2D();
void generateResultOpenCVMorphGrad2D();
void generateResultOpenCVResize2D();
void generateResultOpenCVRotate2D();
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
void registerBenchmarkBuddyResize2D();
void registerBenchmarkBuddyRotate2D();
void registerBenchmarkOpenCVResize2D();
void registerBenchmarkOpenCVRotate2D();

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

  Img<float, 2> img = dip::imread<float, 2>(argv[1], dip::IMGRD_GRAYSCALE);

  initializeMLIRConv2D(argv, img);
  initializeBuddyConv2D(argv, img);
  initializeBuddyCorr2D(argv, img);
  initializeBuddyMorph2D(argv, img);
  initializeOpenCVMorph2D(argv);
  initializeOpenCVFilter2D(argv);
  initializeEigenConvolve2D(argv, img);
  initializeBuddyResize2D(argv, img);
  initializeBuddyRotate2D(argv, img);
  initializeOpenCVResize2D(argv);
  initializeOpenCVRotate2D(argv);

  registerBenchmarkBuddyCorr2D();
  registerBenchmarkOpenCVFilter2D();
  registerBenchmarkBuddyResize2D();
  registerBenchmarkBuddyRotate2D();
  registerBenchmarkOpenCVResize2D();
  registerBenchmarkOpenCVRotate2D();
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
  generateResultMLIRConv2D(img);
  generateResultOpenCVFilter2D();
  generateResultBuddyConv2D(img);
  generateResultBuddyCorr2D(argv, img);
  generateResultBuddyRotate2D(argv, img);
  generateResultBuddyResize2D(argv, img);
  generateResultBuddyErosion2D(argv, img);
  generateResultBuddyDilation2D(argv, img);
  generateResultBuddyOpening2D(argv, img);
  generateResultBuddyClosing2D(argv, img);
  generateResultBuddyTopHat2D(argv, img);
  generateResultBuddyBottomHat2D(argv, img);
  generateResultBuddyMorphGrad2D(argv, img);
  generateResultOpenCVTopHat2D();
  generateResultOpenCVBottomHat2D();
  generateResultOpenCVMorphGrad2D();
  generateResultOpenCVErode2D();
  generateResultOpenCVDilate2D();
  generateResultOpenCVOpening2D();
  generateResultOpenCVClosing2D();
  generateResultOpenCVResize2D();
  generateResultOpenCVRotate2D();
  generateResultEigenConvolve2D();

  return 0;
}
