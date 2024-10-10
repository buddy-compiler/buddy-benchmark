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
void initializeOpenCVResize2D(char **);
void initializeBuddyResize4D(char **, Img<float, 4>);

void generateResultMLIRConv2D(Img<float, 2>);
void generateResultBuddyConv2D(Img<float, 2>);
void generateResultBuddyCorr2D(char **argv, Img<float, 2>);
void generateResultBuddyResize2D(char **argv, Img<float, 2>);
void generateResultBuddyResize4D(char **argv, Img<float, 4>);
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
void generateResultEigenConvolve2D();

void registerBenchmarkBuddyCorr2D();
void registerBenchmarkBuddyErosion2D();
void registerBenchmarkBuddyDilation2D();
void registerBenchmarkBuddyOpening2D();
void registerBenchmarkBuddyClosing2D();
void registerBenchmarkBuddyTopHat2D();
void registerBenchmarkBuddyBottomHat2D();
void registerBenchmarkBuddyMorphGrad2D();
void registerBenchmarkBuddyResize4D();
void registerBenchmarkOpenCVErode2D();
void registerBenchmarkOpenCVDilate2D();
void registerBenchmarkOpenCVOpening2D();
void registerBenchmarkOpenCVClosing2D();
void registerBenchmarkOpenCVTopHat2D();
void registerBenchmarkOpenCVBottomHat2D();
void registerBenchmarkOpenCVMorphGrad2D();
void registerBenchmarkOpenCVFilter2D();
void registerBenchmarkBuddyResize2D();
void registerBenchmarkOpenCVResize2D();

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
  Img<float, 3> imgColar = dip::imread<float, 3>(argv[1], dip::IMGRD_COLOR);
  intptr_t sizes[4] = {1, imgColar.getSizes()[0], imgColar.getSizes()[1], imgColar.getSizes()[2]};
  Img<float, 4> imgColarBatch(imgColar.getData(), sizes);

  initializeMLIRConv2D(argv, img);
  initializeBuddyConv2D(argv, img);
  initializeBuddyCorr2D(argv, img);
  initializeBuddyMorph2D(argv, img);
  initializeOpenCVMorph2D(argv);
  initializeOpenCVFilter2D(argv);
  initializeEigenConvolve2D(argv, img);
  initializeBuddyResize2D(argv, img);
  initializeOpenCVResize2D(argv);
  initializeBuddyResize4D(argv, imgColarBatch);

  registerBenchmarkBuddyCorr2D();
  registerBenchmarkOpenCVFilter2D();
  registerBenchmarkBuddyResize2D();
  registerBenchmarkOpenCVResize2D();
  registerBenchmarkBuddyErosion2D();
  registerBenchmarkBuddyDilation2D();
  registerBenchmarkBuddyOpening2D();
  registerBenchmarkBuddyClosing2D();
  registerBenchmarkBuddyTopHat2D();
  registerBenchmarkBuddyBottomHat2D();
  registerBenchmarkBuddyResize4D();
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
  generateResultBuddyResize2D(argv, img);
  generateResultBuddyErosion2D(argv, img);
  generateResultBuddyDilation2D(argv, img);
  generateResultBuddyOpening2D(argv, img);
  generateResultBuddyClosing2D(argv, img);
  generateResultBuddyTopHat2D(argv, img);
  generateResultBuddyBottomHat2D(argv, img);
  generateResultBuddyMorphGrad2D(argv, img);
  generateResultBuddyResize4D(argv, imgColarBatch);
  generateResultOpenCVTopHat2D();
  generateResultOpenCVBottomHat2D();
  generateResultOpenCVMorphGrad2D();
  generateResultOpenCVErode2D();
  generateResultOpenCVDilate2D();
  generateResultOpenCVOpening2D();
  generateResultOpenCVClosing2D();
  generateResultOpenCVResize2D();
  generateResultEigenConvolve2D();

  return 0;
}
