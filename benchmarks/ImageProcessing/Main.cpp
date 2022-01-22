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

void initializeBM_Conv2D_Buddy(int, char**);
void initializeBM_Corr2D_Buddy(int, char**);
void initializeBM_Filter2D_OpenCV(int, char**);

void generateResultConv2D();
void generateResultCorr2D();

// Run benchmarks.
int main(int argc, char **argv) {
  if (argc != 3)
  {
    throw std::invalid_argument("Wrong format of command line arguments.\n"
    "Correct format is ./image-processing-benchmark <image path> <kernel name>\n where "
    "image path provides path of the image to be processed and kernel name denotes the name "
    "of desired kernel as specified in include/ImageProcessing/Kernels.h\n");
  }

  initializeBM_Conv2D_Buddy(argc, argv);
  initializeBM_Corr2D_Buddy(argc, argv);
  initializeBM_Filter2D_OpenCV(argc, argv);

  ::benchmark::Initialize(&argc, argv);
  ::benchmark::RunSpecifiedBenchmarks();
  // Generate result image.
  generateResultConv2D();
  generateResultCorr2D();

  return 0;
}
