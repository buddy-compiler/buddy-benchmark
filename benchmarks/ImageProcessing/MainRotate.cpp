//===- MainRotate.cpp -----------------------------------------------------===//
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
// This is the main file of the image processing rotate benchmark.
//
//===----------------------------------------------------------------------===//

#include <benchmark/benchmark.h>
#include <stdexcept>

void initializeBuddyRotate2D(char **);
void initializeOpenCVRotate2D(char **);

void generateResultBuddyRotate2D();
void generateResultOpenCVRotate2D();

void registerBenchmarkBuddyRotate2D();
void registerBenchmarkOpenCVRotate2D();

// Run benchmarks.
int main(int argc, char **argv) {
  if (argc != 4) {
    throw std::invalid_argument(
        "Wrong format of command line arguments.\n"
        "Correct format is ./image-processing-rotate-benchmark <image path> "
        "<Rotate option> <RotateAngle> \n where "
        "image path provides path of the image to be processed, Rotate option "
        "available are DEGREE, RADIAN. "
        "RotateAngle accepts a float number for Rotate option."
        "OpenCV rotate() only supports 90, 180 and 270 degree.\n");
  }

  initializeBuddyRotate2D(argv);
  initializeOpenCVRotate2D(argv);

  registerBenchmarkBuddyRotate2D();
  registerBenchmarkOpenCVRotate2D();

  ::benchmark::Initialize(&argc, argv);
  ::benchmark::RunSpecifiedBenchmarks();

  // Generate result image.
  generateResultBuddyRotate2D();
  generateResultOpenCVRotate2D();

  return 0;
}
