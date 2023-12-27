//===- MainResize.cpp -----------------------------------------------------===//
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
// This is the main file of the image processing resize benchmark.
//
//===----------------------------------------------------------------------===//

#include <benchmark/benchmark.h>
#include <stdexcept>

void initializeBuddyResize2D(char **);
void initializeOpenCVResize2D(char **);

void generateResultBuddyResize2D();
void generateResultOpenCVResize2D();

void registerBenchmarkBuddyResize2D();
void registerBenchmarkOpenCVResize2D();

// Run benchmarks.
int main(int argc, char **argv) {
  if (argc != 6) {
    throw std::invalid_argument(
        "Wrong format of command line arguments.\n"
        "Correct format is ./image-processing-resize-benchmark <image path> "
        "<Scale option> <RowNum> <ColNum> <InterpolationOption>\n where "
        "image path provides path of the image to be processed, Scale option "
        "available are SCALE_FACTOR, SCALE_LENGTH. "
        "RowNum and ColNum are the "
        "scale_factors/scale_length for row and col, "
        "Interpolation option available "
        "are NEAREST_NEIGHBOUR_INTERPOLATION, BILINEAR_INTERPOLATION.\n");
  }

  initializeBuddyResize2D(argv);
  initializeOpenCVResize2D(argv);

  registerBenchmarkBuddyResize2D();
  registerBenchmarkOpenCVResize2D();

  ::benchmark::Initialize(&argc, argv);
  ::benchmark::RunSpecifiedBenchmarks();

  // Generate result image.
  generateResultBuddyResize2D();
  generateResultOpenCVResize2D();

  return 0;
}
