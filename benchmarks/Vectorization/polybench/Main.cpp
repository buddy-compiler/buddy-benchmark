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

#include "Utils.hpp"
#include "benchmark/benchmark.h"

#include <cstring>
#include <iostream>
#include <string>

int main(int argc, char **argv) {
  if (argc > 1 && std::string(argv[1]).find("--generate-output=") == 0) {
    std::string sizeStr =
        std::string(argv[1]).substr(strlen("--generate-output="));
    int size_id = polybench::getPolybenchDatasetSizeID(sizeStr);
    if (size_id == -1) {
      std::cerr << "Invalid dataset size: " << sizeStr << std::endl;
      return 1;
    }
    GENERATE_RESULT_FUNCTION_CALLS(size_id);
  } else {
    ::benchmark::Initialize(&argc, argv);
    ::benchmark::RunSpecifiedBenchmarks();

    for (int i = 1; i < argc; ++i) {
      if (std::string(argv[i]).find("--verification-dataset-size=") == 0) {
        std::string sizeStr = std::string(argv[i]).substr(
            std::strlen("--verification-dataset-size="));
        int size_id = polybench::getPolybenchDatasetSizeID(sizeStr);
        if (size_id == -1) {
          std::cerr << "Invalid dataset size: " << sizeStr << std::endl;
          return 1;
        }

        std::cout << "------------------------------------------------"
                  << std::endl;
        VERIFY_RESULT_FUNCTION_CALLS(size_id);
        std::cout << "------------------------------------------------"
                  << std::endl;
        break;
      }
    }
  }
}
