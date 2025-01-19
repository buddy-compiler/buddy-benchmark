//===- Utils.cpp ----------------------------------------------------------===//
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

#ifndef POLYBENCH_UTILS_HPP
#define POLYBENCH_UTILS_HPP

#include <iostream>
#include <string>

namespace polybench {

// Mimic the behavior of POLYBENCH_DUMP_START macro.
inline void startDump() { std::cout << "==BEGIN DUMP_ARRAYS==" << std::endl; }

// Mimic the behavior of POLYBENCH_DUMP_FINISH macro.
inline void finishDump() { std::cout << "==END   DUMP_ARRAYS==" << std::endl; }

// Mimic the behavior of POLYBENCH_DUMP_BEGIN macro.
inline void beginDump(const std::string &name) {
  std::cout << "begin dump: " << name;
}

// Mimic the behavior of POLYBENCH_DUMP_END macro.
inline void endDump(const std::string &name) {
  std::cout << std::endl << "end   dump: " << name << std::endl;
}

// Get the name of a dataset size by its ID.
inline std::string getPolybenchDatasetSizeName(int size_id) {
  switch (size_id) {
  case 0:
    return "mini";
  case 1:
    return "small";
  case 2:
    return "medium";
  case 3:
    return "large";
  case 4:
    return "extralarge";
  default:
    return "unknown";
  }
}

// Get the ID of a dataset size by its name.
inline int getPolybenchDatasetSizeID(const std::string &name) {
  if (name == "mini") {
    return 0;
  } else if (name == "small") {
    return 1;
  } else if (name == "medium") {
    return 2;
  } else if (name == "large") {
    return 3;
  } else if (name == "extralarge") {
    return 4;
  } else {
    return -1;
  }
}

// Verification function. Derived from DeepLearning benchmark.
template <typename DATA_TYPE>
void verify(DATA_TYPE *A, DATA_TYPE *B, int size, const std::string &name) {
  const std::string PASS = "\033[32mPASS\033[0m";
  const std::string FAIL = "\033[31mFAIL\033[0m";

  std::cout << name << " ";
  if (!A || !B) {
    std::cout << FAIL << " (Null pointer detected)" << std::endl;
    return;
  }

  bool isPass = true;
  for (int i = 0; i < size; ++i) {
    if (A[i] != B[i]) {
      std::cout << FAIL << std::endl;
      std::cout << "Index " << i << ":\tA=" << A[i] << " B=" << B[i]
                << std::endl;
      isPass = false;
      break;
    }
  }
  if (isPass) {
    std::cout << PASS << std::endl;
  }
}

} // namespace polybench

void generateResultMLIRPolybench2mm(size_t);
void generateResultMLIRPolybench3mm(size_t);
void generateResultMLIRPolybenchAdi(size_t);
void generateResultMLIRPolybenchAtax(size_t);
void generateResultMLIRPolybenchBicg(size_t);
void generateResultMLIRPolybenchCholesky(size_t);
void generateResultMLIRPolybenchCorrelation(size_t);
void generateResultMLIRPolybenchCovariance(size_t);
void generateResultMLIRPolybenchDeriche(size_t);
void generateResultMLIRPolybenchDoitgen(size_t);
void generateResultMLIRPolybenchDurbin(size_t);
void generateResultMLIRPolybenchFdtd2D(size_t);
void generateResultMLIRPolybenchFloydWarshall(size_t);
void generateResultMLIRPolybenchGemm(size_t);
void generateResultMLIRPolybenchGemver(size_t);
void generateResultMLIRPolybenchGesummv(size_t);
void generateResultMLIRPolybenchGramschmidt(size_t);
void generateResultMLIRPolybenchHeat3D(size_t);
void generateResultMLIRPolybenchJacobi1D(size_t);
void generateResultMLIRPolybenchJacobi2D(size_t);
void generateResultMLIRPolybenchLu(size_t);
void generateResultMLIRPolybenchLudcmp(size_t);
void generateResultMLIRPolybenchMvt(size_t);
void generateResultMLIRPolybenchNussinov(size_t);
void generateResultMLIRPolybenchSeidel2D(size_t);
void generateResultMLIRPolybenchSymm(size_t);
void generateResultMLIRPolybenchSyr2k(size_t);
void generateResultMLIRPolybenchSyrk(size_t);
void generateResultMLIRPolybenchTrisolv(size_t);
void generateResultMLIRPolybenchTrmm(size_t);

void verifyResultMLIRPolybench2mm(size_t);
void verifyResultMLIRPolybench3mm(size_t);
void verifyResultMLIRPolybenchAdi(size_t);
void verifyResultMLIRPolybenchAtax(size_t);
void verifyResultMLIRPolybenchBicg(size_t);
void verifyResultMLIRPolybenchCholesky(size_t);
void verifyResultMLIRPolybenchCorrelation(size_t);
void verifyResultMLIRPolybenchCovariance(size_t);
void verifyResultMLIRPolybenchDeriche(size_t);
void verifyResultMLIRPolybenchDoitgen(size_t);
void verifyResultMLIRPolybenchDurbin(size_t);
void verifyResultMLIRPolybenchFdtd2D(size_t);
void verifyResultMLIRPolybenchFloydWarshall(size_t);
void verifyResultMLIRPolybenchGemm(size_t);
void verifyResultMLIRPolybenchGemver(size_t);
void verifyResultMLIRPolybenchGesummv(size_t);
void verifyResultMLIRPolybenchGramschmidt(size_t);
void verifyResultMLIRPolybenchHeat3D(size_t);
void verifyResultMLIRPolybenchJacobi1D(size_t);
void verifyResultMLIRPolybenchJacobi2D(size_t);
void verifyResultMLIRPolybenchLu(size_t);
void verifyResultMLIRPolybenchLudcmp(size_t);
void verifyResultMLIRPolybenchMvt(size_t);
void verifyResultMLIRPolybenchNussinov(size_t);
void verifyResultMLIRPolybenchSeidel2D(size_t);
void verifyResultMLIRPolybenchSymm(size_t);
void verifyResultMLIRPolybenchSyr2k(size_t);
void verifyResultMLIRPolybenchSyrk(size_t);
void verifyResultMLIRPolybenchTrisolv(size_t);
void verifyResultMLIRPolybenchTrmm(size_t);

#define GENERATE_RESULT_FUNCTION_CALLS(size_id)                                \
  generateResultMLIRPolybench2mm(size_id);                                     \
  generateResultMLIRPolybench3mm(size_id);                                     \
  generateResultMLIRPolybenchAdi(size_id);                                     \
  generateResultMLIRPolybenchAtax(size_id);                                    \
  generateResultMLIRPolybenchBicg(size_id);                                    \
  generateResultMLIRPolybenchCholesky(size_id);                                \
  generateResultMLIRPolybenchCorrelation(size_id);                             \
  generateResultMLIRPolybenchCovariance(size_id);                              \
  generateResultMLIRPolybenchDeriche(size_id);                                 \
  generateResultMLIRPolybenchDoitgen(size_id);                                 \
  generateResultMLIRPolybenchDurbin(size_id);                                  \
  generateResultMLIRPolybenchFdtd2D(size_id);                                  \
  generateResultMLIRPolybenchFloydWarshall(size_id);                           \
  generateResultMLIRPolybenchGemm(size_id);                                    \
  generateResultMLIRPolybenchGemver(size_id);                                  \
  generateResultMLIRPolybenchGesummv(size_id);                                 \
  generateResultMLIRPolybenchGramschmidt(size_id);                             \
  generateResultMLIRPolybenchHeat3D(size_id);                                  \
  generateResultMLIRPolybenchJacobi1D(size_id);                                \
  generateResultMLIRPolybenchJacobi2D(size_id);                                \
  generateResultMLIRPolybenchLu(size_id);                                      \
  generateResultMLIRPolybenchLudcmp(size_id);                                  \
  generateResultMLIRPolybenchMvt(size_id);                                     \
  generateResultMLIRPolybenchNussinov(size_id);                                \
  generateResultMLIRPolybenchSeidel2D(size_id);                                \
  generateResultMLIRPolybenchSymm(size_id);                                    \
  generateResultMLIRPolybenchSyr2k(size_id);                                   \
  generateResultMLIRPolybenchSyrk(size_id);                                    \
  generateResultMLIRPolybenchTrisolv(size_id);                                 \
  generateResultMLIRPolybenchTrmm(size_id);

#define VERIFY_RESULT_FUNCTION_CALLS(size_id)                                  \
  verifyResultMLIRPolybench2mm(size_id);                                       \
  verifyResultMLIRPolybench3mm(size_id);                                       \
  verifyResultMLIRPolybenchAdi(size_id);                                       \
  verifyResultMLIRPolybenchAtax(size_id);                                      \
  verifyResultMLIRPolybenchBicg(size_id);                                      \
  verifyResultMLIRPolybenchCholesky(size_id);                                  \
  verifyResultMLIRPolybenchCorrelation(size_id);                               \
  verifyResultMLIRPolybenchCovariance(size_id);                                \
  verifyResultMLIRPolybenchDeriche(size_id);                                   \
  verifyResultMLIRPolybenchDoitgen(size_id);                                   \
  verifyResultMLIRPolybenchDurbin(size_id);                                    \
  verifyResultMLIRPolybenchFdtd2D(size_id);                                    \
  verifyResultMLIRPolybenchFloydWarshall(size_id);                             \
  verifyResultMLIRPolybenchGemm(size_id);                                      \
  verifyResultMLIRPolybenchGemver(size_id);                                    \
  verifyResultMLIRPolybenchGesummv(size_id);                                   \
  verifyResultMLIRPolybenchGramschmidt(size_id);                               \
  verifyResultMLIRPolybenchHeat3D(size_id);                                    \
  verifyResultMLIRPolybenchJacobi1D(size_id);                                  \
  verifyResultMLIRPolybenchJacobi2D(size_id);                                  \
  verifyResultMLIRPolybenchLu(size_id);                                        \
  verifyResultMLIRPolybenchLudcmp(size_id);                                    \
  verifyResultMLIRPolybenchMvt(size_id);                                       \
  verifyResultMLIRPolybenchNussinov(size_id);                                  \
  verifyResultMLIRPolybenchSeidel2D(size_id);                                  \
  verifyResultMLIRPolybenchSymm(size_id);                                      \
  verifyResultMLIRPolybenchSyr2k(size_id);                                     \
  verifyResultMLIRPolybenchSyrk(size_id);                                      \
  verifyResultMLIRPolybenchTrisolv(size_id);                                   \
  verifyResultMLIRPolybenchTrmm(size_id);

#endif // POLYBENCH_UTILS_HPP
