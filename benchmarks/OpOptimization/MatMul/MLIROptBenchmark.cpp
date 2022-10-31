//===- MLIROptBenchmark.cpp -----------------------------------------------===//
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
// This file implements the benchmark for GEMM operation.
//
//===----------------------------------------------------------------------===//

#include <benchmark/benchmark.h>
#include <buddy/core/Container.h>
#include <cmath>
#include <cstdlib>
#include <iostream>

#include <immintrin.h>
#include <opencv2/core.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/matx.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/opencv.hpp>

namespace {

// Declare the mobilenet C interface.
extern "C" {
void _mlir_ciface_gemm(MemRef<float, 2> *A, MemRef<float, 2> *B,
                       MemRef<float, 2> *C);
}

// TODO: avoid avx512
void fastGEMM(const float *aptr, size_t astep, const float *bptr, size_t bstep,
              float *cptr, size_t cstep, int ma, int na, int nb) {
  int n = 0;

  for (; n <= nb - 32; n += 32) {
    for (int m = 0; m < ma; m += 4) {
      const float *aptr0 = aptr + astep * m;
      const float *aptr1 = aptr + astep * std::min(m + 1, ma - 1);
      const float *aptr2 = aptr + astep * std::min(m + 2, ma - 1);
      const float *aptr3 = aptr + astep * std::min(m + 3, ma - 1);

      float *cptr0 = cptr + cstep * m;
      float *cptr1 = cptr + cstep * std::min(m + 1, ma - 1);
      float *cptr2 = cptr + cstep * std::min(m + 2, ma - 1);
      float *cptr3 = cptr + cstep * std::min(m + 3, ma - 1);

      __m512 d00 = _mm512_setzero_ps(), d01 = _mm512_setzero_ps();
      __m512 d10 = _mm512_setzero_ps(), d11 = _mm512_setzero_ps();
      __m512 d20 = _mm512_setzero_ps(), d21 = _mm512_setzero_ps();
      __m512 d30 = _mm512_setzero_ps(), d31 = _mm512_setzero_ps();

      for (int k = 0; k < na; k++) {
        __m512 a0 = _mm512_set1_ps(aptr0[k]);
        __m512 a1 = _mm512_set1_ps(aptr1[k]);
        __m512 a2 = _mm512_set1_ps(aptr2[k]);
        __m512 a3 = _mm512_set1_ps(aptr3[k]);
        __m512 b0 = _mm512_loadu_ps(bptr + k * bstep + n);
        __m512 b1 = _mm512_loadu_ps(bptr + k * bstep + n + 16);
        d00 = _mm512_fmadd_ps(a0, b0, d00);
        d01 = _mm512_fmadd_ps(a0, b1, d01);
        d10 = _mm512_fmadd_ps(a1, b0, d10);
        d11 = _mm512_fmadd_ps(a1, b1, d11);
        d20 = _mm512_fmadd_ps(a2, b0, d20);
        d21 = _mm512_fmadd_ps(a2, b1, d21);
        d30 = _mm512_fmadd_ps(a3, b0, d30);
        d31 = _mm512_fmadd_ps(a3, b1, d31);
      }

      _mm512_storeu_ps(cptr0 + n, d00);
      _mm512_storeu_ps(cptr0 + n + 16, d01);
      _mm512_storeu_ps(cptr1 + n, d10);
      _mm512_storeu_ps(cptr1 + n + 16, d11);
      _mm512_storeu_ps(cptr2 + n, d20);
      _mm512_storeu_ps(cptr2 + n + 16, d21);
      _mm512_storeu_ps(cptr3 + n, d30);
      _mm512_storeu_ps(cptr3 + n + 16, d31);
    }
  }
}

void BM_GEMM(benchmark::State &state) {
  long M = state.range(0), N = state.range(0), K = state.range(0);
  intptr_t sizesA[2] = {M, K};
  intptr_t sizesB[2] = {K, N};
  intptr_t sizesC[2] = {M, N};

  MemRef<float, 2> A(sizesA, 1.0);
  MemRef<float, 2> B(sizesB, 1.0);
  MemRef<float, 2> C(sizesC, 0);

  int cnt = 0;
  for (auto _ : state) {
    _mlir_ciface_gemm(&A, &B, &C);
    cnt++;
  }

#ifdef OP_TEST
  cv::Mat cvA = cv::Mat::ones(M, N, CV_32F);
  cv::Mat cvB = cv::Mat::ones(M, N, CV_32F);
  cv::Mat cvC = cv::Mat::zeros(M, N, CV_32F);
  for (int i = 0; i < cnt; i++)
    cv::gemm(cvA, cvB, 1.0, cvC, 1.0, cvC, 0);
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      if (C.getData()[i * M + j] == cvC.at<float>(i, j)) {

      } else {
        std::cout << M << std::endl;
        std::cout << "[" << i << ", " << j << "] == " << C.getData()[i * M + j]
                  << ", expect " << cvC.at<float>(i, j) << std::endl;
      }
    }
  }
#endif
#ifdef DEBUG
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      std::cout << C.getData()[i * M + j] << " ";
    }
    std::cout << std::endl;
  }
  assert(false);
#endif
}

void BM_OPENCV_GEMM(benchmark::State &state) {
  long M = state.range(0), N = state.range(0), K = state.range(0);

  float *o_A = (float *)malloc(sizeof(float) * M * K);
  memset(o_A, sizeof(float) * M * K, 1.0);
  float *o_B = (float *)malloc(sizeof(float) * K * N);
  memset(o_B, sizeof(float) * K * N, 1.0);
  float *o_C = (float *)malloc(sizeof(float) * M * N);
  memset(o_C, sizeof(float) * M * N, 0.0);

  for (auto _ : state) {
    // C += A * B;
    fastGEMM(o_A, K, o_B, M, o_C, N, M, K, N);
  }

  free(o_C);
  free(o_B);
  free(o_A);
}

void BM_RAW_GEMM(benchmark::State &state) {
  long M = state.range(0), N = state.range(0), K = state.range(0);
  float *A = (float *)malloc(sizeof(float) * M * K);
  float *B = (float *)malloc(sizeof(float) * K * N);
  float *C = (float *)malloc(sizeof(float) * M * N);

  for (auto _ : state) {
    // C += A * B;
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        for (int k = 0; k < K; k++) {
          C[i * M + j] += A[i * M + k] * B[k * K + j];
        }
      }
    }
  }
}

} // namespace

// Register benchmarking function with different arguments.
BENCHMARK(BM_GEMM)->DenseRange(64, 2048, 64);
BENCHMARK(BM_OPENCV_GEMM)->DenseRange(64, 2048, 64);
