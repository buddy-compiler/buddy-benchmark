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

#include <buddy/core/Container.h>
#include <benchmark/benchmark.h>
#include <cmath>
#include <iostream>
#include <cstdlib>

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

void BM_GEMM(benchmark::State &state) {
  long M = state.range(0), N = state.range(0), K = state.range(0);
  intptr_t sizesA[2] = {M, K};
  intptr_t sizesB[2] = {K, N};
  intptr_t sizesC[2] = {M, N};

  MemRef<float, 2> A(sizesA, 1.0);
  MemRef<float, 2> B(sizesB, 1.0);
  MemRef<float, 2> C(sizesC, 0.0);

  for (auto _ : state) {
    _mlir_ciface_gemm(&A, &B, &C);
  }
  for(int i = 0; i < 10; i ++){
	//std::cout << C.getData()[i] << " ";
  }
  //std::cout << std::endl;
}

void BM_OPENCV_GEMM(benchmark::State &state) {
  long M = state.range(0), N = state.range(0), K = state.range(0);

  cv::Mat A = cv::Mat::ones(M, N, CV_64F);
  cv::Mat B = cv::Mat::ones(M, N, CV_64F);
  cv::Mat C = cv::Mat::zeros(M, N, CV_64F);

  for (auto _ : state) {
      // C += A * B;
      cv::gemm(A, B, 1.0, C, 1.0, C, 0);
  }
}

void BM_RAW_GEMM(benchmark::State &state) {
  long M = state.range(0), N = state.range(0), K = state.range(0);
  float* A = (float*)malloc(sizeof(float) * M * K);
  float* B = (float*)malloc(sizeof(float) * K * N);
  float* C = (float*)malloc(sizeof(float) * M * N);

  for (auto _ : state) {
      // C += A * B;
      for(int i = 0; i < M; i ++){
	for(int j = 0; j < N; j ++){
		for(int k = 0; k < K; k ++){
			C[i * M + j] += A[i * M + k] * B[k * K + j];
		}
	}
      }
  }
}

} // namespace

// Register benchmarking function with different arguments.
BENCHMARK(BM_GEMM)->DenseRange(512, 2048, 64);
BENCHMARK(BM_OPENCV_GEMM)->DenseRange(512, 2048, 64);
// BENCHMARK(BM_RAW_GEMM)->DenseRange(64, 512, 64);
// BENCHMARK(BM_GEMM);
//
