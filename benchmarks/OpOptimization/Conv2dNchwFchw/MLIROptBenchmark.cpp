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
#include <immintrin.h>
#include <benchmark/benchmark.h>
#include <cmath>
#include <iostream>
#include <cstdlib>
#include "opencv2/dnn/all_layers.hpp"

namespace {

// Declare the C interface.
extern "C" {
void _mlir_ciface_conv2d(MemRef<float, 4> *input, MemRef<float, 4> *filter,
                       MemRef<float, 4> *output);
void _mlir_ciface_conv2d_org(MemRef<float, 4> *input, MemRef<float, 4> *filter,
                       MemRef<float, 4> *output);
}

void BM_CONV(benchmark::State &state) {
  long factor = state.range(0);
  long a = 1, b = factor, c = 16 * factor, d = 16 * factor,
       e = 1, f = 32 * factor, g = 32 * factor;

  intptr_t sizesInput[4] = {a, e, c + f, d + g};
  intptr_t sizesFilter[4] = {b, e, f, g};
  intptr_t sizesOutput[4] = {a, b, c, d};

  MemRef<float, 4> input(sizesInput, 1.0);
  MemRef<float, 4> filter(sizesFilter, 1.0);
  MemRef<float, 4> output(sizesOutput, 0);

  for (auto _ : state) {
    _mlir_ciface_conv2d(&input, &filter, &output);
  }

// Test Correctness.
  float* inputAData = new float[a * e * (c + f) * (d + g)];
  for(int i = 0; i < a * e * (c + f) * (d + g); ++ i){
	  inputAData[i] = (i * std::rand()) % 5;
  }

  float* inputBData = new float[b * e * f * g];
  for(int i = 0; i < b * e * f * g; ++ i){
	  inputBData[i] = i % 13;
  }

  float* inputCData = new float[a * b * c * d];
  for(int i = 0; i < a * b * c * d; ++ i){
	  inputCData[i] = i % 5;
  }

  MemRef<float, 4> input_a(inputAData, sizesInput, 0);
  MemRef<float, 4> filter_a(inputBData, sizesFilter, 0);
  MemRef<float, 4> output_a(inputCData, sizesOutput, 0);
  _mlir_ciface_conv2d(&input_a, &filter_a, &output_a);

  MemRef<float, 4> input_b(inputAData, sizesInput, 0);
  MemRef<float, 4> filter_b(inputBData, sizesFilter, 0);
  MemRef<float, 4> output_b(inputCData, sizesOutput, 0);
  _mlir_ciface_conv2d_org(&input_b, &filter_b, &output_b);

  auto dataA = output_a.getData();
  auto dataB = output_b.getData();
  bool isOK = true;
  for(int i = 0; i < a * b * c * d; ++ i){
	  if(dataA[i] != dataB[i]) {
		isOK = false;
		break;
	  }
  }
  if(!isOK){
	std::cerr << "RESULT_ERROR" << std::endl;
	for(int i = 0; i < a; i ++){
		for(int j = 0; j < b; j ++){
			for(int k = 0; k < c; k ++){
				for(int l = 0; l < d; l ++){
					int p = i * a + j * b + k * c + l;
					// std::cerr << dataA[p];	
					if(dataA[p] != dataB[p]) {
						// std::cerr << " (" << dataB[p] << ") ";
						std::cerr << "\t[" << dataA[p] - dataB[p] << "]\t";
					} else
						std::cerr << "\t[ ]\t";
				}
				std::cerr << std::endl;
			}
		}
	}
  }
  assert(isOK);
}

} // namespace

// Register benchmarking function with different arguments.
BENCHMARK(BM_CONV)->DenseRange(1, 20, 1);
