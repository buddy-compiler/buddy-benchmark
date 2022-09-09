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
}

// OpenCV implementation here.
// REF: modules/dnn/src/layers/layers_common.simd.hpp
enum { FASCONV_BASE_VECSZ = 4 };

void fastConv( const float* weights, size_t wstep, const float* bias,
               const float* rowbuf, float* output, const int* outShape,
               int blockSize, int vecsize, int vecsize_aligned,
               const float* relu, bool initOutput )
{
    if(!cv::isAligned<32>(weights)) {
	std::cout << "NO not aligned." << std::endl;
    }

    // 因为Conv需要定义Kernel的大小，所以需要指定？
    int outCn = outShape[1];
    size_t outPlaneSize = outShape[2]*outShape[3];

    float r0 = 1.f, r1 = 1.f, r2 = 1.f;
    __m128 vr0 = _mm_set1_ps(1.f), vr1 = vr0, vr2 = vr0, z = _mm_setzero_ps();
    int CV_DECL_ALIGNED(16) maskbuf[FASCONV_BASE_VECSZ] = {0};
    int rsz = blockSize % FASCONV_BASE_VECSZ;
    for( int i = 0; i < rsz; i++ )
        maskbuf[FASCONV_BASE_VECSZ - i - 1] = -1;
    __m128 mask = _mm_loadu_ps((const float*)maskbuf); // 这里导入了一个mask，相当于将不能整除的部分都mask上去了

    // now compute dot product of the weights
    // and im2row-transformed part of the tensor
    for( int i = 0; i < outCn; i += 3 )
    {
	    // 还是一样的上来直接向量化，取3行
        const float* wptr0 = weights + i*wstep;
        const float* wptr1 = wptr0 + wstep;
        const float* wptr2 = wptr1 + wstep;
	// 输出也取3行
        float* outptr0 = output + i*outPlaneSize;
        float* outptr1 = outptr0 + outPlaneSize;
        float* outptr2 = outptr1 + outPlaneSize;
	// bias 取3个
        float bias0 = bias[i], bias1 = bias[i+1], bias2 = bias[i+2];

	// 这里做了一个溢出保护
        if( i+2 >= outCn )
        {
            wptr2 = wptr1;
            outptr2 = outptr1;
            bias2 = bias1;
            if( i+1 >= outCn ) // 同样的溢出保护 这个想法挺好
            {
                wptr2 = wptr1 = wptr0;
                outptr2 = outptr1 = outptr0;
                bias2 = bias1 = bias0;
            }
        }

        if( relu ) // 融合relu，暂时不看
        {
            r0 = relu[i]; r1 = relu[i+1]; r2 = relu[i+2];
            if( i+2 >= outCn )
            {
                r2 = r1;
                if( i+1 >= outCn )
                    r2 = r1 = r0;
            }
            vr0 = _mm_set1_ps(r0);
            vr1 = _mm_set1_ps(r1);
            vr2 = _mm_set1_ps(r2);
        }

        int j = 0;
        for( ; j < blockSize; j += FASCONV_BASE_VECSZ ) // 这玩意是传进来的，步长是变量
        {
            bool tail = false;
            if (j + FASCONV_BASE_VECSZ > blockSize) // 同样是保护，这里选择拉回来
            {
                if (j == 0)
                    break;
                j = blockSize - FASCONV_BASE_VECSZ;
                tail = true;
            }
            int k = 0;
            const float* rptr = rowbuf + j*vecsize_aligned; // 传进来的

            __m256 vs00 = _mm256_setzero_ps(), vs01 = _mm256_setzero_ps(),
                   vs02 = _mm256_setzero_ps(), vs03 = _mm256_setzero_ps(),
                   vs10 = _mm256_setzero_ps(), vs11 = _mm256_setzero_ps(),
                   vs12 = _mm256_setzero_ps(), vs13 = _mm256_setzero_ps(),
                   vs20 = _mm256_setzero_ps(), vs21 = _mm256_setzero_ps(),
                   vs22 = _mm256_setzero_ps(), vs23 = _mm256_setzero_ps();

#if CV_AVX512_SKX // AVX512VL is necessary to avoid register spilling
            if (vecsize >= 32)
            {
                __m512 vs00_5 = _mm512_setzero_ps(), vs01_5 = _mm512_setzero_ps(),
                       vs02_5 = _mm512_setzero_ps(), vs03_5 = _mm512_setzero_ps(),
                       vs10_5 = _mm512_setzero_ps(), vs11_5 = _mm512_setzero_ps(),
                       vs12_5 = _mm512_setzero_ps(), vs13_5 = _mm512_setzero_ps(),
                       vs20_5 = _mm512_setzero_ps(), vs21_5 = _mm512_setzero_ps(),
                       vs22_5 = _mm512_setzero_ps(), vs23_5 = _mm512_setzero_ps();

                for (; k <= vecsize - 16; k += 16, rptr += 16)
                {
                    __m512 w0 = _mm512_loadu_ps(wptr0 + k);
                    __m512 w1 = _mm512_loadu_ps(wptr1 + k);
                    __m512 w2 = _mm512_loadu_ps(wptr2 + k);
                    __m512 r0 = _mm512_loadu_ps(rptr);

                    vs00_5 = _mm512_fmadd_ps(w0, r0, vs00_5);
                    vs10_5 = _mm512_fmadd_ps(w1, r0, vs10_5);
                    vs20_5 = _mm512_fmadd_ps(w2, r0, vs20_5);

                    r0 = _mm512_loadu_ps(rptr + vecsize_aligned);
                    vs01_5 = _mm512_fmadd_ps(w0, r0, vs01_5);
                    vs11_5 = _mm512_fmadd_ps(w1, r0, vs11_5);
                    vs21_5 = _mm512_fmadd_ps(w2, r0, vs21_5);

                    r0 = _mm512_loadu_ps(rptr + vecsize_aligned*2);
                    vs02_5 = _mm512_fmadd_ps(w0, r0, vs02_5);
                    vs12_5 = _mm512_fmadd_ps(w1, r0, vs12_5);
                    vs22_5 = _mm512_fmadd_ps(w2, r0, vs22_5);

                    r0 = _mm512_loadu_ps(rptr + vecsize_aligned*3);
                    vs03_5 = _mm512_fmadd_ps(w0, r0, vs03_5);
                    vs13_5 = _mm512_fmadd_ps(w1, r0, vs13_5);
                    vs23_5 = _mm512_fmadd_ps(w2, r0, vs23_5);
                }
                /*
                 * now fold the 512 bit accumulator vectors into 256 bit vectors so that the AVX2 code can finish
                 * the tail of the vector
                 */
                vs00 = _mm256_add_ps( _mm512_extractf32x8_ps(vs00_5, 0), _mm512_extractf32x8_ps(vs00_5, 1));
                vs10 = _mm256_add_ps( _mm512_extractf32x8_ps(vs10_5, 0), _mm512_extractf32x8_ps(vs10_5, 1));
                vs20 = _mm256_add_ps( _mm512_extractf32x8_ps(vs20_5, 0), _mm512_extractf32x8_ps(vs20_5, 1));

                vs01 = _mm256_add_ps( _mm512_extractf32x8_ps(vs01_5, 0), _mm512_extractf32x8_ps(vs01_5, 1));
                vs11 = _mm256_add_ps( _mm512_extractf32x8_ps(vs11_5, 0), _mm512_extractf32x8_ps(vs11_5, 1));
                vs21 = _mm256_add_ps( _mm512_extractf32x8_ps(vs21_5, 0), _mm512_extractf32x8_ps(vs21_5, 1));

                vs02 = _mm256_add_ps( _mm512_extractf32x8_ps(vs02_5, 0), _mm512_extractf32x8_ps(vs02_5, 1));
                vs12 = _mm256_add_ps( _mm512_extractf32x8_ps(vs12_5, 0), _mm512_extractf32x8_ps(vs12_5, 1));
                vs22 = _mm256_add_ps( _mm512_extractf32x8_ps(vs22_5, 0), _mm512_extractf32x8_ps(vs22_5, 1));

                vs03 = _mm256_add_ps( _mm512_extractf32x8_ps(vs03_5, 0), _mm512_extractf32x8_ps(vs03_5, 1));
                vs13 = _mm256_add_ps( _mm512_extractf32x8_ps(vs13_5, 0), _mm512_extractf32x8_ps(vs13_5, 1));
                vs23 = _mm256_add_ps( _mm512_extractf32x8_ps(vs23_5, 0), _mm512_extractf32x8_ps(vs23_5, 1));
            }
#endif

            for (; k < vecsize; k += 8, rptr += 8 )
            {
                __m256 w0 = _mm256_load_ps(wptr0 + k); // 每次往出读8个  现在w* -> 3 x 8
                __m256 w1 = _mm256_load_ps(wptr1 + k);
                __m256 w2 = _mm256_load_ps(wptr2 + k);
                __m256 r0 = _mm256_load_ps(rptr); 

                vs00 = _mm256_fmadd_ps(w0, r0, vs00);
                vs10 = _mm256_fmadd_ps(w1, r0, vs10);
                vs20 = _mm256_fmadd_ps(w2, r0, vs20);

                r0 = _mm256_load_ps(rptr + vecsize_aligned);
                vs01 = _mm256_fmadd_ps(w0, r0, vs01);
                vs11 = _mm256_fmadd_ps(w1, r0, vs11);
                vs21 = _mm256_fmadd_ps(w2, r0, vs21);

                r0 = _mm256_load_ps(rptr + vecsize_aligned*2);
                vs02 = _mm256_fmadd_ps(w0, r0, vs02);
                vs12 = _mm256_fmadd_ps(w1, r0, vs12);
                vs22 = _mm256_fmadd_ps(w2, r0, vs22);

                r0 = _mm256_load_ps(rptr + vecsize_aligned*3);
                vs03 = _mm256_fmadd_ps(w0, r0, vs03);
                vs13 = _mm256_fmadd_ps(w1, r0, vs13);
                vs23 = _mm256_fmadd_ps(w2, r0, vs23);
            }

            __m256 t0 = _mm256_hadd_ps(_mm256_hadd_ps(vs00, vs01), _mm256_hadd_ps(vs02, vs03)); // 再将结果收缩2次(相邻两个两两相加），这样4个vec的内容就放在一个里面了
            __m256 t1 = _mm256_hadd_ps(_mm256_hadd_ps(vs10, vs11), _mm256_hadd_ps(vs12, vs13));
            __m256 t2 = _mm256_hadd_ps(_mm256_hadd_ps(vs20, vs21), _mm256_hadd_ps(vs22, vs23));

            t0 = _mm256_add_ps(t0, _mm256_permute2f128_ps(t0, t0, 1)); // 将t0当中的两个交换，然后相加
            t1 = _mm256_add_ps(t1, _mm256_permute2f128_ps(t1, t1, 1));
            t2 = _mm256_add_ps(t2, _mm256_permute2f128_ps(t2, t2, 1));

            __m128 s0, s1, s2;

            if( initOutput )
            {
                s0 = _mm_set1_ps(bias0);
                s1 = _mm_set1_ps(bias1);
                s2 = _mm_set1_ps(bias2);
            }
            else
            {
                s0 = _mm_loadu_ps(outptr0 + j);
                s1 = _mm_loadu_ps(outptr1 + j);
                s2 = _mm_loadu_ps(outptr2 + j);
            }

            s0 = _mm_add_ps(s0, _mm256_castps256_ps128(t0)); // 最后再做一个自相加
            s1 = _mm_add_ps(s1, _mm256_castps256_ps128(t1));
            s2 = _mm_add_ps(s2, _mm256_castps256_ps128(t2));

            if( relu )
            {
                __m128 m0 = _mm_cmp_ps(s0, z, _CMP_GT_OS);
                __m128 m1 = _mm_cmp_ps(s1, z, _CMP_GT_OS);
                __m128 m2 = _mm_cmp_ps(s2, z, _CMP_GT_OS);
                s0 = _mm_blendv_ps(_mm_mul_ps(s0, vr0), s0, m0);
                s1 = _mm_blendv_ps(_mm_mul_ps(s1, vr1), s1, m1);
                s2 = _mm_blendv_ps(_mm_mul_ps(s2, vr2), s2, m2);
            }

            if( tail )
            {
                s0 = _mm_blendv_ps(_mm_loadu_ps(outptr0 + j), s0, mask);
                s1 = _mm_blendv_ps(_mm_loadu_ps(outptr1 + j), s1, mask);
                s2 = _mm_blendv_ps(_mm_loadu_ps(outptr2 + j), s2, mask);
            }

            _mm_storeu_ps(outptr0 + j, s0);
            _mm_storeu_ps(outptr1 + j, s1);
            _mm_storeu_ps(outptr2 + j, s2);
        }

        for( ; j <= blockSize - 2; j += 2 )
        {
            const float* rptr0 = rowbuf + j*vecsize_aligned;
            const float* rptr1 = rowbuf + (j+1)*vecsize_aligned;
            float s00, s01, s10, s11, s20, s21;

            if( initOutput )
            {
                s00 = s01 = bias0;
                s10 = s11 = bias1;
                s20 = s21 = bias2;
            }
            else
            {
                s00 = outptr0[j]; s01 = outptr0[j+1];
                s10 = outptr1[j]; s11 = outptr1[j+1];
                s20 = outptr2[j]; s21 = outptr2[j+1];
            }

            for( int k = 0; k < vecsize; k++ )
            {
                float w0 = wptr0[k], w1 = wptr1[k], w2 = wptr2[k];
                float r = rptr0[k];
                s00 += w0*r; s10 += w1*r; s20 += w2*r;
                r = rptr1[k];
                s01 += w0*r; s11 += w1*r; s21 += w2*r;
            }

            if( relu )
            {
                s00 = s00 > 0.f ? s00 : s00*r0;
                s01 = s01 > 0.f ? s01 : s01*r0;
                s10 = s10 > 0.f ? s10 : s10*r1;
                s11 = s11 > 0.f ? s11 : s11*r1;
                s20 = s20 > 0.f ? s20 : s20*r2;
                s21 = s21 > 0.f ? s21 : s21*r2;
            }

            outptr0[j] = s00;
            outptr0[j+1] = s01;
            outptr1[j] = s10;
            outptr1[j+1] = s11;
            outptr2[j] = s20;
            outptr2[j+1] = s21;
        }

        for( ; j < blockSize; j++ )
        {
            const float* rptr0 = rowbuf + j*vecsize_aligned;
            float s00, s10, s20;

            if( initOutput )
            {
                s00 = bias0;
                s10 = bias1;
                s20 = bias2;
            }
            else
            {
                s00 = outptr0[j];
                s10 = outptr1[j];
                s20 = outptr2[j];
            }

            for( int k = 0; k < vecsize; k++ )
            {
                float w0 = wptr0[k], w1 = wptr1[k], w2 = wptr2[k];
                float r = rptr0[k];
                s00 += w0*r; s10 += w1*r; s20 += w2*r;
            }

            if( relu )
            {
                s00 = s00 > 0.f ? s00 : s00*r0;
                s10 = s10 > 0.f ? s10 : s10*r1;
                s20 = s20 > 0.f ? s20 : s20*r2;
            }

            outptr0[j] = s00;
            outptr1[j] = s10;
            outptr2[j] = s20;
        }
    }
    _mm256_zeroupper();
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
}

void BM_OPENCV_CONV(benchmark::State &state) {
  size_t factor = state.range(0);
  std::cout << "factor: " << factor << std::endl;

  int a = 1, b = factor, c = 16 * factor, d = 16 * factor,
       e = 1, f = 32 * factor, g = 32 * factor;

  float* rowbuf = (float*)malloc(sizeof(float) * a * e * (c + f) * (d + g));
  float* bias = (float*)malloc(sizeof(float) * a * b * c * d);
  // aligned to 32 is needed.
  float* weight = (float*)aligned_alloc(sizeof(float) * b * e * f * g, 32);
  size_t wstep = g;
  float* output = (float*)malloc(sizeof(float) * a * b * c * d);
  int outShape[] = {0, b * e, f, g}; // ?
  int blockSize = d + g; // ?
  int vecsize = 16;
  int vecsize_aligned = 16;
  float* relu = nullptr;
  bool initOutput = false;

  for (auto _ : state) {
    fastConv(weight, wstep, bias, rowbuf, output, outShape, blockSize, vecsize, vecsize_aligned, relu, initOutput);
  }

  free(output);
  free(rowbuf);
  free(bias);
  free(weight);
}

} // namespace

// Register benchmarking function with different arguments.
BENCHMARK(BM_CONV)->DenseRange(1, 20, 1);
// BENCHMARK(BM_OPENCV_CONV)->DenseRange(1, 10, 1);
