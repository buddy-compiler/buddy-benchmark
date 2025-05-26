//===- ExoMatmul.c --------------------------------------------------------===//
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
// This file implements Exo-lang Matmul kernel.
// The kernels are generated from exo-lang python script.
//
//===----------------------------------------------------------------------===//

#include "ExoUtils.h"
#include "gemmini.h"

// clang-format off
void _exo_matmul_4(const float* scale, bool act, const int8_t* A, const int8_t* B, int8_t* C) {
  gemmini_extended_config_st((256), (act), (scale)[0]);
  gemmini_extended_config_ex(WS, 0, 0, 1, 0, 0);
  gemmini_extended3_config_ld((256), 1.0f, 0, 2);
  gemmini_extended3_config_ld((64), 1.0f, 0, 1);
  gemmini_extended3_config_ld(0, 1.0f, 0, 0);

  int8_t *a = (int8_t*) ((uint64_t)gemm_malloc (16 * 16 * 4 * 1 * 196 * sizeof(int8_t)));
  int8_t *b = (int8_t*) ((uint64_t)gemm_malloc (16 * 16 * 4 * 4 * 1 * 4 * sizeof(int8_t)));
  int32_t *res = (int32_t*) ((uint32_t)gemm_acc_malloc (16 * 16 * 4 * 4 * sizeof(int32_t)));
  for (int_fast32_t io = 0; io < 4; io++) {
    for (int_fast32_t i = 0; i < 196; i++) {
      for (int_fast32_t j = 0; j < 4; j++) {
        gemmini_extended_mvin( 0, ((uint64_t) &*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)res)) + ((j) * (1024))/16))),(16), (16) );
        gemmini_extended_mvin( 0, ((uint64_t) &*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)res)) + ((j) * (1024) + 256)/16))),(16), (16) );
        gemmini_extended_mvin( 0, ((uint64_t) &*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)res)) + ((j) * (1024) + (2) * (256))/16))),(16), (16) );
        gemmini_extended_mvin( 0, ((uint64_t) &*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)res)) + ((j) * (1024) + (3) * (256))/16))),(16), (16) );
        if (j == 0) {
          gemmini_extended_mvin2( &A[(16 * i + 3136 * io) * (64)], ((uint64_t) &*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)a)) + ((i) * (1024))/16))), 16*(4), (16) );
        }
        if (io == 0) {
          if (i == 0) {
            gemmini_extended_mvin3( &B[64 * j], ((uint64_t) &*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)b)) + ((j) * (4096))/16))), 16*(4), (16) );
          }
        }
        if (io == 0) {
          if (i == 0) {
            gemmini_extended_mvin3( &B[(16) * (256) + 64 * j], ((uint64_t) &*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)b)) + ((j) * (4096) + 1024)/16))), 16*(4), (16) );
          }
        }
        if (io == 0) {
          if (i == 0) {
            gemmini_extended_mvin3( &B[(32) * (256) + 64 * j], ((uint64_t) &*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)b)) + ((j) * (4096) + (2) * (1024))/16))), 16*(4), (16) );
          }
        }
        if (io == 0) {
          if (i == 0) {
            gemmini_extended_mvin3( &B[(48) * (256) + 64 * j], ((uint64_t) &*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)b)) + ((j) * (4096) + (3) * (1024))/16))), 16*(4), (16) );
          }
        }
        gemmini_extended_preload((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)b)) + ((j) * (4096))/16))), (uint32_t)(&*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)res)) + ((j) * (1024))/16))) | 0x40000000, (16), (16), (16), (16));
        gemmini_extended_compute_preloaded((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)a)) + ((i) * (1024))/16))), ~((uint32_t)0), (16), (16), 16, 16);
        gemmini_extended_preload((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)b)) + ((j) * (4096) + 256)/16))), (uint32_t)(&*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)res)) + ((j) * (1024) + 256)/16))) | 0x40000000, (16), (16), (16), (16));
        gemmini_extended_compute_preloaded((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)a)) + ((i) * (1024))/16))), ~((uint32_t)0), (16), (16), 16, 16);
        gemmini_extended_preload((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)b)) + ((j) * (4096) + (2) * (256))/16))), (uint32_t)(&*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)res)) + ((j) * (1024) + (2) * (256))/16))) | 0x40000000, (16), (16), (16), (16));
        gemmini_extended_compute_preloaded((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)a)) + ((i) * (1024))/16))), ~((uint32_t)0), (16), (16), 16, 16);
        gemmini_extended_preload((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)b)) + ((j) * (4096) + (3) * (256))/16))), (uint32_t)(&*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)res)) + ((j) * (1024) + (3) * (256))/16))) | 0x40000000, (16), (16), (16), (16));
        gemmini_extended_compute_preloaded((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)a)) + ((i) * (1024))/16))), ~((uint32_t)0), (16), (16), 16, 16);
        gemmini_extended_preload((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)b)) + ((j) * (4096) + 1024)/16))), (uint32_t)(&*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)res)) + ((j) * (1024))/16))) | 0x40000000, (16), (16), (16), (16));
        gemmini_extended_compute_preloaded((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)a)) + ((i) * (1024) + 256)/16))), ~((uint32_t)0), (16), (16), 16, 16);
        gemmini_extended_preload((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)b)) + ((j) * (4096) + 1024 + 256)/16))), (uint32_t)(&*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)res)) + ((j) * (1024) + 256)/16))) | 0x40000000, (16), (16), (16), (16));
        gemmini_extended_compute_preloaded((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)a)) + ((i) * (1024) + 256)/16))), ~((uint32_t)0), (16), (16), 16, 16);
        gemmini_extended_preload((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)b)) + ((j) * (4096) + 1024 + (2) * (256))/16))), (uint32_t)(&*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)res)) + ((j) * (1024) + (2) * (256))/16))) | 0x40000000, (16), (16), (16), (16));
        gemmini_extended_compute_preloaded((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)a)) + ((i) * (1024) + 256)/16))), ~((uint32_t)0), (16), (16), 16, 16);
        gemmini_extended_preload((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)b)) + ((j) * (4096) + 1024 + (3) * (256))/16))), (uint32_t)(&*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)res)) + ((j) * (1024) + (3) * (256))/16))) | 0x40000000, (16), (16), (16), (16));
        gemmini_extended_compute_preloaded((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)a)) + ((i) * (1024) + 256)/16))), ~((uint32_t)0), (16), (16), 16, 16);
        gemmini_extended_preload((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)b)) + ((j) * (4096) + (2) * (1024))/16))), (uint32_t)(&*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)res)) + ((j) * (1024))/16))) | 0x40000000, (16), (16), (16), (16));
        gemmini_extended_compute_preloaded((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)a)) + ((i) * (1024) + (2) * (256))/16))), ~((uint32_t)0), (16), (16), 16, 16);
        gemmini_extended_preload((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)b)) + ((j) * (4096) + (2) * (1024) + 256)/16))), (uint32_t)(&*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)res)) + ((j) * (1024) + 256)/16))) | 0x40000000, (16), (16), (16), (16));
        gemmini_extended_compute_preloaded((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)a)) + ((i) * (1024) + (2) * (256))/16))), ~((uint32_t)0), (16), (16), 16, 16);
        gemmini_extended_preload((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)b)) + ((j) * (4096) + (2) * (1024) + (2) * (256))/16))), (uint32_t)(&*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)res)) + ((j) * (1024) + (2) * (256))/16))) | 0x40000000, (16), (16), (16), (16));
        gemmini_extended_compute_preloaded((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)a)) + ((i) * (1024) + (2) * (256))/16))), ~((uint32_t)0), (16), (16), 16, 16);
        gemmini_extended_preload((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)b)) + ((j) * (4096) + (2) * (1024) + (3) * (256))/16))), (uint32_t)(&*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)res)) + ((j) * (1024) + (3) * (256))/16))) | 0x40000000, (16), (16), (16), (16));
        gemmini_extended_compute_preloaded((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)a)) + ((i) * (1024) + (2) * (256))/16))), ~((uint32_t)0), (16), (16), 16, 16);
        gemmini_extended_preload((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)b)) + ((j) * (4096) + (3) * (1024))/16))), (uint32_t)(&*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)res)) + ((j) * (1024))/16))) | 0x40000000, (16), (16), (16), (16));
        gemmini_extended_compute_preloaded((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)a)) + ((i) * (1024) + (3) * (256))/16))), ~((uint32_t)0), (16), (16), 16, 16);
        gemmini_extended_preload((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)b)) + ((j) * (4096) + (3) * (1024) + 256)/16))), (uint32_t)(&*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)res)) + ((j) * (1024) + 256)/16))) | 0x40000000, (16), (16), (16), (16));
        gemmini_extended_compute_preloaded((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)a)) + ((i) * (1024) + (3) * (256))/16))), ~((uint32_t)0), (16), (16), 16, 16);
        gemmini_extended_preload((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)b)) + ((j) * (4096) + (3) * (1024) + (2) * (256))/16))), (uint32_t)(&*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)res)) + ((j) * (1024) + (2) * (256))/16))) | 0x40000000, (16), (16), (16), (16));
        gemmini_extended_compute_preloaded((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)a)) + ((i) * (1024) + (3) * (256))/16))), ~((uint32_t)0), (16), (16), 16, 16);
        gemmini_extended_preload((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)b)) + ((j) * (4096) + (3) * (1024) + (3) * (256))/16))), (uint32_t)(&*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)res)) + ((j) * (1024) + (3) * (256))/16))) | 0x40000000, (16), (16), (16), (16));
        gemmini_extended_compute_preloaded((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)a)) + ((i) * (1024) + (3) * (256))/16))), ~((uint32_t)0), (16), (16), 16, 16);
        gemmini_extended_mvout( ((uint64_t) &C[(16 * i + 3136 * io) * (256) + 64 * j]), (uint32_t) &*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)res)) + ((j) * (1024))/16)), (16), (16) );
        gemmini_extended_mvout( ((uint64_t) &C[(16 * i + 3136 * io) * (256) + 16 + 64 * j]), (uint32_t) &*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)res)) + ((j) * (1024) + 256)/16)), (16), (16) );
        gemmini_extended_mvout( ((uint64_t) &C[(16 * i + 3136 * io) * (256) + 32 + 64 * j]), (uint32_t) &*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)res)) + ((j) * (1024) + (2) * (256))/16)), (16), (16) );
        gemmini_extended_mvout( ((uint64_t) &C[(16 * i + 3136 * io) * (256) + 48 + 64 * j]), (uint32_t) &*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)res)) + ((j) * (1024) + (3) * (256))/16)), (16), (16) );
      }
    }
  }
  gemm_acc_free((uint32_t)(res));
  gemm_free((uint64_t)(b));
  gemm_free((uint64_t)(a));
}
// clang-format on
