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
// conv_3(
//     output : i8[4, 56, 56, 64] @DRAM,
//     bias : i32[1, 64] @DRAM,
//     inp : i8[4, 58, 58, 64] @DRAM,
//     weights : i8[3, 3, 64, 64] @DRAM,
//     act : bool,
//     scale : f32 @DRAM
// )
void _exo_conv_3( int8_t* output, const int32_t* bias, const int8_t* inp, const int8_t* weights, bool act, const float* scale ) {
  gemmini_extended_config_st((64), (act), (scale)[0]);
  gemmini_extended_config_ex(WS, 0, 0, 1, 0, 0);
  gemmini_extended3_config_ld((64), 1.0f, 0, 1);
  gemmini_extended3_config_ld(0, 1.0f, 0, 0);
  int8_t *i_s = (int8_t*) ((uint64_t)gemm_malloc (16 * 16 * 4 * 3 * 30 * sizeof(int8_t)));
  int8_t *i_s_1 = (int8_t*) ((uint64_t)gemm_malloc (16 * 8 * 4 * 3 * 30 * sizeof(int8_t)));
  int8_t *w_s = (int8_t*) ((uint64_t)gemm_malloc (16 * 16 * 4 * 4 * 3 * 3 * sizeof(int8_t)));
  int8_t *w_s_1 = (int8_t*) ((uint64_t)gemm_malloc (16 * 16 * 4 * 4 * 3 * 3 * sizeof(int8_t)));
  int32_t *res = (int32_t*) ((uint32_t)gemm_acc_malloc (16 * 16 * 4 * sizeof(int32_t)));
  int32_t *res_1 = (int32_t*) ((uint32_t)gemm_acc_malloc (16 * 9 * 4 * sizeof(int32_t)));
  for (int_fast32_t b = 0; b < 4; b++) {
    for (int_fast32_t ocol_o = 0; ocol_o < 3; ocol_o++) {
      for (int_fast32_t orow_o = 0; orow_o < 2; orow_o++) {
        for (int_fast32_t orow_io = 0; orow_io < 4; orow_io++) {
          for (int_fast32_t orow_ii = 0; orow_ii < 7; orow_ii++) {
            gemmini_extended_mvin( ((uint64_t) &bias[0]), ((uint32_t) &*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)res)) + (0)/16))), 16, (16) );
            gemmini_extended_mvin( ((uint64_t) &bias[16]), ((uint32_t) &*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)res)) + (256)/16))), 16, (16) );
            gemmini_extended_mvin( ((uint64_t) &bias[32]), ((uint32_t) &*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)res)) + ((2) * (256))/16))), 16, (16) );
            gemmini_extended_mvin( ((uint64_t) &bias[48]), ((uint32_t) &*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)res)) + ((3) * (256))/16))), 16, (16) );
            for (int_fast32_t krow = 0; krow < 3; krow++) {
              for (int_fast32_t kcol = 0; kcol < 3; kcol++) {
                if (ocol_o == 0) {
                  if (b == 0) {
                    if (orow_o == 0) {
                      if (orow_ii + 7 * orow_io == 0) {
                        for (int_fast32_t kch_o = 0; kch_o < 4; kch_o++) {
                          gemmini_extended_mvin2( &weights[(krow) * (12288) + (kcol) * (4096) + (16 * kch_o) * (64)], ((uint64_t) &*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)w_s)) + ((krow) * (12288) + (kcol) * (4096) + (kch_o) * (1024))/16))), 16*(4), (16) );
                        }
                      }
                    }
                  }
                }
                if (orow_ii + 7 * orow_io == 0 || krow == 2) {
                  gemmini_extended4_config_ld(((struct exo_win_2i8c){ &inp[(b) * (215296) + (krow + orow_ii + 7 * orow_io + 28 * orow_o) * (3712) + (kcol + 16 * ocol_o) * (64)], { 64, 1 } }).strides[0]*1, 1.0f, 0, (16), 2);
                  gemmini_extended_mvin3( &inp[(b) * (215296) + (krow + orow_ii + 7 * orow_io + 28 * orow_o) * (3712) + (kcol + 16 * ocol_o) * (64)], ((uint64_t) &*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)i_s)) + ((krow + orow_ii + 7 * orow_io) * (3072) + (kcol) * (1024))/16))), 16*(4), (16) );
                }
                for (int_fast32_t kch_o = 0; kch_o < 4; kch_o++) {
                  gemmini_extended_preload((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)w_s)) + ((krow) * (12288) + (kcol) * (4096) + (kch_o) * (1024))/16))), (uint32_t)(&*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)res)) + (0)/16))) | 0x40000000, (16), (16), (16), (16));
                  gemmini_extended_compute_preloaded((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)i_s)) + ((krow + orow_ii + 7 * orow_io) * (3072) + (kcol) * (1024) + (kch_o) * (256))/16))), ~((uint32_t)0), (16), (16), 16, 16);
                  gemmini_extended_preload((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)w_s)) + ((krow) * (12288) + (kcol) * (4096) + (kch_o) * (1024) + 256)/16))), (uint32_t)(&*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)res)) + (256)/16))) | 0x40000000, (16), (16), (16), (16));
                  gemmini_extended_compute_preloaded((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)i_s)) + ((krow + orow_ii + 7 * orow_io) * (3072) + (kcol) * (1024) + (kch_o) * (256))/16))), ~((uint32_t)0), (16), (16), 16, 16);
                  gemmini_extended_preload((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)w_s)) + ((krow) * (12288) + (kcol) * (4096) + (kch_o) * (1024) + (2) * (256))/16))), (uint32_t)(&*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)res)) + ((2) * (256))/16))) | 0x40000000, (16), (16), (16), (16));
                  gemmini_extended_compute_preloaded((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)i_s)) + ((krow + orow_ii + 7 * orow_io) * (3072) + (kcol) * (1024) + (kch_o) * (256))/16))), ~((uint32_t)0), (16), (16), 16, 16);
                  gemmini_extended_preload((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)w_s)) + ((krow) * (12288) + (kcol) * (4096) + (kch_o) * (1024) + (3) * (256))/16))), (uint32_t)(&*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)res)) + ((3) * (256))/16))) | 0x40000000, (16), (16), (16), (16));
                  gemmini_extended_compute_preloaded((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)i_s)) + ((krow + orow_ii + 7 * orow_io) * (3072) + (kcol) * (1024) + (kch_o) * (256))/16))), ~((uint32_t)0), (16), (16), 16, 16);
                }
              }
            }
            gemmini_extended_mvout( ((uint64_t) &output[(b) * (200704) + (orow_ii + 7 * orow_io + 28 * orow_o) * (3584) + (16 * ocol_o) * (64)]), (uint32_t) &*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)res)) + (0)/16)), (16), (16) );
            gemmini_extended_mvout( ((uint64_t) &output[(b) * (200704) + (orow_ii + 7 * orow_io + 28 * orow_o) * (3584) + (16 * ocol_o) * (64) + 16]), (uint32_t) &*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)res)) + (256)/16)), (16), (16) );
            gemmini_extended_mvout( ((uint64_t) &output[(b) * (200704) + (orow_ii + 7 * orow_io + 28 * orow_o) * (3584) + (16 * ocol_o) * (64) + 32]), (uint32_t) &*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)res)) + ((2) * (256))/16)), (16), (16) );
            gemmini_extended_mvout( ((uint64_t) &output[(b) * (200704) + (orow_ii + 7 * orow_io + 28 * orow_o) * (3584) + (16 * ocol_o) * (64) + 48]), (uint32_t) &*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)res)) + ((3) * (256))/16)), (16), (16) );
          }
        }
      }
    }
    for (int_fast32_t orow_o = 0; orow_o < 2; orow_o++) {
      for (int_fast32_t orow_io = 0; orow_io < 4; orow_io++) {
        for (int_fast32_t orow_ii = 0; orow_ii < 7; orow_ii++) {
          gemmini_extended_mvin( ((uint64_t) &bias[0]), ((uint32_t) &*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)res_1)) + (0)/16))), 16, (8) );
          gemmini_extended_mvin( ((uint64_t) &bias[16]), ((uint32_t) &*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)res_1)) + (144)/16))), 16, (8) );
          gemmini_extended_mvin( ((uint64_t) &bias[32]), ((uint32_t) &*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)res_1)) + ((2) * (144))/16))), 16, (8) );
          gemmini_extended_mvin( ((uint64_t) &bias[48]), ((uint32_t) &*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)res_1)) + ((3) * (144))/16))), 16, (8) );
          for (int_fast32_t krow = 0; krow < 3; krow++) {
            for (int_fast32_t kcol = 0; kcol < 3; kcol++) {
              if (b == 0) {
                if (orow_o == 0) {
                  if (orow_ii + 7 * orow_io == 0) {
                    for (int_fast32_t kch_o = 0; kch_o < 4; kch_o++) {
                      gemmini_extended_mvin2( &weights[(krow) * (12288) + (kcol) * (4096) + (16 * kch_o) * (64)], ((uint64_t) &*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)w_s_1)) + ((krow) * (12288) + (kcol) * (4096) + (kch_o) * (1024))/16))), 16*(4), (16) );
                    }
                  }
                }
              }
              if (orow_ii + 7 * orow_io == 0 || krow == 2) {
                gemmini_extended4_config_ld(((struct exo_win_2i8c){ &inp[(b) * (215296) + (krow + orow_ii + 7 * orow_io + 28 * orow_o) * (3712) + (48 + kcol) * (64)], { 64, 1 } }).strides[0]*1, 1.0f, 0, (8), 2);
                gemmini_extended_mvin3( &inp[(b) * (215296) + (krow + orow_ii + 7 * orow_io + 28 * orow_o) * (3712) + (48 + kcol) * (64)], ((uint64_t) &*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)i_s_1)) + ((krow + orow_ii + 7 * orow_io) * (1536) + (kcol) * (512))/16))), 16*(4), (8) );
              }
              for (int_fast32_t kch_o = 0; kch_o < 4; kch_o++) {
                gemmini_extended_preload((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)w_s_1)) + ((krow) * (12288) + (kcol) * (4096) + (kch_o) * (1024))/16))), (uint32_t)(&*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)res_1)) + (0)/16))) | 0x40000000, (16), (16), (16), (8));
                gemmini_extended_compute_preloaded((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)i_s_1)) + ((krow + orow_ii + 7 * orow_io) * (1536) + (kcol) * (512) + (kch_o) * (128))/16))), ~((uint32_t)0), (16), (8), 16, 16);
                gemmini_extended_preload((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)w_s_1)) + ((krow) * (12288) + (kcol) * (4096) + (kch_o) * (1024) + 256)/16))), (uint32_t)(&*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)res_1)) + (144)/16))) | 0x40000000, (16), (16), (16), (8));
                gemmini_extended_compute_preloaded((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)i_s_1)) + ((krow + orow_ii + 7 * orow_io) * (1536) + (kcol) * (512) + (kch_o) * (128))/16))), ~((uint32_t)0), (16), (8), 16, 16);
                gemmini_extended_preload((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)w_s_1)) + ((krow) * (12288) + (kcol) * (4096) + (kch_o) * (1024) + (2) * (256))/16))), (uint32_t)(&*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)res_1)) + ((2) * (144))/16))) | 0x40000000, (16), (16), (16), (8));
                gemmini_extended_compute_preloaded((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)i_s_1)) + ((krow + orow_ii + 7 * orow_io) * (1536) + (kcol) * (512) + (kch_o) * (128))/16))), ~((uint32_t)0), (16), (8), 16, 16);
                gemmini_extended_preload((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)w_s_1)) + ((krow) * (12288) + (kcol) * (4096) + (kch_o) * (1024) + (3) * (256))/16))), (uint32_t)(&*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)res_1)) + ((3) * (144))/16))) | 0x40000000, (16), (16), (16), (8));
                gemmini_extended_compute_preloaded((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)i_s_1)) + ((krow + orow_ii + 7 * orow_io) * (1536) + (kcol) * (512) + (kch_o) * (128))/16))), ~((uint32_t)0), (16), (8), 16, 16);
              }
            }
          }
          gemmini_extended_mvout( ((uint64_t) &output[(b) * (200704) + (orow_ii + 7 * orow_io + 28 * orow_o) * (3584) + (48) * (64)]), (uint32_t) &*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)res_1)) + (0)/16)), (16), (8) );
          gemmini_extended_mvout( ((uint64_t) &output[(b) * (200704) + (orow_ii + 7 * orow_io + 28 * orow_o) * (3584) + (48) * (64) + 16]), (uint32_t) &*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)res_1)) + (144)/16)), (16), (8) );
          gemmini_extended_mvout( ((uint64_t) &output[(b) * (200704) + (orow_ii + 7 * orow_io + 28 * orow_o) * (3584) + (48) * (64) + 32]), (uint32_t) &*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)res_1)) + ((2) * (144))/16)), (16), (8) );
          gemmini_extended_mvout( ((uint64_t) &output[(b) * (200704) + (orow_ii + 7 * orow_io + 28 * orow_o) * (3584) + (48) * (64) + 48]), (uint32_t) &*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)res_1)) + ((3) * (144))/16)), (16), (8) );
        }
      }
    }
  }
  gemm_acc_free((uint32_t)(res_1));
  gemm_acc_free((uint32_t)(res));
  gemm_free((uint64_t)(w_s_1));
  gemm_free((uint64_t)(w_s));
  gemm_free((uint64_t)(i_s_1));
  gemm_free((uint64_t)(i_s));
}
// clang-format on
