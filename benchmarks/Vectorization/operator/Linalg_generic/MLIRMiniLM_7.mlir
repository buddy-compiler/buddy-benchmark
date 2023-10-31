//===- MLIRMiniLM_7.mlir --------------------------------------------------===//
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
// This file provides the MLIR MLIRMiniLM_7 function.
//
//===----------------------------------------------------------------------===//

#map15 = affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func.func @mlir_minilm_7() {
  %cst = arith.constant 0.693147182 : f32
  %cst_0 = arith.constant 1.44269502 : f32
  %cst_1 = arith.constant 0.499705136 : f32
  %cst_2 = arith.constant 0.168738902 : f32
  %cst_3 = arith.constant 0.0366896503 : f32
  %cst_4 = arith.constant 1.314350e-02 : f32
  %c23_i32 = arith.constant 23 : i32
  %cst_5 = arith.constant 0x7F800000 : f32
  %cst_6 = arith.constant 1.17549435E-38 : f32
  %c127_i32 = arith.constant 127 : i32
  %c-127_i32 = arith.constant -127 : i32
  %cst_158 = arith.constant 1.000000e+00 : f32
  %cst_159 = arith.constant 0.000000e+00 : f32
  %cst_160 = arith.constant 0xFF800000 : f32
  %55 = tensor.empty() : tensor<1x12x12x12xf32>
  %1 = tensor.empty() : tensor<1x12x12x12xf32>
  %cst_7 = arith.constant 2.3: f32
  %147 = linalg.fill ins(%cst_7 : f32) outs(%1 : tensor<1x12x12x12xf32>) -> tensor<1x12x12x12xf32>
  %148 = linalg.generic {indexing_maps = [#map15, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%147 : tensor<1x12x12x12xf32>) outs(%55 : tensor<1x12x12x12xf32>) {
    ^bb0(%in: f32, %out: f32):
      %511 = arith.cmpf uno, %in, %in : f32
      %512 = arith.mulf %in, %cst_0 : f32
      %513 = math.floor %512 : f32
      %514 = arith.mulf %513, %cst : f32
      %515 = arith.subf %in, %514 : f32
      %516 = arith.mulf %515, %515 : f32
      %517 = arith.mulf %516, %516 : f32
      %518 = math.fma %cst_158, %515, %cst_158 : f32
      %519 = math.fma %cst_2, %515, %cst_1 : f32
      %520 = math.fma %cst_4, %515, %cst_3 : f32
      %521 = math.fma %519, %516, %518 : f32
      %522 = math.fma %520, %517, %521 : f32
      %523 = arith.fptosi %513 : f32 to i32
      %524 = arith.addi %523, %c127_i32 : i32
      %525 = arith.shli %524, %c23_i32 : i32
      %526 = arith.bitcast %525 : i32 to f32
      %527 = arith.mulf %522, %526 : f32
      %528 = arith.cmpi sle, %523, %c127_i32 : i32
      %529 = arith.cmpi sge, %523, %c-127_i32 : i32
      %530 = arith.cmpf oeq, %in, %cst_160 : f32
      %531 = arith.cmpf oeq, %in, %cst_5 : f32
      %532 = arith.cmpf ogt, %in, %cst_159 : f32
      %533 = arith.andi %528, %529 : i1
      %534 = arith.select %532, %cst_5, %cst_6 : f32
      %535 = arith.select %533, %527, %534 : f32
      %536 = arith.select %531, %cst_5, %535 : f32
      %537 = arith.select %530, %cst_159, %536 : f32
      %538 = arith.select %511, %in, %537 : f32
      linalg.yield %538 : f32
    } -> tensor<1x12x12x12xf32>
    return
}

