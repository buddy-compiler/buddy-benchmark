//===- MLIRMiniLM_5.mlir ----------------------------------------------------===//
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
// This file provides the MLIR MLIRMiniLM_5 function.
//
//===----------------------------------------------------------------------===//

#map7 = affine_map<(d0, d1, d2) -> (0, d1, d2)>
#map6 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

func.func @mlir_minilm_5() {
    %cst_7 = arith.constant 1.12837911 : f32
    %cst_8 = arith.constant -0.523018539 : f32
    %cst_9 = arith.constant 0.209741712 : f32
    %cst_10 = arith.constant 0.0258146804 : f32
    %cst_11 = arith.constant 1.12750685 : f32
    %cst_12 = arith.constant -0.364721417 : f32
    %cst_13 = arith.constant 0.118407398 : f32
    %cst_14 = arith.constant 0.0370645523 : f32
    %cst_15 = arith.constant -0.00330093061 : f32
    %cst_16 = arith.constant 0.00351961935 : f32
    %cst_17 = arith.constant -0.00141373626 : f32
    %cst_18 = arith.constant 2.53447099E-4 : f32
    %cst_19 = arith.constant -1.71048032E-5 : f32
    %cst_20 = arith.constant -0.463513821 : f32
    %cst_21 = arith.constant 0.519230127 : f32
    %cst_22 = arith.constant -0.131808966 : f32
    %cst_23 = arith.constant 0.0739796459 : f32
    %cst_24 = arith.constant -3.276070e-01 : f32
    %cst_25 = arith.constant 0.448369086 : f32
    %cst_26 = arith.constant -0.0883462652 : f32
    %cst_27 = arith.constant 0.0572442785 : f32
    %cst_28 = arith.constant -2.0606916 : f32
    %cst_29 = arith.constant 1.62705934 : f32
    %cst_30 = arith.constant -0.583389878 : f32
    %cst_31 = arith.constant 0.0821908935 : f32
    %cst_32 = arith.constant 8.000000e-01 : f32
    %cst_33 = arith.constant 2.000000e+00 : f32
    %cst_34 = arith.constant 3.750000e+00 : f32

    %cst_158 = arith.constant 1.000000e+00 : f32
    %cst_159 = arith.constant 0.000000e+00 : f32
    %cst_161 = arith.constant 1.41421354 : f32
    %cst_162 = arith.constant 5.000000e-01 : f32

%98 = tensor.empty() : tensor<1x12x1536xf32>
%101 = linalg.fill ins(%cst_159 : f32) outs(%98 : tensor<1x12x1536xf32>) -> tensor<1x12x1536xf32>

%102 = linalg.generic {indexing_maps = [#map7, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%101 : tensor<1x12x1536xf32>) outs(%98 : tensor<1x12x1536xf32>) {
    ^bb0(%in: f32, %out: f32):
      %511 = arith.divf %in, %cst_161 : f32
      %512 = arith.cmpf olt, %511, %cst_159 : f32
      %513 = arith.negf %511 : f32
      %514 = arith.select %512, %513, %511 : f32
      %515 = arith.cmpf olt, %514, %cst_32 : f32
      %516 = arith.select %515, %cst_7, %cst_11 : f32
      %517 = arith.select %515, %cst_20, %cst_24 : f32
      %518 = arith.select %515, %cst_8, %cst_12 : f32
      %519 = arith.select %515, %cst_21, %cst_25 : f32
      %520 = arith.select %515, %cst_9, %cst_13 : f32
      %521 = arith.select %515, %cst_22, %cst_26 : f32
      %522 = arith.select %515, %cst_10, %cst_14 : f32
      %523 = arith.select %515, %cst_23, %cst_27 : f32
      %524 = arith.cmpf olt, %514, %cst_33 : f32
      %525 = arith.select %524, %cst_159, %cst_15 : f32
      %526 = arith.select %524, %516, %cst_16 : f32
      %527 = arith.select %524, %517, %cst_28 : f32
      %528 = arith.select %524, %518, %cst_17 : f32
      %529 = arith.select %524, %519, %cst_29 : f32
      %530 = arith.select %524, %520, %cst_18 : f32
      %531 = arith.select %524, %521, %cst_30 : f32
      %532 = arith.select %524, %522, %cst_19 : f32
      %533 = arith.select %524, %523, %cst_31 : f32
      %534 = arith.select %524, %cst_159, %cst_158 : f32
      %535 = arith.cmpf ult, %514, %cst_34 : f32
      %536 = math.fma %514, %532, %530 : f32
      %537 = math.fma %514, %536, %528 : f32
      %538 = math.fma %514, %537, %526 : f32
      %539 = math.fma %514, %538, %525 : f32
      %540 = math.fma %514, %533, %531 : f32
      %541 = math.fma %514, %540, %529 : f32
      %542 = math.fma %514, %541, %527 : f32
      %543 = math.fma %514, %542, %cst_158 : f32
      %544 = arith.divf %539, %543 : f32
      %545 = arith.addf %534, %544 : f32
      %546 = arith.select %535, %545, %cst_158 : f32
      %547 = arith.negf %546 : f32
      %548 = arith.select %512, %547, %546 : f32
      %549 = arith.addf %548, %cst_158 : f32
      %550 = arith.mulf %549, %cst_162 : f32
      %551 = arith.mulf %in, %550 : f32
      linalg.yield %551 : f32
    } -> tensor<1x12x1536xf32>

    return
}