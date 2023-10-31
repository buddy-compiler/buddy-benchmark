//===- MLIRMiniLM_4.mlir --------------------------------------------------===//
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
// This file provides the MLIR MLIRMiniLM_4 function.
//
//===----------------------------------------------------------------------===//

#map9 = affine_map<(d0, d1, d2) -> (0, d1, 0)>
#map6 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

func.func @mlir_minilm_4() {
  %cst_163 = arith.constant 9.9999999999999998E-13 : f64
  %1 = tensor.empty() : tensor<1x12x1xf32>
  %cst_1 = arith.constant 2.3: f32
  %86 = linalg.fill ins(%cst_1 : f32) outs(%1 : tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
  %13 = tensor.empty() : tensor<1x12x1xf32>
  %87 = linalg.generic {indexing_maps = [#map9, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%86 : tensor<1x12x1xf32>) outs(%13 : tensor<1x12x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %511 = arith.truncf %cst_163 : f64 to f32
      %512 = arith.addf %in, %511 : f32
      linalg.yield %512 : f32
    } -> tensor<1x12x1xf32>
    return 
}
