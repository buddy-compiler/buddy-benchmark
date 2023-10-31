//===- MLIRMiniLM_2.mlir --------------------------------------------------===//
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
// This file provides the MLIR MLIRMiniLM_2 function.
//
//===----------------------------------------------------------------------===//

#map15 = affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
func.func @mlir_minilm_2() {
  %1 = tensor.empty() : tensor<1x12x12x12xf32>
  %2 = tensor.empty() : tensor<1x12x12x12xf32>
  %cst_1 = arith.constant 2.3: f32
  %cst_2 = arith.constant 0.000000e+00 : f32
  %67 = linalg.fill ins(%cst_1 : f32) outs(%1 : tensor<1x12x12x12xf32>) -> tensor<1x12x12x12xf32>
  %55 = linalg.fill ins(%cst_2 : f32) outs(%2 : tensor<1x12x12x12xf32>) -> tensor<1x12x12x12xf32>
  %68 = linalg.generic {indexing_maps = [#map15, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%67 : tensor<1x12x12x12xf32>) outs(%55 : tensor<1x12x12x12xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x12x12xf32>
    return
}

