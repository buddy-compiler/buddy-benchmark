//===- FIR.mlir -----------------------------------------------------------===//
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
// This file provides the `dap.fir` operations with various types. 
//
//===----------------------------------------------------------------------===//

func.func @buddy_fir_f32(%in : memref<?xf32>, %filter : memref<?xf32>, %out : memref<?xf32>) -> () {
  dap.fir %in, %filter, %out : memref<?xf32>, memref<?xf32>, memref<?xf32>
  return
}

func.func @buddy_fir_f64(%in : memref<?xf64>, %filter : memref<?xf64>, %out : memref<?xf64>) -> () {
  dap.fir %in, %filter, %out : memref<?xf64>, memref<?xf64>, memref<?xf64>
  return
}
