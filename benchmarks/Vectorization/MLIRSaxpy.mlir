//===- MLIRSAXPY.mlir ----------------------------------------------------===//
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
// This file provides the MLIR SAXPY function.
//
//===----------------------------------------------------------------------===//

func.func @mlir_saxpy(%A: memref<?xf32>, %B: memref<?xf32>,
                       %C: memref<?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 8.3 : f32
  %m = memref.dim %A, %c0 : memref<?xf32>
  %n = memref.dim %B, %c0 : memref<?xf32>
  %o = memref.dim %C, %c0 : memref<?xf32>
  %min0 = arith.minui %m, %n : index
  %min1 = arith.minui %min0, %o : index
  scf.for %i = %c0 to %min1 step %c1 {
    %a = memref.load %A[%i] : memref<?xf32>
    %b = memref.load %B[%i] : memref<?xf32>
    %c = arith.mulf %c2, %a : f32
    %d = arith.addf %b, %c : f32
    memref.store %d, %C[%i] : memref<?xf32>
  }
  return
}