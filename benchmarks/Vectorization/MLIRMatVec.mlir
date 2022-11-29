//===- MLIRMatMul.mlir ----------------------------------------------------===//
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
// This file provides the MLIR MatVec function.
//
//===----------------------------------------------------------------------===//

func.func @mlir_matvec(%A: memref<?x?xf32>, %B: memref<?x?xf32>,
                       %C: memref<?x?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %m = memref.dim %A, %c0 : memref<?x?xf32>
  %x = memref.dim %A, %c1 : memref<?x?xf32>
  %n = memref.dim %B, %c1 : memref<?x?xf32>
  scf.for %i = %c0 to %n step %c1 {
    %b = memref.subview %B[0, %i][%x, 1][1, 1] : memref<?x?xf32> to memref<?xf32, strided<[?], offset: ?>>
    %c = memref.subview %C[0, %i][%m, 1][1, 1] : memref<?x?xf32> to memref<?xf32, strided<[?], offset: ?>>
    linalg.matvec ins(%A, %b: memref<?x?xf32>, memref<?xf32, strided<[?], offset: ?>>)
                  outs(%c: memref<?xf32, strided<[?], offset: ?>>)
  }
  return
}
