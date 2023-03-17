//===- MLIRGccLoopsEx1.mlir ----------------------------------------------------===//
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
// This file provides the MLIR GccLoopsEx1 function.
//
//===----------------------------------------------------------------------===//


func.func @mlir_gccloopsex1(%A: memref<?xi32>, %B: memref<?xi32>,
                       %C: memref<?xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %n = memref.dim %B, %c0 : memref<?xi32>
  scf.for %i = %c0 to %n step %c1 {
    %b_val = memref.load %B[%i] : memref<?xi32>
    %c_val = memref.load %C[%i] : memref<?xi32>
    %result = arith.addi %b_val, %c_val : i32
    memref.store %result, %A[%i] : memref<?xi32>
  }
  return
}
