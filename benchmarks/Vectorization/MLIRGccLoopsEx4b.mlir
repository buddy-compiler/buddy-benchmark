//===- MLIRGccLoopsEx4a.mlir ----------------------------------------------------===//
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
// This file provides the MLIR GccLoopsEx4a function.
//
//===----------------------------------------------------------------------===//


func.func @mlir_gccloopsex4b(%A: memref<?xi32>, %B: memref<?xi32>,
                       %C: memref<?xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %n = arith.constant 10 : index

  scf.for %i = %c0 to %n step %c1 {

    %i_b = arith.addi %i, %c1 : index
    %i_c = arith.addi %i, %c3 : index
    %b_val = memref.load %B[%i_b] : memref<?xi32>
    %c_val = memref.load %C[%i_c] : memref<?xi32>
    %result = arith.addi %b_val, %c_val : i32
    memref.store %result, %A[%i] : memref<?xi32>
  }
  return
}



