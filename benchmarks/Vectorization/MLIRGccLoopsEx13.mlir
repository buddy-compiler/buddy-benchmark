//===- MLIRGccLoopsEx13.mlir ----------------------------------------------------===//
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
// This file provides the MLIR GccLoopsEx13 function.
//
//===----------------------------------------------------------------------===//

func.func @mlir_gccloopsex13(%A: memref<?x?xi32>, %B: memref<?x?xi32>, %out: memref<?xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %c5 = arith.constant 5 : index
  %c8 = arith.constant 8 : index
  %sum_0 = arith.constant 0 : i32 

  scf.for %iv = %c0 to %c5 step %c1{
    %sum = scf.for %jv = %c0 to %c16 step %c8
    iter_args(%sum_iter = %sum_0) -> (i32) {
      %a_value = memref.load %A[%iv, %jv] : memref<?x?xi32>
      %b_value = memref.load %B[%iv, %jv] : memref<?x?xi32>
      %a_sub_b = arith.subi %a_value, %b_value : i32
      %1 = arith.addi %sum_iter, %a_sub_b : i32
      scf.yield %1 : i32
    }
    memref.store %sum, %out[%iv] : memref<?xi32>
  }
  return
}




