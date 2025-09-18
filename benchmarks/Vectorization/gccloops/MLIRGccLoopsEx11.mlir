//===- MLIRGccLoopsEx11.mlir ----------------------------------------------------===//
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
// This file provides the MLIR GccLoopsEx11 function.
//
//===----------------------------------------------------------------------===//

func.func @mlir_gccloopsex11(%A: memref<?xi32>, %B: memref<?xi32>,
                            %C: memref<?xi32>, %D: memref<?xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c5 = arith.constant 5 : index

  scf.for %iv = %c0 to %c5 step %c1{
    %two_iv = arith.muli %iv, %c2 : index
    %two_iv_1 = arith.addi %two_iv, %c1 : index
    %b_2_iv = memref.load %B[%two_iv] : memref<?xi32>
    %b_2_iv_1 = memref.load %B[%two_iv_1] : memref<?xi32>
    %c_2_iv = memref.load %C[%two_iv] : memref<?xi32>
    %c_2_iv_1 = memref.load %C[%two_iv_1] : memref<?xi32>
    %temp1 = arith.muli %b_2_iv_1, %c_2_iv_1 : i32
    %temp2 = arith.muli %b_2_iv, %c_2_iv : i32
    %a_value = arith.subi %temp1, %temp2 : i32
    memref.store %a_value, %A[%iv] : memref<?xi32>
    %temp3 = arith.muli %b_2_iv, %c_2_iv_1 : i32
    %temp4 = arith.muli %b_2_iv_1, %c_2_iv : i32
    %d_value = arith.subi %temp3, %temp4 : i32
    memref.store %d_value, %D[%iv] : memref<?xi32>
  }
  return
}




