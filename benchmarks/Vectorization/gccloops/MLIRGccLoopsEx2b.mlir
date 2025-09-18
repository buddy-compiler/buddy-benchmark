//===- MLIRGccLoopsEx2b.mlir ----------------------------------------------------===//
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
// This file provides the MLIR GccLoopsEx2b function.
//
//===----------------------------------------------------------------------===//


func.func @mlir_gccloopsex2b(%A: memref<?xi32>, %B: memref<?xi32>,
                       %C: memref<?xi32>, %N: index) -> index {  
  
  %c0 =  arith.constant 0 : index
  %c1 =  arith.constant 1 : index
  %i_init = arith.constant 0 : index

  %n_final, %i_final = scf.while (%n = %N, %i = %i_init) : (index, index) -> (index, index) {
    %condition = arith.cmpi ugt, %n, %c0 : index
    scf.condition(%condition) %n, %i : index, index
  } do {
  ^bb0(%n: index, %i: index):
    %b_value = memref.load %B[%i] : memref<?xi32>
    %c_value = memref.load %C[%i] : memref<?xi32>
    %value = arith.andi %b_value, %c_value : i32

    memref.store %value, %A[%i] : memref<?xi32>

    %next_i = arith.addi %i, %c1 : index
    %next_n = arith.subi %n, %c1 : index

    scf.yield %next_n, %next_i : index, index
  }
  return %i_final : index
}
