//===- MLIRGccLoopsEx7.mlir ----------------------------------------------------===//
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
// This file provides the MLIR GccLoopsEx7 function.
//
//===----------------------------------------------------------------------===//

func.func @mlir_gccloopsex7(%x: index, %A: memref<?xi32>, %B: memref<?xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  
  scf.for %i = %c0 to %c10 step %c1 {
    %i_plus_x = arith.addi %x, %i : index
    %B_value = memref.load %B[%i_plus_x] : memref<?xi32>
    memref.store %B_value, %A[%i] : memref<?xi32>
  }
  return
}




