//===- MLIRGccLoopsEx21.mlir ----------------------------------------------------===//
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
// This file provides the MLIR GccLoopsEx21 function.
//
//===----------------------------------------------------------------------===//

func.func @mlir_gccloopsex21(%b: memref<?xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %sum_0 = arith.constant 0 : i32

  %sum = scf.for %iv = %c0 to %c10 step %c1
  iter_args(%sum_iter = %sum_0) -> (i32){
    %b_value = memref.load %b[%iv] : memref<?xi32>
    %1 = arith.addi %sum_iter, %b_value : i32
    scf.yield %1 : i32
  }
  memref.store %sum, %b[%c0] : memref<?xi32>
  return
}




