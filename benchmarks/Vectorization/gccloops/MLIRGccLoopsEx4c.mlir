//===- MLIRGccLoopsEx4c.mlir ----------------------------------------------------===//
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
// This file provides the MLIR GccLoopsEx4c function.
//
//===----------------------------------------------------------------------===//


func.func @mlir_gccloopsex4c(%B: memref<?xi32>, %A: memref<?xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %n = arith.constant 10 : index
  %max = arith.constant 4 : i32
  %c0_i32 = arith.constant 0 : i32

  scf.for %i = %c0 to %n step %c1 {
    %j = memref.load %A[%i] : memref<?xi32>
    %condition = arith.cmpi ugt, %j, %max : i32
    %a_value = arith.select %condition, %max, %c0_i32 : i32
    memref.store %a_value, %B[%i] : memref<?xi32>
  }
  return
}



