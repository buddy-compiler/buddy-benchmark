//===- MLIRGccLoopsEx14.mlir ----------------------------------------------------===//
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
// This file provides the MLIR GccLoopsEx14 function.
//
//===----------------------------------------------------------------------===//

func.func @mlir_gccloopsex14(%in: memref<?x?xi32>, %coeff: memref<?x?xi32>, %out: memref<?xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %sum_0 = arith.constant 0 : i32 

  scf.for %kv = %c0 to %c2 step %c1{
    scf.for %jv = %c0 to %c3 step %c1{
      %sum = scf.for %iv = %c0 to %c4 step %c1
        iter_args(%sum_iter = %sum_0) -> (i32){
        %i_plus_k = arith.addi %iv, %kv : index
        %in_value = memref.load %in[%i_plus_k, %jv] : memref<?x?xi32>
        %coeff_value = memref.load %in[%iv, %jv] : memref<?x?xi32>
        %in_mul_coeff = arith.muli %in_value, %coeff_value : i32
        %1 = arith.addi %sum_iter, %in_mul_coeff : i32
        scf.yield %1 : i32
      }
      memref.store %sum, %out[%kv] : memref<?xi32>
    }
  }
  return
}




