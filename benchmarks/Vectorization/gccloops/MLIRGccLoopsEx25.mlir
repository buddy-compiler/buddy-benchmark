//===- MLIRGccLoopsEx25.mlir ----------------------------------------------------===//
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
// This file provides the MLIR GccLoopsEx25 function.
//
//===----------------------------------------------------------------------===//

func.func @mlir_gccloopsex25(%dj: memref<?xi32>, %da: memref<?xf32>, %db: memref<?xf32>,
                             %dc: memref<?xf32>, %dd: memref<?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index

  scf.for %iv = %c0 to %c10 step %c1{
    %da_value = memref.load %da[%iv] :  memref<?xf32>
    %db_value = memref.load %db[%iv] :  memref<?xf32>
    %dc_value = memref.load %dc[%iv] :  memref<?xf32>
    %dd_value = memref.load %dd[%iv] :  memref<?xf32>
    %condition_1 = arith.cmpf ult, %da_value, %db_value : f32
    %condition_2 = arith.cmpf ult, %dc_value, %dd_value : f32
    %dj_value = arith.andi %condition_1, %condition_2 : i1
    %res = arith.extsi %dj_value : i1 to i32
    // memref.store %res, %dj[%iv] : memref<?xi32>  FIX ME: Illegal instruction
  }
  return
}




