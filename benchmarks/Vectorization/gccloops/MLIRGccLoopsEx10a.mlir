//===- MLIRGccLoopsEx10a.mlir ----------------------------------------------------===//
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
// This file provides the MLIR GccLoopsEx10a function.
//
//===----------------------------------------------------------------------===//

func.func @mlir_gccloopsex10a(%sa: memref<?xi16>, %sb: memref<?xi16>, %sc: memref<?xi16>,
                              %ia: memref<?xi32>, %ib: memref<?xi32>, %ic: memref<?xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index 

  scf.for %iv = %c0 to %c10 step %c1{
    %ib_value = memref.load %ib[%iv] : memref<?xi32>
    %ic_value = memref.load %ic[%iv] : memref<?xi32>
    %sum_ib_ic = arith.addi %ib_value, %ic_value : i32
    memref.store %sum_ib_ic, %ia[%iv] : memref<?xi32>

    %sb_value = memref.load %sb[%iv] : memref<?xi16>
    %sc_value = memref.load %sc[%iv] : memref<?xi16>
    %sum_sb_sc = arith.addi %sb_value, %sc_value : i16
    memref.store %sum_sb_sc, %sa[%iv] : memref<?xi16>
  }
  return
}




