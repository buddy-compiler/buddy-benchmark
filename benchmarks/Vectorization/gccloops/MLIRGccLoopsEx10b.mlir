//===- MLIRGccLoopsEx10b.mlir ----------------------------------------------------===//
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
// This file provides the MLIR GccLoopsEx10b function.
//
//===----------------------------------------------------------------------===//

func.func @mlir_gccloopsex10b(%sb: memref<?xi16>, %ia: memref<?xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index 

  scf.for %iv = %c0 to %c10 step %c1{
    %sb_value = memref.load %sb[%iv] : memref<?xi16>
    %sb_value_exti = arith.extsi %sb_value : i16 to i32
    memref.store %sb_value_exti, %ia[%iv] : memref<?xi32>
  }
  return
}




