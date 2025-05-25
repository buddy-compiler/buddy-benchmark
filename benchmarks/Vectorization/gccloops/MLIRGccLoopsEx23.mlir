//===- MLIRGccLoopsEx23.mlir ----------------------------------------------------===//
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
// This file provides the MLIR GccLoopsEx23 function.
//
//===----------------------------------------------------------------------===//

func.func @mlir_gccloopsex23(%src: memref<?xi16>, %dst: memref<?xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c7 = arith.constant 7 : i16
  %c10 = arith.constant 10 : index

  scf.for %iv = %c0 to %c10 step %c1{
    %src_value = memref.load %src[%iv] : memref<?xi16>
    %src_value_shl7 = arith.shrsi %src_value, %c7 : i16
    %res = arith.extsi %src_value_shl7 : i16 to i32
    memref.store %res, %dst[%iv] : memref<?xi32>
  }
  return
}




