//===- MLIRGccLoopsEx8.mlir ----------------------------------------------------===//
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
// This file provides the MLIR GccLoopsEx8 function.
//
//===----------------------------------------------------------------------===//

func.func @mlir_gccloopsex8(%x: i32, %G: memref<?x?xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c5 = arith.constant 5 : index
  %M = memref.dim %G, %c0 : memref<?x?xi32>
  %N = memref.dim %G, %c1 : memref<?x?xi32>

  scf.for %i = %c0 to %c5 step %c1 {
    scf.for %j = %c0 to %c2 step %c1{
      memref.store %x, %G[%i, %j] : memref<?x?xi32>
    }
  }
  return
}




