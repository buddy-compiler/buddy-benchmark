//===- MLIRPolybenchFloydWarshall.mlir ------------------------------------===//
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
// This file provides initialization and kernel functions for the floyd-warshall
// Polybench benchmark. The MLIR code is generated with Polygeist and modified
// manually to run on different dataset sizes.
//
//===----------------------------------------------------------------------===//

func.func @floyd_warshall_init_array(%arg0: i32, %arg1: memref<?x?xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %true = arith.constant true
  %c999_i32 = arith.constant 999 : i32
  %c11_i32 = arith.constant 11 : i32
  %c13_i32 = arith.constant 13 : i32
  %c1_i32 = arith.constant 1 : i32
  %c7_i32 = arith.constant 7 : i32
  %c0_i32 = arith.constant 0 : i32
  %0 = arith.index_cast %arg0 : i32 to index
  scf.for %arg2 = %c0 to %0 step %c1 {
    %1 = arith.index_cast %arg2 : index to i32
    %2 = arith.index_cast %arg0 : i32 to index
    scf.for %arg3 = %c0 to %2 step %c1 {
      %3 = arith.index_cast %arg3 : index to i32
      %4 = arith.muli %1, %3 : i32
      %5 = arith.remsi %4, %c7_i32 : i32
      %6 = arith.addi %5, %c1_i32 : i32
      memref.store %6, %arg1[%arg2, %arg3] : memref<?x?xi32>
      %7 = arith.addi %1, %3 : i32
      %8 = arith.remsi %7, %c13_i32 : i32
      %9 = arith.cmpi eq, %8, %c0_i32 : i32
      %10 = scf.if %9 -> (i1) {
        scf.yield %true : i1
      } else {
        %12 = arith.remsi %7, %c7_i32 : i32
        %13 = arith.cmpi eq, %12, %c0_i32 : i32
        scf.yield %13 : i1
      }
      %11 = scf.if %10 -> (i1) {
        scf.yield %true : i1
      } else {
        %12 = arith.remsi %7, %c11_i32 : i32
        %13 = arith.cmpi eq, %12, %c0_i32 : i32
        scf.yield %13 : i1
      }
      scf.if %11 {
        memref.store %c999_i32, %arg1[%arg2, %arg3] : memref<?x?xi32>
      }
    }
  }
  return
}

func.func @floyd_warshall_kernel(%arg0: i32, %arg1: memref<?x?xi32>) {
  %0 = arith.index_cast %arg0 : i32 to index
  affine.for %arg2 = 0 to %0 {
    affine.for %arg3 = 0 to %0 {
      affine.for %arg4 = 0 to %0 {
        %1 = affine.load %arg1[%arg3, %arg4] : memref<?x?xi32>
        %2 = affine.load %arg1[%arg3, %arg2] : memref<?x?xi32>
        %3 = affine.load %arg1[%arg2, %arg4] : memref<?x?xi32>
        %4 = arith.addi %2, %3 : i32
        %5 = arith.cmpi slt, %1, %4 : i32
        %6 = scf.if %5 -> (i32) {
          scf.yield %1 : i32
        } else {
          %7 = arith.addi %2, %3 : i32
          scf.yield %7 : i32
        }
        affine.store %6, %arg1[%arg3, %arg4] : memref<?x?xi32>
      }
    }
  }
  return
}
