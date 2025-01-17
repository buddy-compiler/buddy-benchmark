//===- MLIRPolybenchTrisolv.mlir ------------------------------------------===//
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
// This file provides initialization and kernel functions for the trisolv
// Polybench benchmark. The MLIR code is generated with Polygeist and modified
// manually to run on different dataset sizes.
//
//===----------------------------------------------------------------------===//

#map = affine_map<(d0) -> (d0)>

func.func @trisolv_init_array(%arg0: i32, %arg1: memref<?x?xf64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 2.000000e+00 : f64
  %c1_i32 = arith.constant 1 : i32
  %cst_0 = arith.constant -9.990000e+02 : f64
  %0 = arith.index_cast %arg0 : i32 to index
  scf.for %arg4 = %c0 to %0 step %c1 {
    %1 = arith.index_cast %arg4 : index to i32
    memref.store %cst_0, %arg2[%arg4] : memref<?xf64>
    %2 = arith.sitofp %1 : i32 to f64
    memref.store %2, %arg3[%arg4] : memref<?xf64>
    %3 = arith.addi %1, %c1_i32 : i32
    %4 = arith.index_cast %3 : i32 to index
    scf.for %arg5 = %c0 to %4 step %c1 {
      %5 = arith.index_cast %arg5 : index to i32
      %6 = arith.addi %1, %arg0 : i32
      %7 = arith.subi %6, %5 : i32
      %8 = arith.addi %7, %c1_i32 : i32
      %9 = arith.sitofp %8 : i32 to f64
      %10 = arith.mulf %9, %cst : f64
      %11 = arith.sitofp %arg0 : i32 to f64
      %12 = arith.divf %10, %11 : f64
      memref.store %12, %arg1[%arg4, %arg5] : memref<?x?xf64>
    }
  }
  return
}

func.func @trisolv(%arg0: i32, %arg1: memref<?x?xf64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>) {
  %0 = arith.index_cast %arg0 : i32 to index
  affine.for %arg4 = 0 to %0 {
    %1 = affine.load %arg3[%arg4] : memref<?xf64>
    affine.store %1, %arg2[%arg4] : memref<?xf64>
    affine.for %arg5 = 0 to #map(%arg4) {
      %5 = affine.load %arg1[%arg4, %arg5] : memref<?x?xf64>
      %6 = affine.load %arg2[%arg5] : memref<?xf64>
      %7 = arith.mulf %5, %6 : f64
      %8 = affine.load %arg2[%arg4] : memref<?xf64>
      %9 = arith.subf %8, %7 : f64
      affine.store %9, %arg2[%arg4] : memref<?xf64>
    }
    %2 = affine.load %arg2[%arg4] : memref<?xf64>
    %3 = affine.load %arg1[%arg4, %arg4] : memref<?x?xf64>
    %4 = arith.divf %2, %3 : f64
    affine.store %4, %arg2[%arg4] : memref<?xf64>
  }
  return
}
