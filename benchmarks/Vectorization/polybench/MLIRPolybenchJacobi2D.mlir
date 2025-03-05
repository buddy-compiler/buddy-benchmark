//===- MLIRPolybenchJacobi2D.mlir -----------------------------------------===//
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
// This file provides initialization and kernel functions for the jacobi-2d
// Polybench benchmark. The MLIR code is generated with Polygeist and modified
// manually to run on different dataset sizes.
//
//===----------------------------------------------------------------------===//

#map = affine_map<()[s0] -> (s0 - 1)>

func.func @jacobi_2d_init_array(%arg0: i32, %arg1: memref<?x?xf64>, %arg2: memref<?x?xf64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 3.000000e+00 : f64
  %c3_i32 = arith.constant 3 : i32
  %cst_0 = arith.constant 2.000000e+00 : f64
  %c2_i32 = arith.constant 2 : i32
  %0 = arith.index_cast %arg0 : i32 to index
  scf.for %arg3 = %c0 to %0 step %c1 {
    %1 = arith.index_cast %arg3 : index to i32
    %2 = arith.index_cast %arg0 : i32 to index
    scf.for %arg4 = %c0 to %2 step %c1 {
      %3 = arith.index_cast %arg4 : index to i32
      %4 = arith.sitofp %1 : i32 to f64
      %5 = arith.addi %3, %c2_i32 : i32
      %6 = arith.sitofp %5 : i32 to f64
      %7 = arith.mulf %4, %6 : f64
      %8 = arith.addf %7, %cst_0 : f64
      %9 = arith.sitofp %arg0 : i32 to f64
      %10 = arith.divf %8, %9 : f64
      memref.store %10, %arg1[%arg3, %arg4] : memref<?x?xf64>
      %11 = arith.addi %3, %c3_i32 : i32
      %12 = arith.sitofp %11 : i32 to f64
      %13 = arith.mulf %4, %12 : f64
      %14 = arith.addf %13, %cst : f64
      %15 = arith.divf %14, %9 : f64
      memref.store %15, %arg2[%arg3, %arg4] : memref<?x?xf64>
    }
  }
  return
}

func.func @jacobi_2d_kernel(%arg0: i32, %arg1: i32, %arg2: memref<?x?xf64>, %arg3: memref<?x?xf64>) {
  %cst = arith.constant 2.000000e-01 : f64
  %0 = arith.index_cast %arg1 : i32 to index
  %1 = arith.index_cast %arg1 : i32 to index
  %2 = arith.index_cast %arg1 : i32 to index
  %3 = arith.index_cast %arg1 : i32 to index
  %4 = arith.index_cast %arg0 : i32 to index
  affine.for %arg4 = 0 to %4 {
    affine.for %arg5 = 1 to #map()[%0] {
      affine.for %arg6 = 1 to #map()[%1] {
        %5 = affine.load %arg2[%arg5, %arg6] : memref<?x?xf64>
        %6 = affine.load %arg2[%arg5, %arg6 - 1] : memref<?x?xf64>
        %7 = arith.addf %5, %6 : f64
        %8 = affine.load %arg2[%arg5, %arg6 + 1] : memref<?x?xf64>
        %9 = arith.addf %7, %8 : f64
        %10 = affine.load %arg2[%arg5 + 1, %arg6] : memref<?x?xf64>
        %11 = arith.addf %9, %10 : f64
        %12 = affine.load %arg2[%arg5 - 1, %arg6] : memref<?x?xf64>
        %13 = arith.addf %11, %12 : f64
        %14 = arith.mulf %13, %cst : f64
        affine.store %14, %arg3[%arg5, %arg6] : memref<?x?xf64>
      }
    }
    affine.for %arg5 = 1 to #map()[%2] {
      affine.for %arg6 = 1 to #map()[%3] {
        %5 = affine.load %arg3[%arg5, %arg6] : memref<?x?xf64>
        %6 = affine.load %arg3[%arg5, %arg6 - 1] : memref<?x?xf64>
        %7 = arith.addf %5, %6 : f64
        %8 = affine.load %arg3[%arg5, %arg6 + 1] : memref<?x?xf64>
        %9 = arith.addf %7, %8 : f64
        %10 = affine.load %arg3[%arg5 + 1, %arg6] : memref<?x?xf64>
        %11 = arith.addf %9, %10 : f64
        %12 = affine.load %arg3[%arg5 - 1, %arg6] : memref<?x?xf64>
        %13 = arith.addf %11, %12 : f64
        %14 = arith.mulf %13, %cst : f64
        affine.store %14, %arg2[%arg5, %arg6] : memref<?x?xf64>
      }
    }
  }
  return
}
