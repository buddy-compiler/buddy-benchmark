//===- MLIRPolybenchGesummv.mlir ------------------------------------------===//
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
// This file provides initialization and kernel functions for the gesummv
// Polybench benchmark. The MLIR code is generated with Polygeist and modified
// manually to run on different dataset sizes.
//
//===----------------------------------------------------------------------===//

func.func @gesummv_init_array(%arg0: i32, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: memref<?x?xf64>, %arg4: memref<?x?xf64>, %arg5: memref<?xf64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2_i32 = arith.constant 2 : i32
  %c1_i32 = arith.constant 1 : i32
  %cst = arith.constant 1.200000e+00 : f64
  %cst_0 = arith.constant 1.500000e+00 : f64
  affine.store %cst_0, %arg1[0] : memref<?xf64>
  affine.store %cst, %arg2[0] : memref<?xf64>
  %0 = arith.index_cast %arg0 : i32 to index
  scf.for %arg6 = %c0 to %0 step %c1 {
    %1 = arith.index_cast %arg6 : index to i32
    %2 = arith.remsi %1, %arg0 : i32
    %3 = arith.sitofp %2 : i32 to f64
    %4 = arith.sitofp %arg0 : i32 to f64
    %5 = arith.divf %3, %4 : f64
    memref.store %5, %arg5[%arg6] : memref<?xf64>
    %6 = arith.index_cast %arg0 : i32 to index
    scf.for %arg7 = %c0 to %6 step %c1 {
      %7 = arith.index_cast %arg7 : index to i32
      %8 = arith.muli %1, %7 : i32
      %9 = arith.addi %8, %c1_i32 : i32
      %10 = arith.remsi %9, %arg0 : i32
      %11 = arith.sitofp %10 : i32 to f64
      %12 = arith.divf %11, %4 : f64
      memref.store %12, %arg3[%arg6, %arg7] : memref<?x?xf64>
      %13 = arith.addi %8, %c2_i32 : i32
      %14 = arith.remsi %13, %arg0 : i32
      %15 = arith.sitofp %14 : i32 to f64
      %16 = arith.divf %15, %4 : f64
      memref.store %16, %arg4[%arg6, %arg7] : memref<?x?xf64>
    }
  }
  return
}

func.func @gesummv_kernel(%arg0: i32, %arg1: f64, %arg2: f64, %arg3: memref<?x?xf64>, %arg4: memref<?x?xf64>, %arg5: memref<?xf64>, %arg6: memref<?xf64>, %arg7: memref<?xf64>) {
  %cst = arith.constant 0.000000e+00 : f64
  %0 = arith.index_cast %arg0 : i32 to index
  affine.for %arg8 = 0 to %0 {
    affine.store %cst, %arg5[%arg8] : memref<?xf64>
    affine.store %cst, %arg7[%arg8] : memref<?xf64>
    affine.for %arg9 = 0 to %0 {
      %6 = affine.load %arg3[%arg8, %arg9] : memref<?x?xf64>
      %7 = affine.load %arg6[%arg9] : memref<?xf64>
      %8 = arith.mulf %6, %7 : f64
      %9 = affine.load %arg5[%arg8] : memref<?xf64>
      %10 = arith.addf %8, %9 : f64
      affine.store %10, %arg5[%arg8] : memref<?xf64>
      %11 = affine.load %arg4[%arg8, %arg9] : memref<?x?xf64>
      %12 = affine.load %arg6[%arg9] : memref<?xf64>
      %13 = arith.mulf %11, %12 : f64
      %14 = affine.load %arg7[%arg8] : memref<?xf64>
      %15 = arith.addf %13, %14 : f64
      affine.store %15, %arg7[%arg8] : memref<?xf64>
    }
    %1 = affine.load %arg5[%arg8] : memref<?xf64>
    %2 = arith.mulf %arg1, %1 : f64
    %3 = affine.load %arg7[%arg8] : memref<?xf64>
    %4 = arith.mulf %arg2, %3 : f64
    %5 = arith.addf %2, %4 : f64
    affine.store %5, %arg7[%arg8] : memref<?xf64>
  }
  return
}
