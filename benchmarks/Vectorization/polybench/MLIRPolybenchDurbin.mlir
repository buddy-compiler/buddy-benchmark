//===- MLIRPolybenchDurbin.mlir -------------------------------------------===//
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
// This file provides initialization and kernel functions for the durbin
// Polybench benchmark. The MLIR code is generated with Polygeist and modified
// manually to run on different dataset sizes.
//
//===----------------------------------------------------------------------===//

#map = affine_map<(d0) -> (d0)>

func.func @durbin_init_array(%arg0: i32, %arg1: memref<?xf64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c1_i32 = arith.constant 1 : i32
  %0 = arith.index_cast %arg0 : i32 to index
  scf.for %arg2 = %c0 to %0 step %c1 {
    %1 = arith.index_cast %arg2 : index to i32
    %2 = arith.addi %arg0, %c1_i32 : i32
    %3 = arith.subi %2, %1 : i32
    %4 = arith.sitofp %3 : i32 to f64
    memref.store %4, %arg1[%arg2] : memref<?xf64>
  }
  return
}

func.func @durbin(%arg0: i32, %arg1: memref<?xf64>, %arg2: memref<?xf64>) {
  %cst = arith.constant 0.000000e+00 : f64
  %cst_0 = arith.constant 1.000000e+00 : f64
  %alloca = memref.alloca() : memref<f64>
  %0 = llvm.mlir.undef : f64
  affine.store %0, %alloca[] : memref<f64>
  %alloca_1 = memref.alloca() : memref<f64>
  affine.store %0, %alloca_1[] : memref<f64>
  %alloca_2 = memref.alloca() : memref<f64>
  affine.store %0, %alloca_2[] : memref<f64>
  // manually modified to use dynamic allocation
  %arg0_cast = arith.index_cast %arg0 : i32 to index
  %alloca_3 = memref.alloca(%arg0_cast) : memref<?xf64>
  %1 = affine.load %arg1[0] : memref<?xf64>
  %2 = arith.negf %1 : f64
  affine.store %2, %arg2[0] : memref<?xf64>
  affine.store %cst_0, %alloca_1[] : memref<f64>
  %3 = affine.load %arg1[0] : memref<?xf64>
  %4 = arith.negf %3 : f64
  affine.store %4, %alloca_2[] : memref<f64>
  %5 = arith.index_cast %arg0 : i32 to index
  affine.for %arg3 = 1 to %5 {
    %6 = affine.load %alloca_2[] : memref<f64>
    %7 = arith.mulf %6, %6 : f64
    %8 = arith.subf %cst_0, %7 : f64
    %9 = affine.load %alloca_1[] : memref<f64>
    %10 = arith.mulf %8, %9 : f64
    affine.store %10, %alloca_1[] : memref<f64>
    affine.store %cst, %alloca[] : memref<f64>
    affine.for %arg4 = 0 to #map(%arg3) {
      %16 = affine.load %arg1[%arg3 - %arg4 - 1] : memref<?xf64>
      %17 = affine.load %arg2[%arg4] : memref<?xf64>
      %18 = arith.mulf %16, %17 : f64
      %19 = affine.load %alloca[] : memref<f64>
      %20 = arith.addf %19, %18 : f64
      affine.store %20, %alloca[] : memref<f64>
    }
    %11 = affine.load %arg1[%arg3] : memref<?xf64>
    %12 = affine.load %alloca[] : memref<f64>
    %13 = arith.addf %11, %12 : f64
    %14 = arith.negf %13 : f64
    %15 = arith.divf %14, %10 : f64
    affine.store %15, %alloca_2[] : memref<f64>
    affine.for %arg4 = 0 to #map(%arg3) {
      %16 = affine.load %arg2[%arg4] : memref<?xf64>
      %17 = affine.load %arg2[%arg3 - %arg4 - 1] : memref<?xf64>
      %18 = arith.mulf %15, %17 : f64
      %19 = arith.addf %16, %18 : f64
      affine.store %19, %alloca_3[%arg4] : memref<?xf64>
    }
    affine.for %arg4 = 0 to #map(%arg3) {
      %16 = affine.load %alloca_3[%arg4] : memref<?xf64>
      affine.store %16, %arg2[%arg4] : memref<?xf64>
    }
    affine.store %15, %arg2[%arg3] : memref<?xf64>
  }
  return
}
