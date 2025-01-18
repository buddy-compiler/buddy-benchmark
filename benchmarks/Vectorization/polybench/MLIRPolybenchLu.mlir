//===- MLIRPolybenchLu.mlir -----------------------------------------------===//
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
// This file provides initialization and kernel functions for the lu Polybench
// benchmark. The MLIR code is generated with Polygeist and modified manually to
// run on different dataset sizes.
//
//===----------------------------------------------------------------------===//

#map = affine_map<(d0) -> (d0)>

func.func @lu_init_array(%arg0: i32, %arg1: memref<?x?xf64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.000000e+00 : f64
  %c1_i32 = arith.constant 1 : i32
  %cst_0 = arith.constant 1.000000e+00 : f64
  %c0_i32 = arith.constant 0 : i32
  %0 = arith.index_cast %arg0 : i32 to index
  scf.for %arg2 = %c0 to %0 step %c1 {
    %4 = arith.index_cast %arg2 : index to i32
    %5 = arith.addi %4, %c1_i32 : i32
    %6 = arith.index_cast %5 : i32 to index
    scf.for %arg3 = %c0 to %6 step %c1 {
      %10 = arith.index_cast %arg3 : index to i32
      %11 = arith.subi %c0_i32, %10 : i32
      %12 = arith.remsi %11, %arg0 : i32
      %13 = arith.sitofp %12 : i32 to f64
      %14 = arith.sitofp %arg0 : i32 to f64
      %15 = arith.divf %13, %14 : f64
      %16 = arith.addf %15, %cst_0 : f64
      memref.store %16, %arg1[%arg2, %arg3] : memref<?x?xf64>
    }
    %7 = arith.addi %4, %c1_i32 : i32
    %8 = arith.index_cast %arg0 : i32 to index
    %9 = arith.index_cast %7 : i32 to index
    scf.for %arg3 = %9 to %8 step %c1 {
      %10 = arith.subi %arg3, %9 : index
      %11 = arith.index_cast %7 : i32 to index
      %12 = arith.addi %11, %10 : index
      memref.store %cst, %arg1[%arg2, %12] : memref<?x?xf64>
    }
    memref.store %cst_0, %arg1[%arg2, %arg2] : memref<?x?xf64>
  }
  // manually modified to use dynamic allocation
  %arg0_cast = arith.index_cast %arg0 : i32 to index
  %alloc = memref.alloc(%arg0_cast, %arg0_cast) : memref<?x?xf64>
  %1 = arith.index_cast %arg0 : i32 to index
  scf.for %arg2 = %c0 to %1 step %c1 {
    %4 = arith.index_cast %arg0 : i32 to index
    scf.for %arg3 = %c0 to %4 step %c1 {
      memref.store %cst, %alloc[%arg2, %arg3] : memref<?x?xf64>
    }
  }
  %2 = arith.index_cast %arg0 : i32 to index
  scf.for %arg2 = %c0 to %2 step %c1 {
    %4 = arith.index_cast %arg0 : i32 to index
    scf.for %arg3 = %c0 to %4 step %c1 {
      %5 = arith.index_cast %arg0 : i32 to index
      scf.for %arg4 = %c0 to %5 step %c1 {
        %6 = memref.load %arg1[%arg3, %arg2] : memref<?x?xf64>
        %7 = memref.load %arg1[%arg4, %arg2] : memref<?x?xf64>
        %8 = arith.mulf %6, %7 : f64
        %9 = memref.load %alloc[%arg3, %arg4] : memref<?x?xf64>
        %10 = arith.addf %9, %8 : f64
        memref.store %10, %alloc[%arg3, %arg4] : memref<?x?xf64>
      }
    }
  }
  %3 = arith.index_cast %arg0 : i32 to index
  scf.for %arg2 = %c0 to %3 step %c1 {
    %4 = arith.index_cast %arg0 : i32 to index
    scf.for %arg3 = %c0 to %4 step %c1 {
      %5 = memref.load %alloc[%arg2, %arg3] : memref<?x?xf64>
      memref.store %5, %arg1[%arg2, %arg3] : memref<?x?xf64>
    }
  }
  memref.dealloc %alloc : memref<?x?xf64>
  return
}

func.func @lu_kernel(%arg0: i32, %arg1: memref<?x?xf64>) {
  %0 = arith.index_cast %arg0 : i32 to index
  affine.for %arg2 = 0 to %0 {
    affine.for %arg3 = 0 to #map(%arg2) {
      affine.for %arg4 = 0 to #map(%arg3) {
        %4 = affine.load %arg1[%arg2, %arg4] : memref<?x?xf64>
        %5 = affine.load %arg1[%arg4, %arg3] : memref<?x?xf64>
        %6 = arith.mulf %4, %5 : f64
        %7 = affine.load %arg1[%arg2, %arg3] : memref<?x?xf64>
        %8 = arith.subf %7, %6 : f64
        affine.store %8, %arg1[%arg2, %arg3] : memref<?x?xf64>
      }
      %1 = affine.load %arg1[%arg3, %arg3] : memref<?x?xf64>
      %2 = affine.load %arg1[%arg2, %arg3] : memref<?x?xf64>
      %3 = arith.divf %2, %1 : f64
      affine.store %3, %arg1[%arg2, %arg3] : memref<?x?xf64>
    }
    affine.for %arg3 = #map(%arg2) to %0 {
      affine.for %arg4 = 0 to #map(%arg2) {
        %1 = affine.load %arg1[%arg2, %arg4] : memref<?x?xf64>
        %2 = affine.load %arg1[%arg4, %arg3] : memref<?x?xf64>
        %3 = arith.mulf %1, %2 : f64
        %4 = affine.load %arg1[%arg2, %arg3] : memref<?x?xf64>
        %5 = arith.subf %4, %3 : f64
        affine.store %5, %arg1[%arg2, %arg3] : memref<?x?xf64>
      }
    }
  }
  return
}
