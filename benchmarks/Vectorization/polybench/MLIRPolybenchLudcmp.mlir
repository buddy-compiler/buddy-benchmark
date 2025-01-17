//===- MLIRPolybenchLudcmp.mlir -------------------------------------------===//
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
// This file provides initialization and kernel functions for the ludcmp
// Polybench benchmark. The MLIR code is generated with Polygeist and modified
// manually to run on different dataset sizes.
//
//===----------------------------------------------------------------------===//

#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0)[s0] -> (-d0 + s0)>

func.func @ludcmp_init_array(%arg0: i32, %arg1: memref<?x?xf64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>, %arg4: memref<?xf64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 1.000000e+00 : f64
  %cst_0 = arith.constant 4.000000e+00 : f64
  %cst_1 = arith.constant 2.000000e+00 : f64
  %c1_i32 = arith.constant 1 : i32
  %cst_2 = arith.constant 0.000000e+00 : f64
  %c0_i32 = arith.constant 0 : i32
  %0 = arith.sitofp %arg0 : i32 to f64
  %1 = arith.index_cast %arg0 : i32 to index
  scf.for %arg5 = %c0 to %1 step %c1 {
    %6 = arith.index_cast %arg5 : index to i32
    memref.store %cst_2, %arg3[%arg5] : memref<?xf64>
    memref.store %cst_2, %arg4[%arg5] : memref<?xf64>
    %7 = arith.addi %6, %c1_i32 : i32
    %8 = arith.sitofp %7 : i32 to f64
    %9 = arith.divf %8, %0 : f64
    %10 = arith.divf %9, %cst_1 : f64
    %11 = arith.addf %10, %cst_0 : f64
    memref.store %11, %arg2[%arg5] : memref<?xf64>
  }
  %2 = arith.index_cast %arg0 : i32 to index
  scf.for %arg5 = %c0 to %2 step %c1 {
    %6 = arith.index_cast %arg5 : index to i32
    %7 = arith.addi %6, %c1_i32 : i32
    %8 = arith.index_cast %7 : i32 to index
    scf.for %arg6 = %c0 to %8 step %c1 {
      %12 = arith.index_cast %arg6 : index to i32
      %13 = arith.subi %c0_i32, %12 : i32
      %14 = arith.remsi %13, %arg0 : i32
      %15 = arith.sitofp %14 : i32 to f64
      %16 = arith.divf %15, %0 : f64
      %17 = arith.addf %16, %cst : f64
      memref.store %17, %arg1[%arg5, %arg6] : memref<?x?xf64>
    }
    %9 = arith.addi %6, %c1_i32 : i32
    %10 = arith.index_cast %arg0 : i32 to index
    %11 = arith.index_cast %9 : i32 to index
    scf.for %arg6 = %11 to %10 step %c1 {
      %12 = arith.subi %arg6, %11 : index
      %13 = arith.index_cast %9 : i32 to index
      %14 = arith.addi %13, %12 : index
      memref.store %cst_2, %arg1[%arg5, %14] : memref<?x?xf64>
    }
    memref.store %cst, %arg1[%arg5, %arg5] : memref<?x?xf64>
  }
  // manually modified to use dynamic allocation
  %arg0_cast = arith.index_cast %arg0 : i32 to index
  %alloc = memref.alloc(%arg0_cast, %arg0_cast) : memref<?x?xf64>
  %3 = arith.index_cast %arg0 : i32 to index
  scf.for %arg5 = %c0 to %3 step %c1 {
    %6 = arith.index_cast %arg0 : i32 to index
    scf.for %arg6 = %c0 to %6 step %c1 {
      memref.store %cst_2, %alloc[%arg5, %arg6] : memref<?x?xf64>
    }
  }
  %4 = arith.index_cast %arg0 : i32 to index
  scf.for %arg5 = %c0 to %4 step %c1 {
    %6 = arith.index_cast %arg0 : i32 to index
    scf.for %arg6 = %c0 to %6 step %c1 {
      %7 = arith.index_cast %arg0 : i32 to index
      scf.for %arg7 = %c0 to %7 step %c1 {
        %8 = memref.load %arg1[%arg6, %arg5] : memref<?x?xf64>
        %9 = memref.load %arg1[%arg7, %arg5] : memref<?x?xf64>
        %10 = arith.mulf %8, %9 : f64
        %11 = memref.load %alloc[%arg6, %arg7] : memref<?x?xf64>
        %12 = arith.addf %11, %10 : f64
        memref.store %12, %alloc[%arg6, %arg7] : memref<?x?xf64>
      }
    }
  }
  %5 = arith.index_cast %arg0 : i32 to index
  scf.for %arg5 = %c0 to %5 step %c1 {
    %6 = arith.index_cast %arg0 : i32 to index
    scf.for %arg6 = %c0 to %6 step %c1 {
      %7 = memref.load %alloc[%arg5, %arg6] : memref<?x?xf64>
      memref.store %7, %arg1[%arg5, %arg6] : memref<?x?xf64>
    }
  }
  memref.dealloc %alloc : memref<?x?xf64>
  return
}

func.func @ludcmp(%arg0: i32, %arg1: memref<?x?xf64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>, %arg4: memref<?xf64>) {
  %0 = arith.index_cast %arg0 : i32 to index
  %1 = arith.index_cast %arg0 : i32 to index
  %2 = arith.index_cast %arg0 : i32 to index
  %3 = arith.index_cast %arg0 : i32 to index
  %4 = arith.index_cast %arg0 : i32 to index
  %5 = arith.index_cast %arg0 : i32 to index
  %alloca = memref.alloca() : memref<f64>
  %6 = llvm.mlir.undef : f64
  affine.store %6, %alloca[] : memref<f64>
  %7 = arith.index_cast %arg0 : i32 to index
  affine.for %arg5 = 0 to %7 {
    affine.for %arg6 = 0 to #map(%arg5) {
      %8 = affine.load %arg1[%arg5, %arg6] : memref<?x?xf64>
      affine.store %8, %alloca[] : memref<f64>
      affine.for %arg7 = 0 to #map(%arg6) {
        %12 = affine.load %arg1[%arg5, %arg7] : memref<?x?xf64>
        %13 = affine.load %arg1[%arg7, %arg6] : memref<?x?xf64>
        %14 = arith.mulf %12, %13 : f64
        %15 = affine.load %alloca[] : memref<f64>
        %16 = arith.subf %15, %14 : f64
        affine.store %16, %alloca[] : memref<f64>
      }
      %9 = affine.load %alloca[] : memref<f64>
      %10 = affine.load %arg1[%arg6, %arg6] : memref<?x?xf64>
      %11 = arith.divf %9, %10 : f64
      affine.store %11, %arg1[%arg5, %arg6] : memref<?x?xf64>
    }
    affine.for %arg6 = #map(%arg5) to %7 {
      %8 = affine.load %arg1[%arg5, %arg6] : memref<?x?xf64>
      affine.store %8, %alloca[] : memref<f64>
      affine.for %arg7 = 0 to #map(%arg5) {
        %10 = affine.load %arg1[%arg5, %arg7] : memref<?x?xf64>
        %11 = affine.load %arg1[%arg7, %arg6] : memref<?x?xf64>
        %12 = arith.mulf %10, %11 : f64
        %13 = affine.load %alloca[] : memref<f64>
        %14 = arith.subf %13, %12 : f64
        affine.store %14, %alloca[] : memref<f64>
      }
      %9 = affine.load %alloca[] : memref<f64>
      affine.store %9, %arg1[%arg5, %arg6] : memref<?x?xf64>
    }
  }
  affine.for %arg5 = 0 to %7 {
    %8 = affine.load %arg2[%arg5] : memref<?xf64>
    affine.store %8, %alloca[] : memref<f64>
    affine.for %arg6 = 0 to #map(%arg5) {
      %10 = affine.load %arg1[%arg5, %arg6] : memref<?x?xf64>
      %11 = affine.load %arg4[%arg6] : memref<?xf64>
      %12 = arith.mulf %10, %11 : f64
      %13 = affine.load %alloca[] : memref<f64>
      %14 = arith.subf %13, %12 : f64
      affine.store %14, %alloca[] : memref<f64>
    }
    %9 = affine.load %alloca[] : memref<f64>
    affine.store %9, %arg4[%arg5] : memref<?xf64>
  }
  affine.for %arg5 = 0 to %0 {
    %8 = affine.load %arg4[-%arg5 + symbol(%1) - 1] : memref<?xf64>
    affine.store %8, %alloca[] : memref<f64>
    affine.for %arg6 = #map1(%arg5)[%3] to %7 {
      %12 = affine.load %arg1[-%arg5 + symbol(%2) - 1, %arg6] : memref<?x?xf64>
      %13 = affine.load %arg3[%arg6] : memref<?xf64>
      %14 = arith.mulf %12, %13 : f64
      %15 = affine.load %alloca[] : memref<f64>
      %16 = arith.subf %15, %14 : f64
      affine.store %16, %alloca[] : memref<f64>
    }
    %9 = affine.load %alloca[] : memref<f64>
    %10 = affine.load %arg1[-%arg5 + symbol(%4) - 1, -%arg5 + symbol(%4) - 1] : memref<?x?xf64>
    %11 = arith.divf %9, %10 : f64
    affine.store %11, %arg3[-%arg5 + symbol(%5) - 1] : memref<?xf64>
  }
  return
}
