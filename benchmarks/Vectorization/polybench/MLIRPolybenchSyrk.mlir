//===- MLIRPolybenchSyrk.mlir ---------------------------------------------===//
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
// This file provides initialization and kernel functions for the syrk
// Polybench benchmark. The MLIR code is generated with Polygeist and modified
// manually to run on different dataset sizes.
//
//===----------------------------------------------------------------------===//

#map = affine_map<(d0) -> (d0 + 1)>
func.func @syrk_init_array(%arg0: i32, %arg1: i32, %arg2: memref<?xf64>, %arg3: memref<?xf64>, %arg4: memref<?x?xf64>, %arg5: memref<?x?xf64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2_i32 = arith.constant 2 : i32
  %c1_i32 = arith.constant 1 : i32
  %cst = arith.constant 1.200000e+00 : f64
  %cst_0 = arith.constant 1.500000e+00 : f64
  affine.store %cst_0, %arg2[0] : memref<?xf64>
  affine.store %cst, %arg3[0] : memref<?xf64>
  %0 = arith.index_cast %arg0 : i32 to index
  scf.for %arg6 = %c0 to %0 step %c1 {
    %2 = arith.index_cast %arg6 : index to i32
    %3 = arith.index_cast %arg1 : i32 to index
    scf.for %arg7 = %c0 to %3 step %c1 {
      %4 = arith.index_cast %arg7 : index to i32
      %5 = arith.muli %2, %4 : i32
      %6 = arith.addi %5, %c1_i32 : i32
      %7 = arith.remsi %6, %arg0 : i32
      %8 = arith.sitofp %7 : i32 to f64
      %9 = arith.sitofp %arg0 : i32 to f64
      %10 = arith.divf %8, %9 : f64
      memref.store %10, %arg5[%arg6, %arg7] : memref<?x?xf64>
    }
  }
  %1 = arith.index_cast %arg0 : i32 to index
  scf.for %arg6 = %c0 to %1 step %c1 {
    %2 = arith.index_cast %arg6 : index to i32
    %3 = arith.index_cast %arg0 : i32 to index
    scf.for %arg7 = %c0 to %3 step %c1 {
      %4 = arith.index_cast %arg7 : index to i32
      %5 = arith.muli %2, %4 : i32
      %6 = arith.addi %5, %c2_i32 : i32
      %7 = arith.remsi %6, %arg1 : i32
      %8 = arith.sitofp %7 : i32 to f64
      %9 = arith.sitofp %arg1 : i32 to f64
      %10 = arith.divf %8, %9 : f64
      memref.store %10, %arg4[%arg6, %arg7] : memref<?x?xf64>
    }
  }
  return
}
func.func @syrk(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: f64, %arg4: memref<?x?xf64>, %arg5: memref<?x?xf64>) {
  %0 = arith.index_cast %arg1 : i32 to index
  %1 = arith.index_cast %arg0 : i32 to index
  affine.for %arg6 = 0 to %1 {
    affine.for %arg7 = 0 to #map(%arg6) {
      %2 = affine.load %arg4[%arg6, %arg7] : memref<?x?xf64>
      %3 = arith.mulf %2, %arg3 : f64
      affine.store %3, %arg4[%arg6, %arg7] : memref<?x?xf64>
    }
    affine.for %arg7 = 0 to %0 {
      affine.for %arg8 = 0 to #map(%arg6) {
        %2 = affine.load %arg5[%arg6, %arg7] : memref<?x?xf64>
        %3 = arith.mulf %arg2, %2 : f64
        %4 = affine.load %arg5[%arg8, %arg7] : memref<?x?xf64>
        %5 = arith.mulf %3, %4 : f64
        %6 = affine.load %arg4[%arg6, %arg8] : memref<?x?xf64>
        %7 = arith.addf %6, %5 : f64
        affine.store %7, %arg4[%arg6, %arg8] : memref<?x?xf64>
      }
    }
  }
  return
}
