//===- MLIRPolybenchBicg.mlir ---------------------------------------------===//
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
// This file provides initialization and kernel functions for the bicg Polybench
// benchmark. The MLIR code is generated with Polygeist and modified manually to
// run on different dataset sizes.
//
//===----------------------------------------------------------------------===//

func.func @bicg_init_array(%arg0: i32, %arg1: i32, %arg2: memref<?x?xf64>, %arg3: memref<?xf64>, %arg4: memref<?xf64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c1_i32 = arith.constant 1 : i32
  %0 = arith.index_cast %arg0 : i32 to index
  scf.for %arg5 = %c0 to %0 step %c1 {
    %2 = arith.index_cast %arg5 : index to i32
    %3 = arith.remsi %2, %arg0 : i32
    %4 = arith.sitofp %3 : i32 to f64
    %5 = arith.sitofp %arg0 : i32 to f64
    %6 = arith.divf %4, %5 : f64
    memref.store %6, %arg4[%arg5] : memref<?xf64>
  }
  %1 = arith.index_cast %arg1 : i32 to index
  scf.for %arg5 = %c0 to %1 step %c1 {
    %2 = arith.index_cast %arg5 : index to i32
    %3 = arith.remsi %2, %arg1 : i32
    %4 = arith.sitofp %3 : i32 to f64
    %5 = arith.sitofp %arg1 : i32 to f64
    %6 = arith.divf %4, %5 : f64
    memref.store %6, %arg3[%arg5] : memref<?xf64>
    %7 = arith.index_cast %arg0 : i32 to index
    scf.for %arg6 = %c0 to %7 step %c1 {
      %8 = arith.index_cast %arg6 : index to i32
      %9 = arith.addi %8, %c1_i32 : i32
      %10 = arith.muli %2, %9 : i32
      %11 = arith.remsi %10, %arg1 : i32
      %12 = arith.sitofp %11 : i32 to f64
      %13 = arith.divf %12, %5 : f64
      memref.store %13, %arg2[%arg5, %arg6] : memref<?x?xf64>
    }
  }
  return
}
func.func @bicg_kernel(%arg0: i32, %arg1: i32, %arg2: memref<?x?xf64>, %arg3: memref<?xf64>, %arg4: memref<?xf64>, %arg5: memref<?xf64>, %arg6: memref<?xf64>) {
  %cst = arith.constant 0.000000e+00 : f64
  %0 = arith.index_cast %arg0 : i32 to index
  affine.for %arg7 = 0 to %0 {
    affine.store %cst, %arg3[%arg7] : memref<?xf64>
  }
  %1 = arith.index_cast %arg1 : i32 to index
  affine.for %arg7 = 0 to %1 {
    affine.store %cst, %arg4[%arg7] : memref<?xf64>
    affine.for %arg8 = 0 to %0 {
      %2 = affine.load %arg3[%arg8] : memref<?xf64>
      %3 = affine.load %arg6[%arg7] : memref<?xf64>
      %4 = affine.load %arg2[%arg7, %arg8] : memref<?x?xf64>
      %5 = arith.mulf %3, %4 : f64
      %6 = arith.addf %2, %5 : f64
      affine.store %6, %arg3[%arg8] : memref<?xf64>
      %7 = affine.load %arg4[%arg7] : memref<?xf64>
      %8 = affine.load %arg2[%arg7, %arg8] : memref<?x?xf64>
      %9 = affine.load %arg5[%arg8] : memref<?xf64>
      %10 = arith.mulf %8, %9 : f64
      %11 = arith.addf %7, %10 : f64
      affine.store %11, %arg4[%arg7] : memref<?xf64>
    }
  }
  return
}
