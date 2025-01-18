//===- MLIRPolybenchAtax.mlir ---------------------------------------------===//
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
// This file provides initialization and kernel functions for the atax Polybench
// benchmark. The MLIR code is generated with Polygeist and modified manually to
// run on different dataset sizes.
//
//===----------------------------------------------------------------------===//

func.func @atax_init_array(%arg0: i32, %arg1: i32, %arg2: memref<?x?xf64>, %arg3: memref<?xf64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c5_i32 = arith.constant 5 : i32
  %cst = arith.constant 1.000000e+00 : f64
  %0 = arith.sitofp %arg1 : i32 to f64
  %1 = arith.index_cast %arg1 : i32 to index
  scf.for %arg4 = %c0 to %1 step %c1 {
    %3 = arith.index_cast %arg4 : index to i32
    %4 = arith.sitofp %3 : i32 to f64
    %5 = arith.divf %4, %0 : f64
    %6 = arith.addf %5, %cst : f64
    memref.store %6, %arg3[%arg4] : memref<?xf64>
  }
  %2 = arith.index_cast %arg0 : i32 to index
  scf.for %arg4 = %c0 to %2 step %c1 {
    %3 = arith.index_cast %arg4 : index to i32
    %4 = arith.index_cast %arg1 : i32 to index
    scf.for %arg5 = %c0 to %4 step %c1 {
      %5 = arith.index_cast %arg5 : index to i32
      %6 = arith.addi %3, %5 : i32
      %7 = arith.remsi %6, %arg1 : i32
      %8 = arith.sitofp %7 : i32 to f64
      %9 = arith.muli %arg0, %c5_i32 : i32
      %10 = arith.sitofp %9 : i32 to f64
      %11 = arith.divf %8, %10 : f64
      memref.store %11, %arg2[%arg4, %arg5] : memref<?x?xf64>
    }
  }
  return
}

func.func @atax_kernel(%arg0: i32, %arg1: i32, %arg2: memref<?x?xf64>, %arg3: memref<?xf64>, %arg4: memref<?xf64>, %arg5: memref<?xf64>) {
  %cst = arith.constant 0.000000e+00 : f64
  %0 = arith.index_cast %arg1 : i32 to index
  affine.for %arg6 = 0 to %0 {
    affine.store %cst, %arg4[%arg6] : memref<?xf64>
  }
  %1 = arith.index_cast %arg0 : i32 to index
  affine.for %arg6 = 0 to %1 {
    affine.store %cst, %arg5[%arg6] : memref<?xf64>
    affine.for %arg7 = 0 to %0 {
      %2 = affine.load %arg5[%arg6] : memref<?xf64>
      %3 = affine.load %arg2[%arg6, %arg7] : memref<?x?xf64>
      %4 = affine.load %arg3[%arg7] : memref<?xf64>
      %5 = arith.mulf %3, %4 : f64
      %6 = arith.addf %2, %5 : f64
      affine.store %6, %arg5[%arg6] : memref<?xf64>
    }
    affine.for %arg7 = 0 to %0 {
      %2 = affine.load %arg4[%arg7] : memref<?xf64>
      %3 = affine.load %arg2[%arg6, %arg7] : memref<?x?xf64>
      %4 = affine.load %arg5[%arg6] : memref<?xf64>
      %5 = arith.mulf %3, %4 : f64
      %6 = arith.addf %2, %5 : f64
      affine.store %6, %arg4[%arg7] : memref<?xf64>
    }
  }
  return
}
