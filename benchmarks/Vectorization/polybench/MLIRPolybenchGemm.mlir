//===- MLIRPolybenchGemm.mlir ---------------------------------------------===//
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
// This file provides initialization and kernel functions for the gemm Polybench
// benchmark. The MLIR code is generated with Polygeist and modified manually to
// run on different dataset sizes.
//
//===----------------------------------------------------------------------===//

func.func @gemm_init_array(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: memref<?xf64>, %arg4: memref<?xf64>, %arg5: memref<?x?xf64>, %arg6: memref<?x?xf64>, %arg7: memref<?x?xf64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2_i32 = arith.constant 2 : i32
  %c1_i32 = arith.constant 1 : i32
  %cst = arith.constant 1.200000e+00 : f64
  %cst_0 = arith.constant 1.500000e+00 : f64
  affine.store %cst_0, %arg3[0] : memref<?xf64>
  affine.store %cst, %arg4[0] : memref<?xf64>
  %0 = arith.index_cast %arg0 : i32 to index
  scf.for %arg8 = %c0 to %0 step %c1 {
    %3 = arith.index_cast %arg8 : index to i32
    %4 = arith.index_cast %arg1 : i32 to index
    scf.for %arg9 = %c0 to %4 step %c1 {
      %5 = arith.index_cast %arg9 : index to i32
      %6 = arith.muli %3, %5 : i32
      %7 = arith.addi %6, %c1_i32 : i32
      %8 = arith.remsi %7, %arg0 : i32
      %9 = arith.sitofp %8 : i32 to f64
      %10 = arith.sitofp %arg0 : i32 to f64
      %11 = arith.divf %9, %10 : f64
      memref.store %11, %arg5[%arg8, %arg9] : memref<?x?xf64>
    }
  }
  %1 = arith.index_cast %arg0 : i32 to index
  scf.for %arg8 = %c0 to %1 step %c1 {
    %3 = arith.index_cast %arg8 : index to i32
    %4 = arith.index_cast %arg2 : i32 to index
    scf.for %arg9 = %c0 to %4 step %c1 {
      %5 = arith.index_cast %arg9 : index to i32
      %6 = arith.addi %5, %c1_i32 : i32
      %7 = arith.muli %3, %6 : i32
      %8 = arith.remsi %7, %arg2 : i32
      %9 = arith.sitofp %8 : i32 to f64
      %10 = arith.sitofp %arg2 : i32 to f64
      %11 = arith.divf %9, %10 : f64
      memref.store %11, %arg6[%arg8, %arg9] : memref<?x?xf64>
    }
  }
  %2 = arith.index_cast %arg2 : i32 to index
  scf.for %arg8 = %c0 to %2 step %c1 {
    %3 = arith.index_cast %arg8 : index to i32
    %4 = arith.index_cast %arg1 : i32 to index
    scf.for %arg9 = %c0 to %4 step %c1 {
      %5 = arith.index_cast %arg9 : index to i32
      %6 = arith.addi %5, %c2_i32 : i32
      %7 = arith.muli %3, %6 : i32
      %8 = arith.remsi %7, %arg1 : i32
      %9 = arith.sitofp %8 : i32 to f64
      %10 = arith.sitofp %arg1 : i32 to f64
      %11 = arith.divf %9, %10 : f64
      memref.store %11, %arg7[%arg8, %arg9] : memref<?x?xf64>
    }
  }
  return
}

func.func @gemm_kernel(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: f64, %arg4: f64, %arg5: memref<?x?xf64>, %arg6: memref<?x?xf64>, %arg7: memref<?x?xf64>) {
  %0 = arith.index_cast %arg1 : i32 to index
  %1 = arith.index_cast %arg2 : i32 to index
  %2 = arith.index_cast %arg0 : i32 to index
  affine.for %arg8 = 0 to %2 {
    affine.for %arg9 = 0 to %0 {
      %3 = affine.load %arg5[%arg8, %arg9] : memref<?x?xf64>
      %4 = arith.mulf %3, %arg4 : f64
      affine.store %4, %arg5[%arg8, %arg9] : memref<?x?xf64>
    }
    affine.for %arg9 = 0 to %1 {
      affine.for %arg10 = 0 to %0 {
        %3 = affine.load %arg6[%arg8, %arg9] : memref<?x?xf64>
        %4 = arith.mulf %arg3, %3 : f64
        %5 = affine.load %arg7[%arg9, %arg10] : memref<?x?xf64>
        %6 = arith.mulf %4, %5 : f64
        %7 = affine.load %arg5[%arg8, %arg10] : memref<?x?xf64>
        %8 = arith.addf %7, %6 : f64
        affine.store %8, %arg5[%arg8, %arg10] : memref<?x?xf64>
      }
    }
  }
  return
}
