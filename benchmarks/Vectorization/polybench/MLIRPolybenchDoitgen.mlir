//===- MLIRPolybenchDoitgen.mlir ------------------------------------------===//
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
// This file provides initialization and kernel functions for the doitgen
// Polybench benchmark. The MLIR code is generated with Polygeist and modified
// manually to run on different dataset sizes.
//
//===----------------------------------------------------------------------===//

func.func @doitgen_kernel(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: memref<?x?x?xf64>, %arg4: memref<?x?xf64>, %arg5: memref<?xf64>) {
  %cst = arith.constant 0.000000e+00 : f64
  %0 = arith.index_cast %arg1 : i32 to index
  %1 = arith.index_cast %arg2 : i32 to index
  %2 = arith.index_cast %arg0 : i32 to index
  affine.for %arg6 = 0 to %2 {
    affine.for %arg7 = 0 to %0 {
      affine.for %arg8 = 0 to %1 {
        affine.store %cst, %arg5[%arg8] : memref<?xf64>
        affine.for %arg9 = 0 to %1 {
          %3 = affine.load %arg3[%arg6, %arg7, %arg9] : memref<?x?x?xf64>
          %4 = affine.load %arg4[%arg9, %arg8] : memref<?x?xf64>
          %5 = arith.mulf %3, %4 : f64
          %6 = affine.load %arg5[%arg8] : memref<?xf64>
          %7 = arith.addf %6, %5 : f64
          affine.store %7, %arg5[%arg8] : memref<?xf64>
        }
      }
      affine.for %arg8 = 0 to %1 {
        %3 = affine.load %arg5[%arg8] : memref<?xf64>
        affine.store %3, %arg3[%arg6, %arg7, %arg8] : memref<?x?x?xf64>
      }
    }
  }
  return
}

func.func @doitgen_init_array(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: memref<?x?x?xf64>, %arg4: memref<?x?xf64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = arith.index_cast %arg0 : i32 to index
  scf.for %arg5 = %c0 to %0 step %c1 {
    %2 = arith.index_cast %arg5 : index to i32
    %3 = arith.index_cast %arg1 : i32 to index
    scf.for %arg6 = %c0 to %3 step %c1 {
      %4 = arith.index_cast %arg6 : index to i32
      %5 = arith.index_cast %arg2 : i32 to index
      scf.for %arg7 = %c0 to %5 step %c1 {
        %6 = arith.index_cast %arg7 : index to i32
        %7 = arith.muli %2, %4 : i32
        %8 = arith.addi %7, %6 : i32
        %9 = arith.remsi %8, %arg2 : i32
        %10 = arith.sitofp %9 : i32 to f64
        %11 = arith.sitofp %arg2 : i32 to f64
        %12 = arith.divf %10, %11 : f64
        memref.store %12, %arg3[%arg5, %arg6, %arg7] : memref<?x?x?xf64>
      }
    }
  }
  %1 = arith.index_cast %arg2 : i32 to index
  scf.for %arg5 = %c0 to %1 step %c1 {
    %2 = arith.index_cast %arg5 : index to i32
    %3 = arith.index_cast %arg2 : i32 to index
    scf.for %arg6 = %c0 to %3 step %c1 {
      %4 = arith.index_cast %arg6 : index to i32
      %5 = arith.muli %2, %4 : i32
      %6 = arith.remsi %5, %arg2 : i32
      %7 = arith.sitofp %6 : i32 to f64
      %8 = arith.sitofp %arg2 : i32 to f64
      %9 = arith.divf %7, %8 : f64
      memref.store %9, %arg4[%arg5, %arg6] : memref<?x?xf64>
    }
  }
  return
}
