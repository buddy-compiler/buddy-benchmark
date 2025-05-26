//===- MLIRPolybench3mm.mlir ----------------------------------------------===//
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
// This file provides initialization and kernel functions for the 3mm Polybench
// benchmark. The MLIR code is generated with Polygeist and modified manually to
// run on different dataset sizes.
//
//===----------------------------------------------------------------------===//

func.func @polybench_3mm_init_array(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: memref<?x?xf64>, %arg6: memref<?x?xf64>, %arg7: memref<?x?xf64>, %arg8: memref<?x?xf64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3_i32 = arith.constant 3 : i32
  %c2_i32 = arith.constant 2 : i32
  %c5_i32 = arith.constant 5 : i32
  %c1_i32 = arith.constant 1 : i32
  %0 = arith.index_cast %arg0 : i32 to index
  scf.for %arg9 = %c0 to %0 step %c1 {
    %4 = arith.index_cast %arg9 : index to i32
    %5 = arith.index_cast %arg2 : i32 to index
    scf.for %arg10 = %c0 to %5 step %c1 {
      %6 = arith.index_cast %arg10 : index to i32
      %7 = arith.muli %4, %6 : i32
      %8 = arith.addi %7, %c1_i32 : i32
      %9 = arith.remsi %8, %arg0 : i32
      %10 = arith.sitofp %9 : i32 to f64
      %11 = arith.muli %arg0, %c5_i32 : i32
      %12 = arith.sitofp %11 : i32 to f64
      %13 = arith.divf %10, %12 : f64
      memref.store %13, %arg5[%arg9, %arg10] : memref<?x?xf64>
    }
  }
  %1 = arith.index_cast %arg2 : i32 to index
  scf.for %arg9 = %c0 to %1 step %c1 {
    %4 = arith.index_cast %arg9 : index to i32
    %5 = arith.index_cast %arg1 : i32 to index
    scf.for %arg10 = %c0 to %5 step %c1 {
      %6 = arith.index_cast %arg10 : index to i32
      %7 = arith.addi %6, %c1_i32 : i32
      %8 = arith.muli %4, %7 : i32
      %9 = arith.addi %8, %c2_i32 : i32
      %10 = arith.remsi %9, %arg1 : i32
      %11 = arith.sitofp %10 : i32 to f64
      %12 = arith.muli %arg1, %c5_i32 : i32
      %13 = arith.sitofp %12 : i32 to f64
      %14 = arith.divf %11, %13 : f64
      memref.store %14, %arg6[%arg9, %arg10] : memref<?x?xf64>
    }
  }
  %2 = arith.index_cast %arg1 : i32 to index
  scf.for %arg9 = %c0 to %2 step %c1 {
    %4 = arith.index_cast %arg9 : index to i32
    %5 = arith.index_cast %arg4 : i32 to index
    scf.for %arg10 = %c0 to %5 step %c1 {
      %6 = arith.index_cast %arg10 : index to i32
      %7 = arith.addi %6, %c3_i32 : i32
      %8 = arith.muli %4, %7 : i32
      %9 = arith.remsi %8, %arg3 : i32
      %10 = arith.sitofp %9 : i32 to f64
      %11 = arith.muli %arg3, %c5_i32 : i32
      %12 = arith.sitofp %11 : i32 to f64
      %13 = arith.divf %10, %12 : f64
      memref.store %13, %arg7[%arg9, %arg10] : memref<?x?xf64>
    }
  }
  %3 = arith.index_cast %arg4 : i32 to index
  scf.for %arg9 = %c0 to %3 step %c1 {
    %4 = arith.index_cast %arg9 : index to i32
    %5 = arith.index_cast %arg3 : i32 to index
    scf.for %arg10 = %c0 to %5 step %c1 {
      %6 = arith.index_cast %arg10 : index to i32
      %7 = arith.addi %6, %c2_i32 : i32
      %8 = arith.muli %4, %7 : i32
      %9 = arith.addi %8, %c2_i32 : i32
      %10 = arith.remsi %9, %arg2 : i32
      %11 = arith.sitofp %10 : i32 to f64
      %12 = arith.muli %arg2, %c5_i32 : i32
      %13 = arith.sitofp %12 : i32 to f64
      %14 = arith.divf %11, %13 : f64
      memref.store %14, %arg8[%arg9, %arg10] : memref<?x?xf64>
    }
  }
  return
}

func.func @polybench_3mm_kernel(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: memref<?x?xf64>, %arg6: memref<?x?xf64>, %arg7: memref<?x?xf64>, %arg8: memref<?x?xf64>, %arg9: memref<?x?xf64>, %arg10: memref<?x?xf64>, %arg11: memref<?x?xf64>) {
  %cst = arith.constant 0.000000e+00 : f64
  %0 = arith.index_cast %arg1 : i32 to index
  %1 = arith.index_cast %arg2 : i32 to index
  %2 = arith.index_cast %arg4 : i32 to index
  %3 = arith.index_cast %arg3 : i32 to index
  %4 = arith.index_cast %arg0 : i32 to index
  affine.for %arg12 = 0 to %4 {
    affine.for %arg13 = 0 to %0 {
      affine.store %cst, %arg5[%arg12, %arg13] : memref<?x?xf64>
      affine.for %arg14 = 0 to %1 {
        %6 = affine.load %arg6[%arg12, %arg14] : memref<?x?xf64>
        %7 = affine.load %arg7[%arg14, %arg13] : memref<?x?xf64>
        %8 = arith.mulf %6, %7 : f64
        %9 = affine.load %arg5[%arg12, %arg13] : memref<?x?xf64>
        %10 = arith.addf %9, %8 : f64
        affine.store %10, %arg5[%arg12, %arg13] : memref<?x?xf64>
      }
    }
  }
  %5 = arith.index_cast %arg1 : i32 to index
  affine.for %arg12 = 0 to %5 {
    affine.for %arg13 = 0 to %3 {
      affine.store %cst, %arg8[%arg12, %arg13] : memref<?x?xf64>
      affine.for %arg14 = 0 to %2 {
        %6 = affine.load %arg9[%arg12, %arg14] : memref<?x?xf64>
        %7 = affine.load %arg10[%arg14, %arg13] : memref<?x?xf64>
        %8 = arith.mulf %6, %7 : f64
        %9 = affine.load %arg8[%arg12, %arg13] : memref<?x?xf64>
        %10 = arith.addf %9, %8 : f64
        affine.store %10, %arg8[%arg12, %arg13] : memref<?x?xf64>
      }
    }
  }
  affine.for %arg12 = 0 to %4 {
    affine.for %arg13 = 0 to %3 {
      affine.store %cst, %arg11[%arg12, %arg13] : memref<?x?xf64>
      affine.for %arg14 = 0 to %5 {
        %6 = affine.load %arg5[%arg12, %arg14] : memref<?x?xf64>
        %7 = affine.load %arg8[%arg14, %arg13] : memref<?x?xf64>
        %8 = arith.mulf %6, %7 : f64
        %9 = affine.load %arg11[%arg12, %arg13] : memref<?x?xf64>
        %10 = arith.addf %9, %8 : f64
        affine.store %10, %arg11[%arg12, %arg13] : memref<?x?xf64>
      }
    }
  }
  return
}
