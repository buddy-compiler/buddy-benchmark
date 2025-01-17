//===- MLIRPolybenchAdi.mlir ----------------------------------------------===//
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
// This file provides initialization and kernel functions for the adi Polybench
// benchmark. The MLIR code is generated with Polygeist and modified manually to
// run on different dataset sizes.
//
//===----------------------------------------------------------------------===//

#map = affine_map<()[s0] -> (s0 + 1)>
#map1 = affine_map<()[s0] -> (s0 - 1)>

func.func  @adi_init_array(%arg0: i32, %arg1: memref<?x?xf64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = arith.index_cast %arg0 : i32 to index
  scf.for %arg2 = %c0 to %0 step %c1 {
    %1 = arith.index_cast %arg2 : index to i32
    %2 = arith.index_cast %arg0 : i32 to index
    scf.for %arg3 = %c0 to %2 step %c1 {
      %3 = arith.index_cast %arg3 : index to i32
      %4 = arith.addi %1, %arg0 : i32
      %5 = arith.subi %4, %3 : i32
      %6 = arith.sitofp %5 : i32 to f64
      %7 = arith.sitofp %arg0 : i32 to f64
      %8 = arith.divf %6, %7 : f64
      memref.store %8, %arg1[%arg2, %arg3] : memref<?x?xf64>
    }
  }
  return
}

func.func  @adi(%arg0: i32, %arg1: i32, %arg2: memref<?x?xf64>, %arg3: memref<?x?xf64>, %arg4: memref<?x?xf64>, %arg5: memref<?x?xf64>) {
  %cst = arith.constant 1.000000e+00 : f64
  %cst_0 = arith.constant 2.000000e+00 : f64
  %cst_1 = arith.constant 0.000000e+00 : f64
  %0 = arith.index_cast %arg1 : i32 to index
  %1 = arith.index_cast %arg1 : i32 to index
  %2 = arith.index_cast %arg1 : i32 to index
  %3 = arith.index_cast %arg1 : i32 to index
  %4 = arith.index_cast %arg1 : i32 to index
  %5 = arith.index_cast %arg1 : i32 to index
  %6 = arith.index_cast %arg1 : i32 to index
  %7 = arith.index_cast %arg1 : i32 to index
  %8 = arith.index_cast %arg1 : i32 to index
  %9 = arith.index_cast %arg1 : i32 to index
  %10 = arith.index_cast %arg1 : i32 to index
  %11 = arith.index_cast %arg1 : i32 to index
  %12 = arith.index_cast %arg1 : i32 to index
  %13 = arith.index_cast %arg1 : i32 to index
  %14 = arith.index_cast %arg1 : i32 to index
  %15 = arith.index_cast %arg1 : i32 to index
  %16 = arith.sitofp %arg1 : i32 to f64
  %17 = arith.divf %cst, %16 : f64
  %18 = arith.sitofp %arg0 : i32 to f64
  %19 = arith.divf %cst, %18 : f64
  %20 = arith.mulf %19, %cst_0 : f64
  %21 = arith.mulf %17, %17 : f64
  %22 = arith.divf %20, %21 : f64
  %23 = arith.divf %19, %21 : f64
  %24 = arith.negf %22 : f64
  %25 = arith.divf %24, %cst_0 : f64
  %26 = arith.addf %22, %cst : f64
  %27 = arith.negf %23 : f64
  %28 = arith.divf %27, %cst_0 : f64
  %29 = arith.addf %23, %cst : f64
  %30 = arith.index_cast %arg0 : i32 to index
  %31 = arith.negf %25 : f64
  %32 = arith.negf %28 : f64
  %33 = arith.mulf %28, %cst_0 : f64
  %34 = arith.addf %33, %cst : f64
  %35 = arith.negf %28 : f64
  %36 = arith.negf %25 : f64
  %37 = arith.mulf %25, %cst_0 : f64
  %38 = arith.addf %37, %cst : f64
  affine.for %arg6 = 1 to #map()[%30] {
    affine.for %arg7 = 1 to #map1()[%0] {
      affine.store %cst, %arg3[0, %arg7] : memref<?x?xf64>
      affine.store %cst_1, %arg4[%arg7, 0] : memref<?x?xf64>
      %39 = affine.load %arg3[0, %arg7] : memref<?x?xf64>
      affine.store %39, %arg5[%arg7, 0] : memref<?x?xf64>
      affine.for %arg8 = 1 to #map1()[%1] {
        %40 = affine.load %arg4[%arg7, %arg8 - 1] : memref<?x?xf64>
        %41 = arith.mulf %25, %40 : f64
        %42 = arith.addf %41, %26 : f64
        %43 = arith.divf %31, %42 : f64
        affine.store %43, %arg4[%arg7, %arg8] : memref<?x?xf64>
        %44 = affine.load %arg2[%arg8, %arg7 - 1] : memref<?x?xf64>
        %45 = arith.mulf %32, %44 : f64
        %46 = affine.load %arg2[%arg8, %arg7] : memref<?x?xf64>
        %47 = arith.mulf %34, %46 : f64
        %48 = arith.addf %45, %47 : f64
        %49 = affine.load %arg2[%arg8, %arg7 + 1] : memref<?x?xf64>
        %50 = arith.mulf %28, %49 : f64
        %51 = arith.subf %48, %50 : f64
        %52 = affine.load %arg5[%arg7, %arg8 - 1] : memref<?x?xf64>
        %53 = arith.mulf %25, %52 : f64
        %54 = arith.subf %51, %53 : f64
        %55 = arith.mulf %25, %40 : f64
        %56 = arith.addf %55, %26 : f64
        %57 = arith.divf %54, %56 : f64
        affine.store %57, %arg5[%arg7, %arg8] : memref<?x?xf64>
      }
      affine.store %cst, %arg3[symbol(%2) - 1, %arg7] : memref<?x?xf64>
      affine.for %arg8 = 1 to #map1()[%3] {
        %40 = affine.load %arg4[%arg7, -%arg8 + symbol(%4) - 1] : memref<?x?xf64>
        %41 = affine.load %arg3[-%arg8 + symbol(%5), %arg7] : memref<?x?xf64>
        %42 = arith.mulf %40, %41 : f64
        %43 = affine.load %arg5[%arg7, -%arg8 + symbol(%6) - 1] : memref<?x?xf64>
        %44 = arith.addf %42, %43 : f64
        affine.store %44, %arg3[-%arg8 + symbol(%7) - 1, %arg7] : memref<?x?xf64>
      }
    }
    affine.for %arg7 = 1 to #map1()[%8] {
      affine.store %cst, %arg2[%arg7, 0] : memref<?x?xf64>
      affine.store %cst_1, %arg4[%arg7, 0] : memref<?x?xf64>
      %39 = affine.load %arg2[%arg7, 0] : memref<?x?xf64>
      affine.store %39, %arg5[%arg7, 0] : memref<?x?xf64>
      affine.for %arg8 = 1 to #map1()[%9] {
        %40 = affine.load %arg4[%arg7, %arg8 - 1] : memref<?x?xf64>
        %41 = arith.mulf %28, %40 : f64
        %42 = arith.addf %41, %29 : f64
        %43 = arith.divf %35, %42 : f64
        affine.store %43, %arg4[%arg7, %arg8] : memref<?x?xf64>
        %44 = affine.load %arg3[%arg7 - 1, %arg8] : memref<?x?xf64>
        %45 = arith.mulf %36, %44 : f64
        %46 = affine.load %arg3[%arg7, %arg8] : memref<?x?xf64>
        %47 = arith.mulf %38, %46 : f64
        %48 = arith.addf %45, %47 : f64
        %49 = affine.load %arg3[%arg7 + 1, %arg8] : memref<?x?xf64>
        %50 = arith.mulf %25, %49 : f64
        %51 = arith.subf %48, %50 : f64
        %52 = affine.load %arg5[%arg7, %arg8 - 1] : memref<?x?xf64>
        %53 = arith.mulf %28, %52 : f64
        %54 = arith.subf %51, %53 : f64
        %55 = arith.mulf %28, %40 : f64
        %56 = arith.addf %55, %29 : f64
        %57 = arith.divf %54, %56 : f64
        affine.store %57, %arg5[%arg7, %arg8] : memref<?x?xf64>
      }
      affine.store %cst, %arg2[%arg7, symbol(%10) - 1] : memref<?x?xf64>
      affine.for %arg8 = 1 to #map1()[%11] {
        %40 = affine.load %arg4[%arg7, -%arg8 + symbol(%12) - 1] : memref<?x?xf64>
        %41 = affine.load %arg2[%arg7, -%arg8 + symbol(%13)] : memref<?x?xf64>
        %42 = arith.mulf %40, %41 : f64
        %43 = affine.load %arg5[%arg7, -%arg8 + symbol(%14) - 1] : memref<?x?xf64>
        %44 = arith.addf %42, %43 : f64
        affine.store %44, %arg2[%arg7, -%arg8 + symbol(%15) - 1] : memref<?x?xf64>
      }
    }
  }
  return
}
