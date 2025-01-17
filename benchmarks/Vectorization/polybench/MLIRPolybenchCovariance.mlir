//===- MLIRPolybenchCovariance.mlir ---------------------------------------===//
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
// This file provides initialization and kernel functions for the covariance
// Polybench benchmark. The MLIR code is generated with Polygeist and modified
// manually to run on different dataset sizes.
//
//===----------------------------------------------------------------------===//

#map = affine_map<(d0) -> (d0)>

func.func @covariance_init_array(%arg0: i32, %arg1: i32, %arg2: memref<?xf64>, %arg3: memref<?x?xf64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // manually modified to casting M to double
  %m_cast = arith.sitofp %arg0 : i32 to f64
  %0 = arith.sitofp %arg1 : i32 to f64
  affine.store %0, %arg2[0] : memref<?xf64>
  // manually modified to use parameteric loop bounds
  %arg0_cast = arith.index_cast %arg0 : i32 to index
  %arg1_cast = arith.index_cast %arg1 : i32 to index
  scf.for %arg4 = %c0 to %arg1_cast step %c1 {
    %1 = arith.index_cast %arg4 : index to i32
    scf.for %arg5 = %c0 to %arg0_cast step %c1 {
      %2 = arith.index_cast %arg5 : index to i32
      %3 = arith.sitofp %1 : i32 to f64
      %4 = arith.sitofp %2 : i32 to f64
      %5 = arith.mulf %3, %4 : f64
      %6 = arith.divf %5, %m_cast : f64
      memref.store %6, %arg3[%arg4, %arg5] : memref<?x?xf64>
    }
  }
  return
}

func.func @covariance(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: memref<?x?xf64>, %arg4: memref<?x?xf64>, %arg5: memref<?xf64>) {
  %cst = arith.constant 0.000000e+00 : f64
  %cst_0 = arith.constant 1.000000e+00 : f64
  %0 = arith.index_cast %arg1 : i32 to index
  %1 = arith.index_cast %arg0 : i32 to index
  affine.for %arg6 = 0 to %1 {
    affine.store %cst, %arg5[%arg6] : memref<?xf64>
    affine.for %arg7 = 0 to %0 {
      %6 = affine.load %arg3[%arg7, %arg6] : memref<?x?xf64>
      %7 = affine.load %arg5[%arg6] : memref<?xf64>
      %8 = arith.addf %7, %6 : f64
      affine.store %8, %arg5[%arg6] : memref<?xf64>
    }
    %4 = affine.load %arg5[%arg6] : memref<?xf64>
    %5 = arith.divf %4, %arg2 : f64
    affine.store %5, %arg5[%arg6] : memref<?xf64>
  }
  %2 = arith.index_cast %arg1 : i32 to index
  affine.for %arg6 = 0 to %2 {
    affine.for %arg7 = 0 to %1 {
      %4 = affine.load %arg5[%arg7] : memref<?xf64>
      %5 = affine.load %arg3[%arg6, %arg7] : memref<?x?xf64>
      %6 = arith.subf %5, %4 : f64
      affine.store %6, %arg3[%arg6, %arg7] : memref<?x?xf64>
    }
  }
  %3 = arith.subf %arg2, %cst_0 : f64
  affine.for %arg6 = 0 to %1 {
    affine.for %arg7 = #map(%arg6) to %1 {
      affine.store %cst, %arg4[%arg6, %arg7] : memref<?x?xf64>
      affine.for %arg8 = 0 to %2 {
        %6 = affine.load %arg3[%arg8, %arg6] : memref<?x?xf64>
        %7 = affine.load %arg3[%arg8, %arg7] : memref<?x?xf64>
        %8 = arith.mulf %6, %7 : f64
        %9 = affine.load %arg4[%arg6, %arg7] : memref<?x?xf64>
        %10 = arith.addf %9, %8 : f64
        affine.store %10, %arg4[%arg6, %arg7] : memref<?x?xf64>
      }
      %4 = affine.load %arg4[%arg6, %arg7] : memref<?x?xf64>
      %5 = arith.divf %4, %3 : f64
      affine.store %5, %arg4[%arg6, %arg7] : memref<?x?xf64>
      affine.store %5, %arg4[%arg7, %arg6] : memref<?x?xf64>
    }
  }
  return
}
