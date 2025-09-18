//===- MLIRPolybenchCorrelation.mlir --------------------------------------===//
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
// This file provides initialization and kernel functions for the correlation
// Polybench benchmark. The MLIR code is generated with Polygeist and modified
// manually to run on different dataset sizes.
//
//===----------------------------------------------------------------------===//

#map = affine_map<()[s0] -> (s0 - 1)>
#map1 = affine_map<(d0) -> (d0 + 1)>

func.func @correlation_init_array(%arg0: i32, %arg1: i32, %arg2: memref<?xf64>, %arg3: memref<?x?xf64>) {
	%c0 = arith.constant 0 : index
	%c1 = arith.constant 1 : index
	%cst = arith.constant 2.600000e+03 : f64
	%cst_0 = arith.constant 3.000000e+03 : f64
	affine.store %cst_0, %arg2[0] : memref<?xf64>
	// manually modified to use parameteric loop bounds
	%arg0_cast = arith.index_cast %arg0 : i32 to index
	%arg1_cast = arith.index_cast %arg1 : i32 to index
	scf.for %arg4 = %c0 to %arg1_cast step %c1 {
		%0 = arith.index_cast %arg4 : index to i32
		scf.for %arg5 = %c0 to %arg0_cast step %c1 {
			%1 = arith.index_cast %arg5 : index to i32
			%2 = arith.muli %0, %1 : i32
			%3 = arith.sitofp %2 : i32 to f64
			%4 = arith.divf %3, %cst : f64
			%5 = arith.sitofp %0 : i32 to f64
			%6 = arith.addf %4, %5 : f64
			memref.store %6, %arg3[%arg4, %arg5] : memref<?x?xf64>
		}
	}
	return
}

func.func @correlation_kernel(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: memref<?x?xf64>, %arg4: memref<?x?xf64>, %arg5: memref<?xf64>, %arg6: memref<?xf64>) {
	%cst = arith.constant 1.000000e-01 : f64
	%cst_0 = arith.constant 0.000000e+00 : f64
	%cst_1 = arith.constant 1.000000e+00 : f64
	%0 = arith.index_cast %arg1 : i32 to index
	%1 = arith.index_cast %arg0 : i32 to index
	%2 = arith.index_cast %arg0 : i32 to index
	%3 = arith.index_cast %arg0 : i32 to index
	affine.for %arg7 = 0 to %3 {
		affine.store %cst_0, %arg5[%arg7] : memref<?xf64>
		affine.for %arg8 = 0 to %0 {
			%8 = affine.load %arg3[%arg8, %arg7] : memref<?x?xf64>
			%9 = affine.load %arg5[%arg7] : memref<?xf64>
			%10 = arith.addf %9, %8 : f64
			affine.store %10, %arg5[%arg7] : memref<?xf64>
		}
		%6 = affine.load %arg5[%arg7] : memref<?xf64>
		%7 = arith.divf %6, %arg2 : f64
		affine.store %7, %arg5[%arg7] : memref<?xf64>
	}
	affine.for %arg7 = 0 to %3 {
		affine.store %cst_0, %arg6[%arg7] : memref<?xf64>
		affine.for %arg8 = 0 to %0 {
			%11 = affine.load %arg3[%arg8, %arg7] : memref<?x?xf64>
			%12 = affine.load %arg5[%arg7] : memref<?xf64>
			%13 = arith.subf %11, %12 : f64
			%14 = arith.mulf %13, %13 : f64
			%15 = affine.load %arg6[%arg7] : memref<?xf64>
			%16 = arith.addf %15, %14 : f64
			affine.store %16, %arg6[%arg7] : memref<?xf64>
		}
		%6 = affine.load %arg6[%arg7] : memref<?xf64>
		%7 = arith.divf %6, %arg2 : f64
		%8 = math.sqrt %7 : f64
		%9 = arith.cmpf ole, %8, %cst : f64
		%10 = arith.select %9, %cst_1, %8 : f64
		affine.store %10, %arg6[%arg7] : memref<?xf64>
	}
	%4 = arith.index_cast %arg1 : i32 to index
	%5 = math.sqrt %arg2 : f64
	affine.for %arg7 = 0 to %4 {
		affine.for %arg8 = 0 to %3 {
			%6 = affine.load %arg5[%arg8] : memref<?xf64>
			%7 = affine.load %arg3[%arg7, %arg8] : memref<?x?xf64>
			%8 = arith.subf %7, %6 : f64
			affine.store %8, %arg3[%arg7, %arg8] : memref<?x?xf64>
			%9 = affine.load %arg6[%arg8] : memref<?xf64>
			%10 = arith.mulf %5, %9 : f64
			%11 = arith.divf %8, %10 : f64
			affine.store %11, %arg3[%arg7, %arg8] : memref<?x?xf64>
		}
	}
	affine.for %arg7 = 0 to #map()[%1] {
		affine.store %cst_1, %arg4[%arg7, %arg7] : memref<?x?xf64>
		affine.for %arg8 = #map1(%arg7) to %3 {
			affine.store %cst_0, %arg4[%arg7, %arg8] : memref<?x?xf64>
			affine.for %arg9 = 0 to %4 {
				%7 = affine.load %arg3[%arg9, %arg7] : memref<?x?xf64>
				%8 = affine.load %arg3[%arg9, %arg8] : memref<?x?xf64>
				%9 = arith.mulf %7, %8 : f64
				%10 = affine.load %arg4[%arg7, %arg8] : memref<?x?xf64>
				%11 = arith.addf %10, %9 : f64
				affine.store %11, %arg4[%arg7, %arg8] : memref<?x?xf64>
			}
			%6 = affine.load %arg4[%arg7, %arg8] : memref<?x?xf64>
			affine.store %6, %arg4[%arg8, %arg7] : memref<?x?xf64>
		}
	}
	affine.store %cst_1, %arg4[symbol(%2) - 1, symbol(%2) - 1] : memref<?x?xf64>
	return
}
