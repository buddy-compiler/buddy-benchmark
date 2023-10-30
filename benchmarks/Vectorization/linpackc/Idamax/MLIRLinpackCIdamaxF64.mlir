//===- MLIRLinpackCIdamaxF32.mlir -----------------------------------------===//
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
// This file provides the MLIR linpackc idamaxf32 function.
//
//===----------------------------------------------------------------------===//
func.func @mlir_linpackcidamaxf64(%n : i32,  %dx: memref<?xf64>, %incx : i32) -> i32
{
  %i0 = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %i1 = arith.constant 1 : i32
  %n_index = arith.index_cast %n : i32 to index
  %incx_index = arith.index_cast %incx : i32 to index

  %cond1 = arith.cmpi "sle", %n, %i1 : i32
  %cond2 = arith.cmpi "eq", %n, %i1 : i32
  %cond3 = arith.cmpi "ne", %incx, %i1 : i32

  cf.cond_br %cond1, ^terminator0, ^continue0

  ^terminator0:
    %in1 = arith.constant -1 : i32
    return %in1 : i32

  ^continue0:
    cf.cond_br %cond2, ^terminator1, ^continue1
  
  ^terminator1:
    return %i0 : i32

  ^continue1:
    cf.cond_br %cond3, ^continue2, ^continue3
// 	if(incx != 1) {

// 		/* code for increment not equal to 1 */

// 		ix = 0;
// 		dmax = fabs((double)dx[0]);
// 		ix = ix + incx;
// 		for (i = 1; i < n; i++) {
// 			if(fabs((double)dx[ix]) > dmax)  {
// 				itemp = i;
// 				dmax = fabs((double)dx[ix]);
// 			}
// 			ix = ix + incx;
// 		}
// 	}
  ^continue2:
    %ix_0 = arith.constant 0 : i32
    %dx_double = memref.load %dx[%c0] : memref<?xf64>
    %dmax_0 = math.absf %dx_double : f64
    %ix_1 = arith.addi %ix_0, %incx : i32
    %ix_1_index = arith.index_cast %ix_1 : i32 to index
    %itemp_0 = arith.constant 0 : i32
    %itemp_0_index = arith.constant 0 : index
    %dmax, %ix, %itemp = scf.for %i_0 = %c1 to %n_index step %c1
    iter_args(%dmax_iter = %dmax_0, %ix_iter = %ix_1_index, %itemp_iter = %itemp_0_index) -> (f64, index, index) {
      %dx_1_double = memref.load %dx[%ix_iter] : memref<?xf64>
      %dx_1_double_abs = math.absf %dx_1_double : f64
      %cond4 = arith.cmpf "ugt", %dx_1_double_abs, %dmax_iter : f64
      %itemp_next, %dmax_next = scf.if %cond4 -> (index, f64){
        scf.yield %i_0, %dx_1_double_abs : index, f64
      } else {
        scf.yield %itemp_iter, %dmax_iter : index, f64
      }
      %ix_next = arith.addi %ix_iter, %incx_index : index
      scf.yield %dmax_next, %ix_next , %itemp_next: f64, index, index
      }
      %itemp_i32 = arith.index_cast %itemp :  index to i32
      return %itemp_i32 : i32
  ^continue3:
    // else {
// 		/* code for increment equal to 1 */
// 		itemp = 0;
// 		dmax = fabs((double)dx[0]);
// 		for (i = 1; i < n; i++) {
// 			if(fabs((double)dx[i]) > dmax) {
// 				itemp = i;
// 				dmax = fabs((double)dx[i]);
// 			}
// 		}
// 	}
// 	return (itemp);
    %itemp_0_1 = arith.constant 0 : i32
    %itemp_0_1_index = arith.constant 0 : index
    %dx_0_1_double = memref.load %dx[%c0] : memref<?xf64>
    %dmax_0_1 = math.absf %dx_0_1_double : f64
    %dmax_1, %itemp_1 = scf.for %i_0_1 = %c1 to %n_index step %c1
    iter_args(%dmax_iter_1 = %dmax_0_1, %itemp_iter_1 = %itemp_0_1_index) -> (f64, index) {
      %dx_1_1_double = memref.load %dx[%i_0_1] : memref<?xf64>
      %dx_1_1_double_abs = math.absf %dx_1_1_double : f64
      %cond4 = arith.cmpf "ugt", %dx_1_1_double_abs, %dmax_iter_1 : f64
      %itemp_next_1, %dmax_next_1 = scf.if %cond4 -> (index, f64){
        scf.yield %i_0_1, %dx_1_1_double_abs : index, f64
      } else {
        scf.yield %itemp_iter_1, %dmax_iter_1 : index, f64
      }
      scf.yield %dmax_next_1, %itemp_next_1: f64, index
      }
      %itemp_1_i32 = arith.index_cast %itemp_1 : index to i32
      return %itemp_1_i32 : i32
}
