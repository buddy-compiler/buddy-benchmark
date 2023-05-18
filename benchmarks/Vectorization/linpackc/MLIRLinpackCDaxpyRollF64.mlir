//===- MLIRLinpackCDaxpyRollF64.mlir --------------------------------------===//
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
// This file provides the MLIR linpackc daxpy function.
//
//===----------------------------------------------------------------------===//


func.func @mlir_linpackcdaxpyrollf64(%n : index, %da : f64, %dx: memref<?xf64>, %incx : index, 
                                   %dy: memref<?xf64>, %incy : index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %da_i = arith.fptosi %da : f64 to i64
  %da_index = arith.index_cast %da_i : i64 to index

  %cond1 = arith.cmpi "sle", %n, %c0 : index
  %cond2 = arith.cmpi "eq", %da_index, %c0 : index
  %cond3 = arith.cmpi "ne", %incx, %c1 : index
  %cond4 = arith.cmpi "ne", %incy, %c1 : index
  %cond5 = arith.ori %cond3, %cond4 : i1
  
  cf.cond_br %cond1, ^terminator, ^continue0
  ^continue0:
    cf.cond_br %cond2, ^terminator, ^continue1

  ^continue1:
    cf.cond_br %cond5, ^continue2, ^continue3

  ^continue2:
    %ix = arith.constant 0 : index
    %iy = arith.constant 0 : index
    %cond6 = arith.cmpi "slt", %incx, %c0 : index
    %cond7 = arith.cmpi "slt", %incy, %c0 : index
    %ix_0 = scf.if %cond6 -> (index) {
      %tmp = arith.subi %c1, %n : index
      %ix_1 = arith.muli %tmp, %incx : index
      scf.yield %ix_1 : index
    } else {
      scf.yield %ix : index
    }
    %iy_0 = scf.if %cond7 -> (index) {
      %tmp = arith.subi %c1, %n : index
      %iy_1 = arith.muli %tmp, %incy : index
      scf.yield %iy_1 : index
    } else{
      scf.yield %iy : index
    }
    scf.for %i_0 = %c0 to %n step %c1 {
      %dx_val_0 = memref.load %dx[%ix_0] : memref<?xf64>
      %dy_val_0 = memref.load %dy[%iy_0] : memref<?xf64>
      %result_0 = arith.mulf %da, %dx_val_0 : f64
      %new_dy_val_0 = arith.addf %dy_val_0, %result_0 : f64
      memref.store %new_dy_val_0, %dy[%iy_0] : memref<?xf64>
      %ix_2 = arith.addi %ix, %incx : index
      %iy_2 = arith.addi %iy, %incy : index
    }
    return

  ^continue3:
    scf.for %i_1 = %c0 to %n step %c1 {
      %dx_val_1 = memref.load %dx[%i_1] : memref<?xf64>
      %dy_val_1 = memref.load %dy[%i_1] : memref<?xf64>
      %result_1 = arith.mulf %da, %dx_val_1 : f64
      %new_dy_val_1 = arith.addf %dy_val_1, %result_1 : f64
      memref.store %new_dy_val_1, %dy[%i_1] : memref<?xf64>
    }
    return
  
  ^terminator:
    return
}
