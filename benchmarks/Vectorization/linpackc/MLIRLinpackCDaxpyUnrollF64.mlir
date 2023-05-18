//===- MLIRLinpackCDaxpyUnrollF64.mlir ------------------------------------===//
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


func.func @mlir_linpackcdaxpyunrollf64(%n : index, %da : f64, %dx: memref<?xf64>, %incx : index, 
                                   %dy: memref<?xf64>, %incy : index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %da_i = arith.fptosi %da : f64 to i64
  %da_index = arith.index_cast %da_i : i64 to index
  %m = arith.remui %n, %c4 : index

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
    %cond8 = arith.cmpi "ne", %m, %c0 : index
    scf.if %cond8 {
      scf.for %i_1 = %c0 to %m step %c1 {
        %dx_val_1 = memref.load %dx[%i_1] : memref<?xf64>
        %dy_val_1 = memref.load %dy[%i_1] : memref<?xf64>
        %result_1 = arith.mulf %da, %dx_val_1 : f64
        %new_dy_val_1 = arith.addf %dy_val_1, %result_1 : f64
        memref.store %new_dy_val_1, %dy[%i_1] : memref<?xf64>
      }
    }

  %cond9 = arith.cmpi "slt", %n, %c4 : index
  cf.cond_br %cond9, ^terminator, ^continue4

  ^continue4:
    scf.for %i_2 = %m to %n step %c4 {
      %dx_val_2 = memref.load %dx[%i_2] : memref<?xf64>
      %dy_val_2 = memref.load %dy[%i_2] : memref<?xf64>
      %result_2 = arith.mulf %da, %dx_val_2 : f64
      %new_dy_val_2 = arith.addf %dy_val_2, %result_2 : f64
      memref.store %new_dy_val_2, %dy[%i_2] : memref<?xf64>

      %i_2_1 = arith.addi %i_2, %c1 : index
      %dx_val_3 = memref.load %dx[%i_2_1] : memref<?xf64>
      %dy_val_3 = memref.load %dy[%i_2_1] : memref<?xf64>
      %result_3 = arith.mulf %da, %dx_val_3 : f64
      %new_dy_val_3 = arith.addf %dy_val_3, %result_3 : f64
      memref.store %new_dy_val_3, %dy[%i_2_1] : memref<?xf64>

      %i_2_2 = arith.addi %i_2, %c2 : index
      %dx_val_4 = memref.load %dx[%i_2_2] : memref<?xf64>
      %dy_val_4 = memref.load %dy[%i_2_2] : memref<?xf64>
      %result_4 = arith.mulf %da, %dx_val_4 : f64
      %new_dy_val_4 = arith.addf %dy_val_4, %result_4 : f64
      memref.store %new_dy_val_4, %dy[%i_2_2] : memref<?xf64>

      %i_2_3 = arith.addi %i_2, %c3 : index
      %dx_val_5 = memref.load %dx[%i_2_3] : memref<?xf64>
      %dy_val_5 = memref.load %dy[%i_2_3] : memref<?xf64>
      %result_5 = arith.mulf %da, %dx_val_5 : f64
      %new_dy_val_5 = arith.addf %dy_val_5, %result_5 : f64
      memref.store %new_dy_val_5, %dy[%i_2_3] : memref<?xf64>
    }
    return
    
  ^terminator:
    return
}
