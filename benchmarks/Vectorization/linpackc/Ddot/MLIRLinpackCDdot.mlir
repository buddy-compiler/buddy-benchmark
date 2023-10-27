//===- MLIRLinpackCDdot.mlir --------------------------------------===//
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
// This file provides the MLIR linpackc ddot function.
//
//===----------------------------------------------------------------------===//
func.func @mlir_linpackcddotrollTYPE_PLACEHOLDER(%n : i32, %dx: memref<?xTYPE_PLACEHOLDER>, %incx : i32, 
                                   %dy: memref<?xTYPE_PLACEHOLDER>, %incy : i32) -> TYPE_PLACEHOLDER {
  %c0 = arith.constant 0 : index
  %i0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : index
  %i1 = arith.constant 1 : i32
  %c4 = arith.constant 4 : index
  %f_0 = arith.constant 0.0 : TYPE_PLACEHOLDER
  %n_index = arith.index_cast %n : i32 to index
  %dtemp_0 = arith.constant 0.0 : TYPE_PLACEHOLDER

  %cond1 = arith.cmpi "sle", %n, %i0 : i32
  %cond2 = arith.cmpi "ne", %incx, %i1 : i32
  %cond3 = arith.cmpi "ne", %incy, %i1 : i32
  %cond4 = arith.ori %cond2, %cond3 : i1

  cf.cond_br %cond1, ^terminator0, ^continue0

  ^terminator0: 
    return %f_0 :TYPE_PLACEHOLDER

  ^continue0:
    cf.cond_br %cond4, ^continue1, ^continue2

  ^continue1:
    %ix = arith.constant 0 : i32
    %iy = arith.constant 0 : i32
    %cond6 = arith.cmpi "slt", %incx, %i0 : i32
    %cond7 = arith.cmpi "slt", %incy, %i0 : i32
    %ix_0 = scf.if %cond6 -> (i32) {
      %tmp = arith.subi %i1, %n : i32
      %ix_1 = arith.muli %tmp, %incx : i32
      scf.yield %ix_1 : i32
    } else {
      scf.yield %ix : i32
    }
    %iy_0 = scf.if %cond7 -> (i32) {
      %tmp = arith.subi %i1, %n : i32
      %iy_1 = arith.muli %tmp, %incy : i32
      scf.yield %iy_1 : i32
    } else{
      scf.yield %iy : i32
    }

    %incx_index = arith.index_cast %incx : i32 to index
    %incy_index = arith.index_cast %incy : i32 to index
    %ix_0_index = arith.index_cast %ix_0 : i32 to index
    %iy_0_index = arith.index_cast %iy_0 : i32 to index

    %ix_3, %iy_3, %dtemp_1 = scf.for %i_0 = %c0 to %n_index step %c1
    iter_args(%ix_4 = %ix_0_index, %iy_4 = %iy_0_index, %dtemp_iter_1 = %dtemp_0) -> (index, index, TYPE_PLACEHOLDER){
      %dx_val_0 = memref.load %dx[%ix_4] : memref<?xTYPE_PLACEHOLDER>
      %dy_val_0 = memref.load %dy[%iy_4] : memref<?xTYPE_PLACEHOLDER>
      %result_0 = arith.mulf %dx_val_0, %dy_val_0 : TYPE_PLACEHOLDER
      %dtemp_next_1 = arith.addf %dtemp_iter_1, %result_0 : TYPE_PLACEHOLDER
      %ix_2 = arith.addi %ix_4, %incx_index : index
      %iy_2 = arith.addi %iy_4, %incy_index : index
      scf.yield %ix_2, %iy_2, %dtemp_next_1 : index, index, TYPE_PLACEHOLDER
    }
    return %dtemp_1 : TYPE_PLACEHOLDER
    
  ^continue2:
    %dtemp_2 = scf.for %i_1 = %c0 to %n_index step %c1 
      iter_args(%dtemp_iter_2 = %dtemp_0) -> (TYPE_PLACEHOLDER) {
      %dx_val_1 = memref.load %dx[%i_1] : memref<?xTYPE_PLACEHOLDER>
      %dy_val_1 = memref.load %dy[%i_1] : memref<?xTYPE_PLACEHOLDER>
      %result_1 = arith.mulf %dx_val_1, %dy_val_1 : TYPE_PLACEHOLDER
      %dtemp_next_2 = arith.addf %dtemp_iter_2, %result_1 : TYPE_PLACEHOLDER 
      scf.yield %dtemp_next_2 : TYPE_PLACEHOLDER
    }
    return %dtemp_2 : TYPE_PLACEHOLDER
  
}


func.func @mlir_linpackcddotunrollTYPE_PLACEHOLDER(%n : i32, %dx: memref<?xTYPE_PLACEHOLDER>, %incx : i32, 
                                   %dy: memref<?xTYPE_PLACEHOLDER>, %incy : i32) -> TYPE_PLACEHOLDER {
  %c0 = arith.constant 0 : index
  %i0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : index
  %i1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %i5 = arith.constant 5 : i32
  %f_0 = arith.constant 0.0 : TYPE_PLACEHOLDER
  %n_index = arith.index_cast %n : i32 to index
  %dtemp_0 = arith.constant 0.0 : TYPE_PLACEHOLDER

  %cond1 = arith.cmpi "sle", %n, %i0 : i32
  %cond2 = arith.cmpi "ne", %incx, %i1 : i32
  %cond3 = arith.cmpi "ne", %incy, %i1 : i32
  %cond4 = arith.ori %cond2, %cond3 : i1

  cf.cond_br %cond1, ^terminator0, ^continue0

  ^terminator0: 
    return %f_0 :TYPE_PLACEHOLDER

  ^continue0:
    cf.cond_br %cond4, ^continue1, ^continue2

  ^continue1:
    %ix = arith.constant 0 : i32
    %iy = arith.constant 0 : i32
    %cond5 = arith.cmpi "slt", %incx, %i0 : i32
    %cond6 = arith.cmpi "slt", %incy, %i0 : i32
    %ix_0 = scf.if %cond5 -> (i32) {
      %tmp = arith.subi %i1, %n : i32
      %ix_1 = arith.muli %tmp, %incx : i32
      scf.yield %ix_1 : i32
    } else {
      scf.yield %ix : i32
    }
    %iy_0 = scf.if %cond6 -> (i32) {
      %tmp = arith.subi %i1, %n : i32
      %iy_1 = arith.muli %tmp, %incy : i32
      scf.yield %iy_1 : i32
    } else{
      scf.yield %iy : i32
    }

    %incx_index = arith.index_cast %incx : i32 to index
    %incy_index = arith.index_cast %incy : i32 to index
    %ix_0_index = arith.index_cast %ix_0 : i32 to index
    %iy_0_index = arith.index_cast %iy_0 : i32 to index

    %ix_3, %iy_3, %dtemp_1 = scf.for %i_0 = %c0 to %n_index step %c1
    iter_args(%ix_4 = %ix_0_index, %iy_4 = %iy_0_index, %dtemp_iter_1 = %dtemp_0) -> (index, index, TYPE_PLACEHOLDER){
      %dx_val_0 = memref.load %dx[%ix_4] : memref<?xTYPE_PLACEHOLDER>
      %dy_val_0 = memref.load %dy[%iy_4] : memref<?xTYPE_PLACEHOLDER>
      %result_0 = arith.mulf %dx_val_0, %dy_val_0 : TYPE_PLACEHOLDER
      %dtemp_next_1 = arith.addf %dtemp_iter_1, %result_0 : TYPE_PLACEHOLDER
      %ix_2 = arith.addi %ix_4, %incx_index : index
      %iy_2 = arith.addi %iy_4, %incy_index : index
      scf.yield %ix_2, %iy_2, %dtemp_next_1 : index, index, TYPE_PLACEHOLDER
    }
    return %dtemp_1 : TYPE_PLACEHOLDER
    
  ^continue2:
    %m = arith.remsi %n , %i5 : i32
    %m_index = arith.index_cast %incx : i32 to index
    %c5 = arith.index_cast %i5 : i32 to index
    %cond7 = arith.cmpi "ne", %m, %i0 : i32
    cf.cond_br %cond7, ^continue3, ^continue4
    ^continue3:
        %dtemp_2 = scf.for %i_1 = %c0 to %m_index step %c1 
        iter_args(%dtemp_iter_2 = %dtemp_0) -> (TYPE_PLACEHOLDER) {
            %dx_val_1 = memref.load %dx[%i_1] : memref<?xTYPE_PLACEHOLDER>
            %dy_val_1 = memref.load %dy[%i_1] : memref<?xTYPE_PLACEHOLDER>
            %result_1 = arith.mulf %dx_val_1, %dy_val_1 : TYPE_PLACEHOLDER
            %dtemp_next_2 = arith.addf %dtemp_iter_2, %result_1 : TYPE_PLACEHOLDER 
            scf.yield %dtemp_next_2 : TYPE_PLACEHOLDER
        }
        %cond8 =  arith.cmpi "slt", %n, %i5 : i32
        cf.cond_br %cond8, ^terminator1, ^continue4
    ^terminator1: 
      return %dtemp_2 : TYPE_PLACEHOLDER
    ^continue4:
      %dtemp_3 =scf.for %i_2 = %m_index to %n_index step %c5 
      iter_args(%dtemp_iter_3 = %dtemp_0) -> (TYPE_PLACEHOLDER) {
      %dx_val_2 = memref.load %dx[%i_2] : memref<?xTYPE_PLACEHOLDER>
      %dy_val_2 = memref.load %dy[%i_2] : memref<?xTYPE_PLACEHOLDER>
      %result_2 = arith.mulf %dx_val_2, %dy_val_2 : TYPE_PLACEHOLDER

      %i_2_1 = arith.addi %i_2, %c1 : index
      %dx_val_3 = memref.load %dx[%i_2_1] : memref<?xTYPE_PLACEHOLDER>
      %dy_val_3 = memref.load %dy[%i_2_1] : memref<?xTYPE_PLACEHOLDER>
      %result_3 = arith.mulf %dx_val_3, %dy_val_3 : TYPE_PLACEHOLDER

      %i_2_2 = arith.addi %i_2, %c2 : index
      %dx_val_4 = memref.load %dx[%i_2_2] : memref<?xTYPE_PLACEHOLDER>
      %dy_val_4 = memref.load %dy[%i_2_2] : memref<?xTYPE_PLACEHOLDER>
      %result_4 = arith.mulf %dx_val_4, %dy_val_4 : TYPE_PLACEHOLDER

      %i_2_3 = arith.addi %i_2, %c3 : index
      %dx_val_5 = memref.load %dx[%i_2_3] : memref<?xTYPE_PLACEHOLDER>
      %dy_val_5 = memref.load %dy[%i_2_3] : memref<?xTYPE_PLACEHOLDER>
      %result_5 = arith.mulf %dx_val_5, %dy_val_5 : TYPE_PLACEHOLDER

      %i_2_4 = arith.addi %i_2, %c4 : index
      %dx_val_6 = memref.load %dx[%i_2_4] : memref<?xTYPE_PLACEHOLDER>
      %dy_val_6 = memref.load %dy[%i_2_4] : memref<?xTYPE_PLACEHOLDER>
      %result_6 = arith.mulf %dx_val_6, %dy_val_6 : TYPE_PLACEHOLDER

      %dtemp_temp_1 = arith.addf %dtemp_iter_3, %result_2 : TYPE_PLACEHOLDER
      %dtemp_temp_2 = arith.addf %dtemp_temp_1, %result_3 : TYPE_PLACEHOLDER
      %dtemp_temp_3 = arith.addf %dtemp_temp_2, %result_4 : TYPE_PLACEHOLDER
      %dtemp_temp_4 = arith.addf %dtemp_temp_3, %result_5 : TYPE_PLACEHOLDER
      %dtemp_next_3 = arith.addf %dtemp_temp_4, %result_6 : TYPE_PLACEHOLDER

      scf.yield %dtemp_next_3 : TYPE_PLACEHOLDER
    }
    return %dtemp_3 : TYPE_PLACEHOLDER
}
