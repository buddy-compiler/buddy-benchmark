//===- MLIRLinpackCDscal.mlir ---------------------------------------------===//
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
// This file provides the MLIR linpackc dscal function.
//
//===----------------------------------------------------------------------===//

func.func @mlir_linpackcdscalrollTYPE_PLACEHOLDER(%n : i32, %da : TYPE_PLACEHOLDER, %dx: memref<?xTYPE_PLACEHOLDER>, %incx : i32) 
{
  %i0 = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %i1 = arith.constant 1 : i32
  %c4 = arith.constant 4 : index
  %da_i = arith.fptosi %da : TYPE_PLACEHOLDER to i32
  %n_index = arith.index_cast %n : i32 to index
  %incx_index = arith.index_cast %incx : i32 to index

  %cond1 = arith.cmpi "sle", %n, %i0 : i32
  %cond2 = arith.cmpi "ne", %incx, %i1 : i32

  cf.cond_br %cond1, ^terminator, ^continue0
  
  ^continue0:
    cf.cond_br %cond2, ^continue1, ^continue2

  ^continue1:
// 		nincx = n*incx;
// 		for (i = 0; i < nincx; i = i + incx)
// 			dx[i] = da*dx[i];
// 		return;
    %nincx = arith.muli %n , %incx : i32
    %nincx_index = arith.index_cast %nincx : i32 to index
    scf.for %i_0 = %c0 to %nincx_index step %incx_index
    {
      %dx_val_0 = memref.load %dx[%i_0] : memref<?xTYPE_PLACEHOLDER>
      %result_0 = arith.mulf %da, %dx_val_0 : TYPE_PLACEHOLDER
      memref.store %result_0, %dx[%i_0] : memref<?xTYPE_PLACEHOLDER>
    }
    return

  ^continue2:
// #ifdef ROLL
// 	for (i = 0; i < n; i++)
// 		dx[i] = da*dx[i];
// #endif
    scf.for %i_1 = %c0 to %n_index step %c1 {
      %dx_val_1 = memref.load %dx[%i_1] : memref<?xTYPE_PLACEHOLDER>
      %result_1 = arith.mulf %da, %dx_val_1 : TYPE_PLACEHOLDER
      memref.store %result_1, %dx[%i_1] : memref<?xTYPE_PLACEHOLDER>
    }
    return
  
  ^terminator:
    return
}

func.func @mlir_linpackcdscalunrollTYPE_PLACEHOLDER(%n : i32, %da : TYPE_PLACEHOLDER, %dx: memref<?xTYPE_PLACEHOLDER>, %incx : i32) 
{
  %i0 = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %i1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %i5 = arith.constant 5 : i32
  %da_i = arith.fptosi %da : TYPE_PLACEHOLDER to i32
  %n_index = arith.index_cast %n : i32 to index
  %incx_index = arith.index_cast %incx : i32 to index

  %cond1 = arith.cmpi "sle", %n, %i0 : i32
  %cond2 = arith.cmpi "ne", %incx, %i1 : i32

  cf.cond_br %cond1, ^terminator, ^continue0
  
  ^continue0:
    cf.cond_br %cond2, ^continue1, ^continue2

  ^continue1:
// 		nincx = n*incx;
// 		for (i = 0; i < nincx; i = i + incx)
// 			dx[i] = da*dx[i];
// 		return;
    %nincx = arith.muli %n , %incx : i32
    %nincx_index = arith.index_cast %nincx : i32 to index
    scf.for %i_0 = %c0 to %nincx_index step %incx_index
    {
      %dx_val_0 = memref.load %dx[%i_0] : memref<?xTYPE_PLACEHOLDER>
      %result_0 = arith.mulf %da, %dx_val_0 : TYPE_PLACEHOLDER
      memref.store %result_0, %dx[%i_0] : memref<?xTYPE_PLACEHOLDER>
    }
    return

  ^continue2:
// #ifdef UNROLL
// 	m = n % 5;
// 	if (m != 0) {
// 		for (i = 0; i < m; i++)
// 			dx[i] = da*dx[i];
// 		if (n < 5) return;
// 	}
// 	for (i = m; i < n; i = i + 5){
// 		dx[i] = da*dx[i];
// 		dx[i+1] = da*dx[i+1];
// 		dx[i+2] = da*dx[i+2];
// 		dx[i+3] = da*dx[i+3];
// 		dx[i+4] = da*dx[i+4];
// 	}
// #endif
    %m = arith.remsi %n , %i5 : i32
    %m_index = arith.index_cast %incx : i32 to index
    %c5 = arith.index_cast %i5 : i32 to index
    %cond7 = arith.cmpi "ne", %m, %i0 : i32
    cf.cond_br %cond7, ^continue3, ^continue4
    ^continue3:
        scf.for %i_1 = %c0 to %m_index step %c1 {
            %dx_val_1 = memref.load %dx[%i_1] : memref<?xTYPE_PLACEHOLDER>
            %result_1 = arith.mulf %dx_val_1, %da : TYPE_PLACEHOLDER
            memref.store %result_1, %dx[%i_1] : memref<?xTYPE_PLACEHOLDER>
        }
        %cond8 =  arith.cmpi "slt", %n, %i5 : i32
        cf.cond_br %cond8, ^terminator, ^continue4
 
    ^continue4:
      scf.for %i_2 = %m_index to %n_index step %c5 {
      %dx_val_2 = memref.load %dx[%i_2] : memref<?xTYPE_PLACEHOLDER>
      %result_2 = arith.mulf %dx_val_2, %da : TYPE_PLACEHOLDER
      memref.store %result_2, %dx[%i_2] : memref<?xTYPE_PLACEHOLDER>

      %i_2_1 = arith.addi %i_2, %c1 : index
      %dx_val_3 = memref.load %dx[%i_2_1] : memref<?xTYPE_PLACEHOLDER>
      %result_3 = arith.mulf %dx_val_3, %da : TYPE_PLACEHOLDER
      memref.store %result_3, %dx[%i_2_1] : memref<?xTYPE_PLACEHOLDER>

      %i_2_2 = arith.addi %i_2, %c2 : index
      %dx_val_4 = memref.load %dx[%i_2_2] : memref<?xTYPE_PLACEHOLDER>
      %result_4 = arith.mulf %dx_val_4, %da : TYPE_PLACEHOLDER
      memref.store %result_4, %dx[%i_2_2] : memref<?xTYPE_PLACEHOLDER>

      %i_2_3 = arith.addi %i_2, %c3 : index
      %dx_val_5 = memref.load %dx[%i_2_3] : memref<?xTYPE_PLACEHOLDER>
      %result_5 = arith.mulf %dx_val_5, %da : TYPE_PLACEHOLDER
      memref.store %result_5, %dx[%i_2_3] : memref<?xTYPE_PLACEHOLDER>

      %i_2_4 = arith.addi %i_2, %c4 : index
      %dx_val_6 = memref.load %dx[%i_2_4] : memref<?xTYPE_PLACEHOLDER>
      %result_6 = arith.mulf %dx_val_6, %da : TYPE_PLACEHOLDER
      memref.store %result_6, %dx[%i_2_4] : memref<?xTYPE_PLACEHOLDER>
    }
    return
  ^terminator:
    return
}
