//===- MLIRLinpackCMatgen.mlir --------------------------------------===//
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
// This file provides the MLIR linpackc matgenf32 function.
//
//===----------------------------------------------------------------------===//
func.func @mlir_linpackcmatgenTYPE_PLACEHOLDER(%a : memref<?xTYPE_PLACEHOLDER>, %lda : index, %n : index, %b : memref<?xTYPE_PLACEHOLDER>, %norma : memref<1xTYPE_PLACEHOLDER>)
{
	%c0 = arith.constant 0 : index
  	%c1 = arith.constant 1 : index
	%c3125 = arith.constant 3125 : i32
	%c65536 = arith.constant 65536 : i32
	%c0.0 = arith.constant 0.0 :TYPE_PLACEHOLDER
	%c32768.0 = arith.constant 32768.0 : TYPE_PLACEHOLDER
	%c16384.0 = arith.constant 16384.0 : TYPE_PLACEHOLDER
	%init_0 = arith.constant 1325 : i32
	//*norma = 0.0;
	memref.store %c0.0, %norma[%c0] : memref<1xTYPE_PLACEHOLDER>
	scf.for %j_0 = %c0 to %n step %c1 
	iter_args(%init_iter1 = %init_0) ->(i32){	
		%init = scf.for %i_0 = %c0 to %n step %c1 
			iter_args(%init_iter2 = %init_iter1) ->(i32){
			%temp_a_index = arith.muli %lda, %j_0: index
			%a_index = arith.addi %temp_a_index, %i_0 : index
			// init = 3125*init % 65536;
			// a[lda*j+i] = (init - 32768.0)/16384.0;
			%temp_init = arith.muli %c3125 , %init_iter2 : i32
			%new_init = arith.remsi %temp_init, %c65536 : i32
			%new_init_TYPE_PLACEHOLDER = arith.sitofp %new_init : i32 to TYPE_PLACEHOLDER
			%temp_data = arith.subf %new_init_TYPE_PLACEHOLDER, %c32768.0 : TYPE_PLACEHOLDER
			%new_data = arith.divf %temp_data, %c16384.0 : TYPE_PLACEHOLDER
			memref.store %new_data, %a[%a_index] : memref<?xTYPE_PLACEHOLDER>
			//*norma = (a[lda*j+i] > *norma) ? a[lda*j+i] : *norma;
			%old_data = memref.load  %norma[%c0] : memref<1xTYPE_PLACEHOLDER>
			%cond_0 = arith.cmpf ugt, %new_data, %old_data :TYPE_PLACEHOLDER
			%max_data = arith.select %cond_0, %new_data, %old_data : TYPE_PLACEHOLDER
			memref.store  %max_data, %norma[%c0] : memref<1xTYPE_PLACEHOLDER>
			scf.yield %new_init : i32
		}
		scf.yield %init : i32
	}
	// for (i = 0; i < n; i++) {
    //       b[i] = 0.0;
	// }
	scf.for %i_1 = %c0 to %n step %c1 {
		memref.store %c0.0 , %b[%i_1] : memref<?xTYPE_PLACEHOLDER>
	}

	// for (j = 0; j < n; j++) {
	// 	for (i = 0; i < n; i++) {
	// 		b[i] = b[i] + a[lda*j+i];
	// 	}
	// }
	scf.for %j_2 = %c0 to %n step %c1 {
		scf.for %i_2 = %c0 to %n step %c1 {
			%temp_a_index_2 = arith.muli %lda, %j_2 : index
			%a_index_2 = arith.addi %temp_a_index_2, %i_2 : index
			%b_data = memref.load %b[%i_2] : memref<?xTYPE_PLACEHOLDER>
			%a_data = memref.load %a[%a_index_2] : memref<?xTYPE_PLACEHOLDER>
			%result = arith.addf %b_data, %a_data : TYPE_PLACEHOLDER
			memref.store %result , %b[%i_2] : memref<?xTYPE_PLACEHOLDER>
		}
	}

	return
}
