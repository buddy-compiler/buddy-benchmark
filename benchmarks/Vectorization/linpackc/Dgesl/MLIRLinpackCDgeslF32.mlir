//===- MLIRLinpackCDgeslF32.mlir ------------------------------------------===//
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
// This file provides the MLIR linpackc dgesl function.
//
//===----------------------------------------------------------------------===//
module{
func.func private @mlir_linpackcdaxpyrollf32(%n : i32, %da : f32, %dx: memref<?xf32>, %incx : i32,
                                   %dy: memref<?xf32>, %incy : i32)-> ()
func.func private @mlir_linpackcddotrollf32(%n : i32, %dx: memref<?xf32>, %incx : i32, 
                                   %dy: memref<?xf32>, %incy : i32) -> f32                                    
func.func private @mlir_linpackcdaxpyunrollf32(%n : i32, %da : f32, %dx: memref<?xf32>, %incx : i32,
                                   %dy: memref<?xf32>, %incy : i32)-> ()
func.func private @mlir_linpackcddotunrollf32(%n : i32, %dx: memref<?xf32>, %incx : i32, 
                                   %dy: memref<?xf32>, %incy : i32) -> f32     

func.func @get_val_dgesl_f32(%a: memref<?xf32>, %lda : index, %i : index, %j : index ) -> f32{
  // m[lda*i+j];
  %lda_mi = arith.muli %lda, %i : index
  %lda_mi_aj = arith.addi %lda_mi, %j : index
  %a_val = memref.load %a[%lda_mi_aj] : memref<?xf32>
  return %a_val : f32
}

func.func @set_val_dgesl_f32(%a: memref<?xf32>, %lda : index, %i : index, %j : index, %t: f32){
  // m[lda*i+j];
  %lda_mi = arith.muli %lda, %i : index
  %lda_mi_aj = arith.addi %lda_mi, %j : index
  memref.store %t, %a[%lda_mi_aj] : memref<?xf32>
  return 
}  

func.func @get_offset_dgesl_f32(%lda : index, %i : index, %j : index) -> index {
  // m[lda*i+j];
  %lda_mi = arith.muli %lda, %i : index
  %lda_mi_aj = arith.addi %lda_mi, %j : index
  return %lda_mi_aj : index
}     

func.func @mlir_linpackcdgeslrollf32(%a : memref<?xf32>, %lda : i32, %n: i32, %ipvt : memref<?xi32>, %b : memref<?xf32>, %job: i32) 
{
  // REAL ddot(),t;
  // 	int k,kb,l,nm1;
  // nm1 = n - 1;
  %i0 = arith.constant 0 : i32
  %f0 = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %i1 = arith.constant 1 : i32   
  %nf1 = arith.constant -1.0 : f32  
  %nm1 = arith.subi %n, %i1 : i32
  %n_index = arith.index_cast %n : i32 to index
  %nm1_index = arith.index_cast %nm1 : i32 to index
  %lda_index = arith.index_cast %lda : i32 to index
  %length_a = memref.dim %a, %c0 : memref<?xf32>
  %length_b = memref.dim %b, %c0 : memref<?xf32>
  //   if (job == 0) {
  %cond1 = arith.cmpi "eq", %job, %i0 : i32
  scf.if %cond1{
    // if (nm1 >= 1) {
    %cond2 = arith.cmpi "sge", %nm1, %i1 : i32
     scf.if %cond2{
		// 	for (k = 0; k < nm1; k++) {
		// 		l = ipvt[k];
		// 		t = b[l];
		// 		if (l != k){ 
		// 			b[l] = b[k];
		// 			b[k] = t;
		// 		}	
		// 		daxpy(n-(k+1),t,&a[lda*k+k+1],1,&b[k+1],1);
		// 	}
		// }
        scf.for %k_0_index = %c0 to %nm1_index step %c1{
          %k_0_index_p1 = arith.addi %k_0_index, %c1 : index
          %k_0_p1 = arith.index_cast %k_0_index_p1 : index to i32
          %l_0 = memref.load %ipvt[%k_0_index] : memref<?xi32>
          %l_0_index = arith.index_cast %l_0  : i32 to index
          %t_0 = memref.load %b[%l_0_index] : memref<?xf32>
          %cond3 = arith.cmpi "ne", %l_0_index, %k_0_index : index
          scf.if %cond3{
          //   b[l] = b[k];
					//   b[k] = t;
          %b_val_0 = memref.load %b[%k_0_index] : memref<?xf32>
          memref.store %b_val_0, %b[%l_0_index] : memref<?xf32> 
          memref.store %t_0, %b[%k_0_index] : memref<?xf32>  
          } 
          // daxpy(n-(k+1),t,&a[lda*k+k+1],1,&b[k+1],1);
          %n_sk_p1 = arith.subi %n, %k_0_p1 : i32
          %offset_0 = func.call @get_offset_dgesl_f32(%lda_index, %k_0_index, %k_0_index_p1 ) 
                    : (index, index, index) -> index
          %rest_0 = arith.subi %length_a, %offset_0 :index
          %a_offset_0_tmp = memref.subview %a[%offset_0] [%rest_0] [%c1]: memref<?xf32> to memref<?xf32, strided<[?], offset: ?>>
          %a_offset_0 = memref.cast %a_offset_0_tmp: memref<?xf32, strided<[?], offset: ?>> to memref<?xf32>
          %rest_1 = arith.subi %length_b, %k_0_index_p1 :index
          %b_offset_1_tmp = memref.subview %b[%k_0_index_p1] [%rest_1] [%c1]: memref<?xf32> to memref<?xf32, strided<[?], offset: ?>>
          %b_offset_1 = memref.cast %b_offset_1_tmp: memref<?xf32, strided<[?], offset: ?>> to memref<?xf32>
          func.call @mlir_linpackcdaxpyrollf32(%n_sk_p1, %t_0, %a_offset_0, %i1, %b_offset_1, %i1) 
                    : ( i32, f32,  memref<?xf32>,  i32, memref<?xf32>, i32) -> ()

      }
    }
    // for (kb = 0; kb < n; kb++) {
		//     k = n - (kb + 1);
		//     b[k] = b[k]/a[lda*k+k];
		//     t = -b[k];
		//     daxpy(k,t,&a[lda*k+0],1,&b[0],1);
		// }
    scf.for %kb_0_index = %c0 to %n_index step %c1{
        %kb_0_p1_index = arith.addi %kb_0_index, %c1 : index
        %n_skb_0_p1_index = arith.subi %n_index, %kb_0_p1_index : index
        %n_skb_0_p1 = arith.index_cast %n_skb_0_p1_index : index to i32
        %a_val_1 = func.call @get_val_dgesl_f32(%a, %lda_index, %n_skb_0_p1_index, %n_skb_0_p1_index ) 
                    : (memref<?xf32>, index, index, index) -> f32
        %b_val_2 = memref.load %b[%n_skb_0_p1_index] : memref<?xf32>
        %b_val_3 =  arith.divf %b_val_2, %a_val_1 : f32
        memref.store %b_val_3, %b[%n_skb_0_p1_index] : memref<?xf32>
        //  daxpy(k,t,&a[lda*k+0],1,&b[0],1);
        %t_1 =  arith.negf %b_val_3 : f32
        %offset_2 = func.call @get_offset_dgesl_f32(%lda_index, %n_skb_0_p1_index, %c0 ) 
                    : (index, index, index) -> index
        %rest_2 = arith.subi %length_a, %offset_2 :index
        %a_offset_2_tmp = memref.subview %a[%offset_2] [%rest_2] [%c1]: memref<?xf32> to memref<?xf32, strided<[?], offset: ?>>
        %a_offset_2 = memref.cast %a_offset_2_tmp: memref<?xf32, strided<[?], offset: ?>> to memref<?xf32>
        func.call @mlir_linpackcdaxpyrollf32(%n_skb_0_p1, %t_1, %a_offset_2, %i1, %b, %i1) 
                    : ( i32, f32,  memref<?xf32>,  i32, memref<?xf32>, i32) -> ()
    }
  }else{
    //   for (k = 0; k < n; k++) {
		// 	t = ddot(k,&a[lda*k+0],1,&b[0],1);
		// 	b[k] = (b[k] - t)/a[lda*k+k];
		// }
    scf.for %k_1_index = %c0 to %n_index step %c1{
      %k_1 = arith.index_cast %k_1_index : index to i32
      %offset_3 = func.call @get_offset_dgesl_f32(%lda_index, %k_1_index, %c0 ) 
                    : (index, index, index) -> index
      %rest_3 = arith.subi %length_a, %offset_3 :index
      %a_offset_3_tmp = memref.subview %a[%offset_3] [%rest_3] [%c1]: memref<?xf32> to memref<?xf32, strided<[?], offset: ?>>
      %a_offset_3 = memref.cast %a_offset_3_tmp: memref<?xf32, strided<[?], offset: ?>> to memref<?xf32>
      %t_2 =  func.call @mlir_linpackcddotrollf32(%k_1, %a_offset_3, %i1, %b, %i1)
                                   : (i32, memref<?xf32>, i32, memref<?xf32>, i32) -> f32   
    }

    // if (nm1 >= 1) {
		// 	for (kb = 1; kb < nm1; kb++) {
		// 		k = n - (kb+1);
		// 		b[k] = b[k] + ddot(n-(k+1),&a[lda*k+k+1],1,&b[k+1],1);
		// 		l = ipvt[k];
		// 		if (l != k) {
		// 			t = b[l];
		// 			b[l] = b[k];
		// 			b[k] = t;
		// 		}
		// 	}
		// }
    %cond3 = arith.cmpi "sge", %nm1, %i1 : i32
    scf.if %cond3{
        scf.for %kb_1_index = %c1 to %nm1_index step %c1{
          %kb_1_p1_index = arith.addi %kb_1_index, %c1 : index
          %kb_1_p1 = arith.index_cast %kb_1_p1_index : index to i32
          %k_3_index = arith.subi %n_index, %kb_1_p1_index : index
          %k_3_p1_index = arith.addi %k_3_index, %c1 : index
          %n_sk_3_p1_index = arith.subi %n_index, %k_3_p1_index : index
          %n_sk_3_p1 = arith.index_cast %n_sk_3_p1_index : index to i32
          %K_3 = arith.index_cast %k_3_index : index to i32
          // b[k] = b[k] + ddot(n-(k+1),&a[lda*k+k+1],1,&b[k+1],1);
          %offset_4 = func.call @get_offset_dgesl_f32(%lda_index, %k_3_index, %k_3_p1_index ) 
                    : (index, index, index) -> index
          %rest_4 = arith.subi %length_a, %offset_4 :index
          %a_offset_4_tmp = memref.subview %a[%offset_4] [%rest_4] [%c1]: memref<?xf32> to memref<?xf32, strided<[?], offset: ?>>
          %a_offset_4 = memref.cast %a_offset_4_tmp: memref<?xf32, strided<[?], offset: ?>> to memref<?xf32>
          %rest_5 = arith.subi %length_b, %k_3_p1_index :index
          %b_offset_5_tmp = memref.subview %b[%k_3_p1_index] [%rest_5] [%c1]: memref<?xf32> to memref<?xf32, strided<[?], offset: ?>>
          %b_offset_5 = memref.cast %b_offset_5_tmp: memref<?xf32, strided<[?], offset: ?>> to memref<?xf32>
          %t_2 =  func.call @mlir_linpackcddotrollf32(%n_sk_3_p1, %a_offset_4, %i1, %b_offset_5, %i1)
                                   :(i32, memref<?xf32>, i32, memref<?xf32>, i32) -> f32 
          // l = ipvt[k];
          %l_1 = memref.load %ipvt[%k_3_index] : memref<?xi32>
          %l_1_index = arith.index_cast %l_1  : i32 to index
          %cond4 = arith.cmpi "ne", %l_1, %K_3 : i32
          scf.if %cond4{
          // t = b[l];
					// b[l] = b[k];
					// b[k] = t;
          %t_3 = memref.load %b[%l_1_index] : memref<?xf32>
          %b_val_4 = memref.load %b[%k_3_index] : memref<?xf32>
          memref.store %b_val_4, %b[%l_1_index] : memref<?xf32> 
          memref.store %t_3, %b[%k_3_index] : memref<?xf32>  
          } 
      }
    }
  }
  return 
}

func.func @mlir_linpackcdgeslunrollf32(%a : memref<?xf32>, %lda : i32, %n: i32, %ipvt : memref<?xi32>, %b : memref<?xf32>, %job: i32) 
{
  // REAL ddot(),t;
  // 	int k,kb,l,nm1;
  // nm1 = n - 1;
  %i0 = arith.constant 0 : i32
  %f0 = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %i1 = arith.constant 1 : i32   
  %nf1 = arith.constant -1.0 : f32  
  %nm1 = arith.subi %n, %i1 : i32
  %n_index = arith.index_cast %n : i32 to index
  %nm1_index = arith.index_cast %nm1 : i32 to index
  %lda_index = arith.index_cast %lda : i32 to index
  %length_a = memref.dim %a, %c0 : memref<?xf32>
  %length_b = memref.dim %b, %c0 : memref<?xf32>
  //   if (job == 0) {
  %cond1 = arith.cmpi "eq", %job, %i0 : i32
  scf.if %cond1{
    // if (nm1 >= 1) {
    %cond2 = arith.cmpi "sge", %nm1, %i1 : i32
     scf.if %cond2{
		// 	for (k = 0; k < nm1; k++) {
		// 		l = ipvt[k];
		// 		t = b[l];
		// 		if (l != k){ 
		// 			b[l] = b[k];
		// 			b[k] = t;
		// 		}	
		// 		daxpy(n-(k+1),t,&a[lda*k+k+1],1,&b[k+1],1);
		// 	}
		// }
        scf.for %k_0_index = %c0 to %nm1_index step %c1{
          %k_0_index_p1 = arith.addi %k_0_index, %c1 : index
          %k_0_p1 = arith.index_cast %k_0_index_p1 : index to i32
          %l_0 = memref.load %ipvt[%k_0_index] : memref<?xi32>
          %l_0_index = arith.index_cast %l_0  : i32 to index
          %t_0 = memref.load %b[%l_0_index] : memref<?xf32>
          %cond3 = arith.cmpi "ne", %l_0_index, %k_0_index : index
          scf.if %cond3{
          //   b[l] = b[k];
					//   b[k] = t;
          %b_val_0 = memref.load %b[%k_0_index] : memref<?xf32>
          memref.store %b_val_0, %b[%l_0_index] : memref<?xf32> 
          memref.store %t_0, %b[%k_0_index] : memref<?xf32>  
          } 
          // daxpy(n-(k+1),t,&a[lda*k+k+1],1,&b[k+1],1);
          %n_sk_p1 = arith.subi %n, %k_0_p1 : i32
          %offset_0 = func.call @get_offset_dgesl_f32(%lda_index, %k_0_index, %k_0_index_p1 ) 
                    : (index, index, index) -> index
          %rest_0 = arith.subi %length_a, %offset_0 :index
          %a_offset_0_tmp = memref.subview %a[%offset_0] [%rest_0] [%c1]: memref<?xf32> to memref<?xf32, strided<[?], offset: ?>>
          %a_offset_0 = memref.cast %a_offset_0_tmp: memref<?xf32, strided<[?], offset: ?>> to memref<?xf32>
          %rest_1 = arith.subi %length_b, %k_0_index_p1 :index
          %b_offset_1_tmp = memref.subview %b[%k_0_index_p1] [%rest_1] [%c1]: memref<?xf32> to memref<?xf32, strided<[?], offset: ?>>
          %b_offset_1 = memref.cast %b_offset_1_tmp: memref<?xf32, strided<[?], offset: ?>> to memref<?xf32>
          func.call @mlir_linpackcdaxpyunrollf32(%n_sk_p1, %t_0, %a_offset_0, %i1, %b_offset_1, %i1) 
                    : ( i32, f32,  memref<?xf32>,  i32, memref<?xf32>, i32) -> ()

      }
    }
    // for (kb = 0; kb < n; kb++) {
		//     k = n - (kb + 1);
		//     b[k] = b[k]/a[lda*k+k];
		//     t = -b[k];
		//     daxpy(k,t,&a[lda*k+0],1,&b[0],1);
		// }
    scf.for %kb_0_index = %c0 to %n_index step %c1{
        %kb_0_p1_index = arith.addi %kb_0_index, %c1 : index
        %n_skb_0_p1_index = arith.subi %n_index, %kb_0_p1_index : index
        %n_skb_0_p1 = arith.index_cast %n_skb_0_p1_index : index to i32
        %a_val_1 = func.call @get_val_dgesl_f32(%a, %lda_index, %n_skb_0_p1_index, %n_skb_0_p1_index ) 
                    : (memref<?xf32>, index, index, index) -> f32
        %b_val_2 = memref.load %b[%n_skb_0_p1_index] : memref<?xf32>
        %b_val_3 =  arith.divf %b_val_2, %a_val_1 : f32
        memref.store %b_val_3, %b[%n_skb_0_p1_index] : memref<?xf32>
        //  daxpy(k,t,&a[lda*k+0],1,&b[0],1);
        %t_1 =  arith.negf %b_val_3 : f32
        %offset_2 = func.call @get_offset_dgesl_f32(%lda_index, %n_skb_0_p1_index, %c0 ) 
                    : (index, index, index) -> index
        %rest_2 = arith.subi %length_a, %offset_2 :index
        %a_offset_2_tmp = memref.subview %a[%offset_2] [%rest_2] [%c1]: memref<?xf32> to memref<?xf32, strided<[?], offset: ?>>
        %a_offset_2 = memref.cast %a_offset_2_tmp: memref<?xf32, strided<[?], offset: ?>> to memref<?xf32>
        func.call @mlir_linpackcdaxpyunrollf32(%n_skb_0_p1, %t_1, %a_offset_2, %i1, %b, %i1) 
                    : ( i32, f32,  memref<?xf32>,  i32, memref<?xf32>, i32) -> ()
    }
  }else{
    //   for (k = 0; k < n; k++) {
		// 	t = ddot(k,&a[lda*k+0],1,&b[0],1);
		// 	b[k] = (b[k] - t)/a[lda*k+k];
		// }
    scf.for %k_1_index = %c0 to %n_index step %c1{
      %k_1 = arith.index_cast %k_1_index : index to i32
      %offset_3 = func.call @get_offset_dgesl_f32(%lda_index, %k_1_index, %c0 ) 
                    : (index, index, index) -> index
      %rest_3 = arith.subi %length_a, %offset_3 :index
      %a_offset_3_tmp = memref.subview %a[%offset_3] [%rest_3] [%c1]: memref<?xf32> to memref<?xf32, strided<[?], offset: ?>>
      %a_offset_3 = memref.cast %a_offset_3_tmp: memref<?xf32, strided<[?], offset: ?>> to memref<?xf32>
      %t_2 =  func.call @mlir_linpackcddotunrollf32(%k_1, %a_offset_3, %i1, %b, %i1)
                                   : (i32, memref<?xf32>, i32, memref<?xf32>, i32) -> f32   
    }

    // if (nm1 >= 1) {
		// 	for (kb = 1; kb < nm1; kb++) {
		// 		k = n - (kb+1);
		// 		b[k] = b[k] + ddot(n-(k+1),&a[lda*k+k+1],1,&b[k+1],1);
		// 		l = ipvt[k];
		// 		if (l != k) {
		// 			t = b[l];
		// 			b[l] = b[k];
		// 			b[k] = t;
		// 		}
		// 	}
		// }
    %cond3 = arith.cmpi "sge", %nm1, %i1 : i32
    scf.if %cond3{
        scf.for %kb_1_index = %c1 to %nm1_index step %c1{
          %kb_1_p1_index = arith.addi %kb_1_index, %c1 : index
          %kb_1_p1 = arith.index_cast %kb_1_p1_index : index to i32
          %k_3_index = arith.subi %n_index, %kb_1_p1_index : index
          %k_3_p1_index = arith.addi %k_3_index, %c1 : index
          %n_sk_3_p1_index = arith.subi %n_index, %k_3_p1_index : index
          %n_sk_3_p1 = arith.index_cast %n_sk_3_p1_index : index to i32
          %K_3 = arith.index_cast %k_3_index : index to i32
          // b[k] = b[k] + ddot(n-(k+1),&a[lda*k+k+1],1,&b[k+1],1);
          %offset_4 = func.call @get_offset_dgesl_f32(%lda_index, %k_3_index, %k_3_p1_index ) 
                    : (index, index, index) -> index
          %rest_4 = arith.subi %length_a, %offset_4 :index
          %a_offset_4_tmp = memref.subview %a[%offset_4] [%rest_4] [%c1]: memref<?xf32> to memref<?xf32, strided<[?], offset: ?>>
          %a_offset_4 = memref.cast %a_offset_4_tmp: memref<?xf32, strided<[?], offset: ?>> to memref<?xf32>
          %rest_5 = arith.subi %length_b, %k_3_p1_index :index
          %b_offset_5_tmp = memref.subview %b[%k_3_p1_index] [%rest_5] [%c1]: memref<?xf32> to memref<?xf32, strided<[?], offset: ?>>
          %b_offset_5 = memref.cast %b_offset_5_tmp: memref<?xf32, strided<[?], offset: ?>> to memref<?xf32>
          %t_2 =  func.call @mlir_linpackcddotunrollf32(%n_sk_3_p1, %a_offset_4, %i1, %b_offset_5, %i1)
                                   :(i32, memref<?xf32>, i32, memref<?xf32>, i32) -> f32 
          // l = ipvt[k];
          %l_1 = memref.load %ipvt[%k_3_index] : memref<?xi32>
          %l_1_index = arith.index_cast %l_1  : i32 to index
          %cond4 = arith.cmpi "ne", %l_1, %K_3 : i32
          scf.if %cond4{
          // t = b[l];
					// b[l] = b[k];
					// b[k] = t;
          %t_3 = memref.load %b[%l_1_index] : memref<?xf32>
          %b_val_4 = memref.load %b[%k_3_index] : memref<?xf32>
          memref.store %b_val_4, %b[%l_1_index] : memref<?xf32> 
          memref.store %t_3, %b[%k_3_index] : memref<?xf32>  
          } 
      }
    }
  }
  return 
}


}
