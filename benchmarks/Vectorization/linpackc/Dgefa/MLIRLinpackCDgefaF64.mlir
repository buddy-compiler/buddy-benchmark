//===- MLIRLinpackCDgefaF64.mlir ------------------------------------------===//
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
// This file provides the MLIR linpackc dgefa function.
//
//===----------------------------------------------------------------------===//
module{
func.func private @mlir_linpackcidamaxf64(%n : i32,  %dx: memref<?xf64>, %incx : i32) -> i32
func.func private @mlir_linpackcdscalrollf64(%n : i32, %da : f64, %dx: memref<?xf64>, %incx : i32) -> () 
func.func private @mlir_linpackcdaxpyrollf64(%n : i32, %da : f64, %dx: memref<?xf64>, %incx : i32,
                                   %dy: memref<?xf64>, %incy : i32) -> ()
func.func private @mlir_linpackcdscalunrollf64(%n : i32, %da : f64, %dx: memref<?xf64>, %incx : i32) -> () 
func.func private @mlir_linpackcdaxpyunrollf64(%n : i32, %da : f64, %dx: memref<?xf64>, %incx : i32,
                                   %dy: memref<?xf64>, %incy : i32) -> ()

func.func @get_val_dgefa_f64(%a: memref<?xf64>, %lda : index, %i : index, %j : index ) -> f64{
  // m[lda*i+j];
  %lda_mi = arith.muli %lda, %i : index
  %lda_mi_aj = arith.addi %lda_mi, %j : index
  %a_val = memref.load %a[%lda_mi_aj] : memref<?xf64>
  return %a_val : f64
}

func.func @set_val_dgefa_f64(%a: memref<?xf64>, %lda : index, %i : index, %j : index, %t: f64){
  // m[lda*i+j];
  %lda_mi = arith.muli %lda, %i : index
  %lda_mi_aj = arith.addi %lda_mi, %j : index
  memref.store %t, %a[%lda_mi_aj] : memref<?xf64>
  return 
}  

func.func @get_offset_dgefa_f64(%lda : index, %i : index, %j : index) -> index {
  // m[lda*i+j];
  %lda_mi = arith.muli %lda, %i : index
  %lda_mi_aj = arith.addi %lda_mi, %j : index
  return %lda_mi_aj : index
} 
// dgefa(a,lda,n,ipvt,info)
func.func @mlir_linpackcdgefarollf64(%a : memref<?xf64>, %lda : i32, %n: i32, %ipvt : memref<?xi32>, %info : memref<i32>) 
{
  // REAL t;
  // int idamax(),j,k,kp1,l,nm1;
  // 	*info = 0;
  // 	nm1 = n - 1;
  %i0 = arith.constant 0 : i32
  %f0 = arith.constant 0.0 : f64
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %i1 = arith.constant 1 : i32   
  %nf1 = arith.constant -1.0 : f64  
  memref.store %i0, %info[] : memref<i32>
  %nm1 = arith.subi %n, %i1 : i32
  %n_index = arith.index_cast %n : i32 to index
  %nm1_index = arith.index_cast %nm1 : i32 to index
  %lda_index = arith.index_cast %lda : i32 to index
  %length = memref.dim %a, %c0 : memref<?xf64>
  //   if (nm1 >=  0) {
  %cond1 = arith.cmpi "sge", %nm1, %i0 : i32
  scf.if %cond1{
    scf.for %k_0_index = %c0 to %nm1_index step %c1{
  // 			kp1 = k + 1;
  //           		/* find l = pivot index	*/
  // 			l = idamax(n-k,&a[lda*k+k],1) + k;
  // 			ipvt[k] = l;

      %k_0_index_p1 = arith.addi %k_0_index, %c1 : index
      %k_0 = arith.index_cast %k_0_index : index to i32
      %k_0_p1 = arith.addi %k_0, %i1 : i32
      %n_sk = arith.subi %n, %k_0 :i32
      
      %offset_0 = func.call @get_offset_dgefa_f64(%lda_index, %k_0_index, %k_0_index ) 
                    : (index, index, index) -> index
      
      %rest_0 = arith.subi %length, %offset_0 :index
      %a_offset_0_tmp = memref.subview %a[%offset_0] [%rest_0] [%c1]: memref<?xf64> to memref<?xf64, strided<[?], offset: ?>>
      %a_offset_0 = memref.cast %a_offset_0_tmp: memref<?xf64, strided<[?], offset: ?>> to memref<?xf64>
      %l_temp = func.call @mlir_linpackcidamaxf64(%n_sk, %a_offset_0, %i1)
        : (i32, memref<?xf64>, i32) -> i32
      %l = arith.addi %l_temp, %k_0 : i32
      memref.store %l, %ipvt[%k_0_index] : memref<?xi32>

      // if (a[lda*k+l] != ZERO) {
      %l_index = arith.index_cast %l : i32 to index
      %a_val_0 = func.call @get_val_dgefa_f64(%a, %lda_index, %k_0_index, %k_0_index ) 
        : (memref<?xf64>, index, index, index) -> f64
      %a_val_1 = func.call @get_val_dgefa_f64(%a, %lda_index, %k_0_index, %l_index ) 
        : (memref<?xf64>, index, index, index) -> f64
      %cond2 = arith.cmpf "une", %a_val_1, %f0 : f64
      scf.if %cond2 {
          %cond3 = arith.cmpi "ne", %l, %k_0 : i32
          scf.if %cond3{
            //     if (l != k) {
            // 	t = a[lda*k+l];
            // 	a[lda*k+l] = a[lda*k+k];
            // 	a[lda*k+k] = t; 
            // }
            func.call @set_val_dgefa_f64(%a, %lda_index, %k_0_index, %l_index, %a_val_0) 
                : (memref<?xf64>, index, index, index, f64) -> ()
            func.call @set_val_dgefa_f64(%a, %lda_index, %k_0_index, %k_0_index, %a_val_1 ) 
                : (memref<?xf64>, index, index, index, f64) -> ()
          }
          //   t = -ONE/a[lda*k+k];
          %a_val_2 = func.call @get_val_dgefa_f64(%a, %lda_index, %k_0_index, %k_0_index ) 
                    : (memref<?xf64>, index, index, index) -> f64
          %t_1 = arith.divf %nf1, %a_val_2 :f64
          %n_sk_p1 = arith.subi %n, %k_0_p1 : i32
          %a_val_3 = func.call @get_val_dgefa_f64(%a, %lda_index, %k_0_index, %k_0_index_p1 ) 
                    : (memref<?xf64>, index, index, index) -> f64
          %offset_3 = func.call @get_offset_dgefa_f64(%lda_index, %k_0_index, %k_0_index_p1 ) 
                    : (index, index, index) -> index
          %rest_3 = arith.subi %length, %offset_3 :index
          %a_offset_3_tmp = memref.subview %a[%offset_3] [%rest_3] [%c1]: memref<?xf64> to memref<?xf64, strided<[?], offset: ?>>
          %a_offset_3 = memref.cast %a_offset_3_tmp: memref<?xf64, strided<[?], offset: ?>> to memref<?xf64>
          // dscal(n-(k+1),t,&a[lda*k+k+1],1);          
          func.call @mlir_linpackcdscalrollf64(%n_sk_p1 , %t_1, %a_offset_3, %i1)
                    : (i32, f64, memref<?xf64>, i32) -> ()

          scf.for %j_0_index = %k_0_index_p1 to %n_index step %c1{
            // t = a[lda*j+l]; 
            %t_2 = func.call @get_val_dgefa_f64(%a, %lda_index, %j_0_index, %l_index ) 
                    : (memref<?xf64>, index, index, index) -> f64
            //   if (l != k) {
            // 	a[lda*j+l] = a[lda*j+k];
            // 	a[lda*j+k] = t;
            // }
            scf.if %cond3{
                %a_val_4 = func.call @get_val_dgefa_f64(%a, %lda_index, %j_0_index, %k_0_index ) 
                    : (memref<?xf64>, index, index, index) -> f64
                func.call @set_val_dgefa_f64(%a, %lda_index, %j_0_index, %l_index, %a_val_4) 
                    : (memref<?xf64>, index, index, index, f64) -> ()
                func.call @set_val_dgefa_f64(%a, %lda_index, %j_0_index, %k_0_index, %t_2) 
                    : (memref<?xf64>, index, index, index, f64) -> ()
            }   
            // daxpy(n-(k+1),t,&a[lda*k+k+1],1,
					  //     &a[lda*j+k+1],1);     
            %offset_4 = func.call @get_offset_dgefa_f64(%lda_index, %k_0_index, %k_0_index_p1 ) 
                    : (index, index, index) -> index
            %rest_4 = arith.subi %length, %offset_4 :index
            %a_offset_4_tmp = memref.subview %a[%offset_4] [%rest_4] [%c1]: memref<?xf64> to memref<?xf64, strided<[?], offset: ?>>
            %a_offset_4 = memref.cast %a_offset_4_tmp: memref<?xf64, strided<[?], offset: ?>> to memref<?xf64>
            
            %offset_5 = func.call @get_offset_dgefa_f64(%lda_index, %j_0_index, %k_0_index_p1 ) 
                    : (index, index, index) -> index
            %rest_5 = arith.subi %length, %offset_5 :index
            %a_offset_5_tmp = memref.subview %a[%offset_5] [%rest_5] [%c1]: memref<?xf64> to memref<?xf64, strided<[?], offset: ?>>
            %a_offset_5 = memref.cast %a_offset_5_tmp: memref<?xf64, strided<[?], offset: ?>> to memref<?xf64>
            func.call @mlir_linpackcdaxpyrollf64(%n_sk_p1, %t_2, %a_offset_4, %i1, %a_offset_5, %i1) 
                    : ( i32, f64,  memref<?xf64>,  i32, memref<?xf64>, i32) -> ()
          }
      }else{
           //*info = k;
            memref.store %k_0, %info[] : memref<i32>
      }
    }
  }
  // ipvt[n-1] = n-1;
	// if (a[lda*(n-1)+(n-1)] == ZERO) *info = n-1;
    %n_s1_index = arith.subi %n_index, %c1 :index
    %n_s1 = arith.subi %n, %i1 :i32
    memref.store %n_s1, %ipvt[%n_s1_index] : memref<?xi32>
    %a_val_6 = func.call @get_val_dgefa_f64(%a, %lda_index, %n_s1_index, %n_s1_index ) 
                    : (memref<?xf64>, index, index, index) -> f64
    %cond4 = arith.cmpf "ueq", %a_val_6, %f0 : f64
    scf.if %cond4 {
       memref.store %n_s1, %info[] : memref<i32>
    }
  
  return 
}

func.func @mlir_linpackcdgefaunrollf64(%a : memref<?xf64>, %lda : i32, %n: i32, %ipvt : memref<?xi32>, %info : memref<i32>) 
{
  // REAL t;
  // int idamax(),j,k,kp1,l,nm1;
  // 	*info = 0;
  // 	nm1 = n - 1;
  %i0 = arith.constant 0 : i32
  %f0 = arith.constant 0.0 : f64
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %i1 = arith.constant 1 : i32   
  %nf1 = arith.constant -1.0 : f64  
  memref.store %i0, %info[] : memref<i32>
  %nm1 = arith.subi %n, %i1 : i32
  %n_index = arith.index_cast %n : i32 to index
  %nm1_index = arith.index_cast %nm1 : i32 to index
  %lda_index = arith.index_cast %lda : i32 to index
  %length = memref.dim %a, %c0 : memref<?xf64>
  //   if (nm1 >=  0) {
  %cond1 = arith.cmpi "sge", %nm1, %i0 : i32
  scf.if %cond1{
    scf.for %k_0_index = %c0 to %nm1_index step %c1{
  // 			kp1 = k + 1;
  //           		/* find l = pivot index	*/
  // 			l = idamax(n-k,&a[lda*k+k],1) + k;
  // 			ipvt[k] = l;

      %k_0_index_p1 = arith.addi %k_0_index, %c1 : index
      %k_0 = arith.index_cast %k_0_index : index to i32
      %k_0_p1 = arith.addi %k_0, %i1 : i32
      %n_sk = arith.subi %n, %k_0 :i32
      
      %offset_0 = func.call @get_offset_dgefa_f64(%lda_index, %k_0_index, %k_0_index ) 
                    : (index, index, index) -> index
      
      %rest_0 = arith.subi %length, %offset_0 :index
      %a_offset_0_tmp = memref.subview %a[%offset_0] [%rest_0] [%c1]: memref<?xf64> to memref<?xf64, strided<[?], offset: ?>>
      %a_offset_0 = memref.cast %a_offset_0_tmp: memref<?xf64, strided<[?], offset: ?>> to memref<?xf64>
      %l_temp = func.call @mlir_linpackcidamaxf64(%n_sk, %a_offset_0, %i1)
        : (i32, memref<?xf64>, i32) -> i32
      %l = arith.addi %l_temp, %k_0 : i32
      memref.store %l, %ipvt[%k_0_index] : memref<?xi32>

      // if (a[lda*k+l] != ZERO) {
      %l_index = arith.index_cast %l : i32 to index
      %a_val_0 = func.call @get_val_dgefa_f64(%a, %lda_index, %k_0_index, %k_0_index ) 
        : (memref<?xf64>, index, index, index) -> f64
      %a_val_1 = func.call @get_val_dgefa_f64(%a, %lda_index, %k_0_index, %l_index ) 
        : (memref<?xf64>, index, index, index) -> f64
      %cond2 = arith.cmpf "une", %a_val_1, %f0 : f64
      scf.if %cond2 {
          %cond3 = arith.cmpi "ne", %l, %k_0 : i32
          scf.if %cond3{
            //     if (l != k) {
            // 	t = a[lda*k+l];
            // 	a[lda*k+l] = a[lda*k+k];
            // 	a[lda*k+k] = t; 
            // }
            func.call @set_val_dgefa_f64(%a, %lda_index, %k_0_index, %l_index, %a_val_0) 
                : (memref<?xf64>, index, index, index, f64) -> ()
            func.call @set_val_dgefa_f64(%a, %lda_index, %k_0_index, %k_0_index, %a_val_1 ) 
                : (memref<?xf64>, index, index, index, f64) -> ()
          }
          //   t = -ONE/a[lda*k+k];
          %a_val_2 = func.call @get_val_dgefa_f64(%a, %lda_index, %k_0_index, %k_0_index ) 
                    : (memref<?xf64>, index, index, index) -> f64
          %t_1 = arith.divf %nf1, %a_val_2 :f64
          %n_sk_p1 = arith.subi %n, %k_0_p1 : i32
          %a_val_3 = func.call @get_val_dgefa_f64(%a, %lda_index, %k_0_index, %k_0_index_p1 ) 
                    : (memref<?xf64>, index, index, index) -> f64
          %offset_3 = func.call @get_offset_dgefa_f64(%lda_index, %k_0_index, %k_0_index_p1 ) 
                    : (index, index, index) -> index
          %rest_3 = arith.subi %length, %offset_3 :index
          %a_offset_3_tmp = memref.subview %a[%offset_3] [%rest_3] [%c1]: memref<?xf64> to memref<?xf64, strided<[?], offset: ?>>
          %a_offset_3 = memref.cast %a_offset_3_tmp: memref<?xf64, strided<[?], offset: ?>> to memref<?xf64>
          // dscal(n-(k+1),t,&a[lda*k+k+1],1);          
          func.call @mlir_linpackcdscalunrollf64(%n_sk_p1 , %t_1, %a_offset_3, %i1)
                    : (i32, f64, memref<?xf64>, i32) -> ()

          scf.for %j_0_index = %k_0_index_p1 to %n_index step %c1{
            // t = a[lda*j+l]; 
            %t_2 = func.call @get_val_dgefa_f64(%a, %lda_index, %j_0_index, %l_index ) 
                    : (memref<?xf64>, index, index, index) -> f64
            //   if (l != k) {
            // 	a[lda*j+l] = a[lda*j+k];
            // 	a[lda*j+k] = t;
            // }
            scf.if %cond3{
                %a_val_4 = func.call @get_val_dgefa_f64(%a, %lda_index, %j_0_index, %k_0_index ) 
                    : (memref<?xf64>, index, index, index) -> f64
                func.call @set_val_dgefa_f64(%a, %lda_index, %j_0_index, %l_index, %a_val_4) 
                    : (memref<?xf64>, index, index, index, f64) -> ()
                func.call @set_val_dgefa_f64(%a, %lda_index, %j_0_index, %k_0_index, %t_2) 
                    : (memref<?xf64>, index, index, index, f64) -> ()
            }   
            // daxpy(n-(k+1),t,&a[lda*k+k+1],1,
					  //     &a[lda*j+k+1],1);     
            %offset_4 = func.call @get_offset_dgefa_f64(%lda_index, %k_0_index, %k_0_index_p1 ) 
                    : (index, index, index) -> index
            %rest_4 = arith.subi %length, %offset_4 :index
            %a_offset_4_tmp = memref.subview %a[%offset_4] [%rest_4] [%c1]: memref<?xf64> to memref<?xf64, strided<[?], offset: ?>>
            %a_offset_4 = memref.cast %a_offset_4_tmp: memref<?xf64, strided<[?], offset: ?>> to memref<?xf64>
            
            %offset_5 = func.call @get_offset_dgefa_f64(%lda_index, %j_0_index, %k_0_index_p1 ) 
                    : (index, index, index) -> index
            %rest_5 = arith.subi %length, %offset_5 :index
            %a_offset_5_tmp = memref.subview %a[%offset_5] [%rest_5] [%c1]: memref<?xf64> to memref<?xf64, strided<[?], offset: ?>>
            %a_offset_5 = memref.cast %a_offset_5_tmp: memref<?xf64, strided<[?], offset: ?>> to memref<?xf64>
            func.call @mlir_linpackcdaxpyunrollf64(%n_sk_p1, %t_2, %a_offset_4, %i1, %a_offset_5, %i1) 
                    : ( i32, f64,  memref<?xf64>,  i32, memref<?xf64>, i32) -> ()
          }
      }else{
           //*info = k;
            memref.store %k_0, %info[] : memref<i32>
      }
    }
  }
  // ipvt[n-1] = n-1;
	// if (a[lda*(n-1)+(n-1)] == ZERO) *info = n-1;
    %n_s1_index = arith.subi %n_index, %c1 :index
    %n_s1 = arith.subi %n, %i1 :i32
    memref.store %n_s1, %ipvt[%n_s1_index] : memref<?xi32>
    %a_val_6 = func.call @get_val_dgefa_f64(%a, %lda_index, %n_s1_index, %n_s1_index ) 
                    : (memref<?xf64>, index, index, index) -> f64
    %cond4 = arith.cmpf "ueq", %a_val_6, %f0 : f64
    scf.if %cond4 {
       memref.store %n_s1, %info[] : memref<i32>
    }
  
  return 
}

}