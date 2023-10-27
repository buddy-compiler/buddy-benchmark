//===- MLIRLinpackCDmxpy.mlir --------------------------------------===//
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
// This file provides the MLIR linpackc dmxpy function.
//
//===----------------------------------------------------------------------===//

func.func @my_funcTYPE_PLACEHOLDER(%ldm : index, %x: memref<?xTYPE_PLACEHOLDER>, %m : memref<?xTYPE_PLACEHOLDER>, %i : index, %j : index ) -> TYPE_PLACEHOLDER{
  // x[j]*m[ldm*j+i];
  %x_val = memref.load %x[%j] : memref<?xTYPE_PLACEHOLDER>
  %ldm_mj = arith.muli %ldm, %j : index
  %ldm_mj_ai = arith.addi %ldm_mj, %i : index
  %m_val = memref.load %m[%ldm_mj_ai] : memref<?xTYPE_PLACEHOLDER>
  %x_val_mm_val = arith.mulf %x_val, %m_val : TYPE_PLACEHOLDER
  return %x_val_mm_val : TYPE_PLACEHOLDER
}

func.func @mlir_linpackcdmxpyTYPE_PLACEHOLDER(%n1 : i32, %y: memref<?xTYPE_PLACEHOLDER>, %n2 : i32,
                                 %ldm : i32, %x: memref<?xTYPE_PLACEHOLDER>, %m : memref<?xTYPE_PLACEHOLDER>) {
  %c0 = arith.constant 0 : index
  %i0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : index
  %i1 = arith.constant 1 : i32
  %i2 = arith.constant 2 : i32
  %c4 = arith.constant 4 : index
  %i4 = arith.constant 4 : i32
  %i8 = arith.constant 8 : i32
  %i16 = arith.constant 16 : i32
  %c16 = arith.index_cast %i16 : i32 to index
  %n1_index = arith.index_cast %n1 : i32 to index
  %n2_index = arith.index_cast %n2 : i32 to index
  %ldm_index = arith.index_cast %ldm : i32 to index

  %j_1 = arith.remsi %n2, %i2: i32
  %j_2 = arith.remsi %n2, %i4 : i32
  %j_3 = arith.remsi %n2, %i8 : i32
  %j_4 = arith.remsi %n2, %i16 : i32
  %j_min = arith.addi %j_4, %i16 : i32

  %cond1 = arith.cmpi "sge", %j_1, %i1 : i32 
  scf.if %cond1 { 
      %j_1_s1 = arith.subi %j_1, %i1 : i32
      %j_1_s1_index = arith.index_cast %j_1_s1 : i32 to index
      scf.for %i_0 = %c0 to %n1_index step %c1 {
        %y_val_0 = memref.load %y[%i_0] : memref<?xTYPE_PLACEHOLDER>
        %x_val_0_mm_val_0 = func.call @my_funcTYPE_PLACEHOLDER(%ldm_index, %x, %m, %i_0, %j_1_s1_index) 
        : (index, memref<?xTYPE_PLACEHOLDER>, memref<?xTYPE_PLACEHOLDER>, index, index) -> TYPE_PLACEHOLDER
        %x_val_0_mm_val_0_ay_val_0 = arith.addf %x_val_0_mm_val_0, %y_val_0 : TYPE_PLACEHOLDER
        memref.store %x_val_0_mm_val_0_ay_val_0, %y[%i_0] : memref<?xTYPE_PLACEHOLDER>
      }
    } 

  %cond2 = arith.cmpi "sge", %j_2, %i2 : i32 
  scf.if %cond2 { 
      %j_2_s1 = arith.subi %j_2, %i1 : i32
      %j_2_s1_index = arith.index_cast %j_2_s1 : i32 to index
      %j_2_s1_s1_index = arith.subi %j_2_s1_index, %c1 : index
      scf.for %i_1 = %c0 to %n1_index step %c1 {
        %y_val_1 = memref.load %y[%i_1] : memref<?xTYPE_PLACEHOLDER>
        %x_val_1_mm_val_1 = func.call @my_funcTYPE_PLACEHOLDER(%ldm_index, %x, %m, %i_1, %j_2_s1_index) 
        : (index, memref<?xTYPE_PLACEHOLDER>, memref<?xTYPE_PLACEHOLDER>, index, index) -> TYPE_PLACEHOLDER
        %x_val_1_mm_val_2 = func.call @my_funcTYPE_PLACEHOLDER(%ldm_index, %x, %m, %i_1, %j_2_s1_s1_index) 
        : (index, memref<?xTYPE_PLACEHOLDER>, memref<?xTYPE_PLACEHOLDER>, index, index) -> TYPE_PLACEHOLDER
        %val_1_tmp = arith.addf %x_val_1_mm_val_1, %x_val_1_mm_val_2 : TYPE_PLACEHOLDER
        %val_1 = arith.addf %y_val_1, %val_1_tmp : TYPE_PLACEHOLDER
        memref.store %val_1, %y[%i_1] : memref<?xTYPE_PLACEHOLDER>
      }
    } 

  %cond3 = arith.cmpi "sge", %j_3, %i4 : i32 
  scf.if %cond3 { 
      %j_3_s1 = arith.subi %j_3, %i1 : i32
      %j_3_s1_index = arith.index_cast %j_3_s1 : i32 to index
      %j_3_s1_s1_index = arith.subi %j_3_s1_index, %c1 : index
      %j_3_s1_s2_index = arith.subi %j_3_s1_s1_index, %c1 : index
      %j_3_s1_s3_index = arith.subi %j_3_s1_s2_index, %c1 : index
      scf.for %i_2 = %c0 to %n1_index step %c1 {
        %y_val_2 = memref.load %y[%i_2] : memref<?xTYPE_PLACEHOLDER>
        %x_val_2_mm_val_1 = func.call @my_funcTYPE_PLACEHOLDER(%ldm_index, %x, %m, %i_2, %j_3_s1_index) 
        : (index, memref<?xTYPE_PLACEHOLDER>, memref<?xTYPE_PLACEHOLDER>, index, index) -> TYPE_PLACEHOLDER
        %x_val_2_mm_val_2 = func.call @my_funcTYPE_PLACEHOLDER(%ldm_index, %x, %m, %i_2, %j_3_s1_s1_index) 
        : (index, memref<?xTYPE_PLACEHOLDER>, memref<?xTYPE_PLACEHOLDER>, index, index) -> TYPE_PLACEHOLDER
        %x_val_2_mm_val_3 = func.call @my_funcTYPE_PLACEHOLDER(%ldm_index, %x, %m, %i_2, %j_3_s1_s2_index) 
        : (index, memref<?xTYPE_PLACEHOLDER>, memref<?xTYPE_PLACEHOLDER>, index, index) -> TYPE_PLACEHOLDER
        %x_val_2_mm_val_4 = func.call @my_funcTYPE_PLACEHOLDER(%ldm_index, %x, %m, %i_2, %j_3_s1_s3_index) 
        : (index, memref<?xTYPE_PLACEHOLDER>, memref<?xTYPE_PLACEHOLDER>, index, index) -> TYPE_PLACEHOLDER
        %val_2_tmp1 = arith.addf %x_val_2_mm_val_1, %x_val_2_mm_val_2 : TYPE_PLACEHOLDER
        %val_2_tmp2 = arith.addf %val_2_tmp1, %x_val_2_mm_val_3 : TYPE_PLACEHOLDER
        %val_2_tmp3 = arith.addf %val_2_tmp2, %x_val_2_mm_val_4 : TYPE_PLACEHOLDER
        %val_2 = arith.addf %y_val_2, %val_2_tmp3 : TYPE_PLACEHOLDER
        memref.store %val_2, %y[%i_2] : memref<?xTYPE_PLACEHOLDER>
      }
    }

  %cond4 = arith.cmpi "sge", %j_4, %i8 : i32 
  scf.if %cond4 { 
      %j_4_s1 = arith.subi %j_4, %i1 : i32
      %j_4_s1_index = arith.index_cast %j_4_s1 : i32 to index
      %j_4_s1_s1_index = arith.subi %j_4_s1_index, %c1 : index
      %j_4_s1_s2_index = arith.subi %j_4_s1_s1_index, %c1 : index
      %j_4_s1_s3_index = arith.subi %j_4_s1_s2_index, %c1 : index
      %j_4_s1_s4_index = arith.subi %j_4_s1_s3_index, %c1 : index
      %j_4_s1_s5_index = arith.subi %j_4_s1_s4_index, %c1 : index
      %j_4_s1_s6_index = arith.subi %j_4_s1_s5_index, %c1 : index
      %j_4_s1_s7_index = arith.subi %j_4_s1_s6_index, %c1 : index
      scf.for %i_3 = %c0 to %n1_index step %c1 {
        %y_val_3 = memref.load %y[%i_3] : memref<?xTYPE_PLACEHOLDER>
        %x_val_3_mm_val_1 = func.call @my_funcTYPE_PLACEHOLDER(%ldm_index, %x, %m, %i_3, %j_4_s1_index) 
        : (index, memref<?xTYPE_PLACEHOLDER>, memref<?xTYPE_PLACEHOLDER>, index, index) -> TYPE_PLACEHOLDER
        %x_val_3_mm_val_2 = func.call @my_funcTYPE_PLACEHOLDER(%ldm_index, %x, %m, %i_3, %j_4_s1_s1_index) 
        : (index, memref<?xTYPE_PLACEHOLDER>, memref<?xTYPE_PLACEHOLDER>, index, index) -> TYPE_PLACEHOLDER
        %x_val_3_mm_val_3 = func.call @my_funcTYPE_PLACEHOLDER(%ldm_index, %x, %m, %i_3, %j_4_s1_s2_index) 
        : (index, memref<?xTYPE_PLACEHOLDER>, memref<?xTYPE_PLACEHOLDER>, index, index) -> TYPE_PLACEHOLDER
        %x_val_3_mm_val_4 = func.call @my_funcTYPE_PLACEHOLDER(%ldm_index, %x, %m, %i_3, %j_4_s1_s3_index) 
        : (index, memref<?xTYPE_PLACEHOLDER>, memref<?xTYPE_PLACEHOLDER>, index, index) -> TYPE_PLACEHOLDER
        %x_val_3_mm_val_5 = func.call @my_funcTYPE_PLACEHOLDER(%ldm_index, %x, %m, %i_3, %j_4_s1_s4_index) 
        : (index, memref<?xTYPE_PLACEHOLDER>, memref<?xTYPE_PLACEHOLDER>, index, index) -> TYPE_PLACEHOLDER
        %x_val_3_mm_val_6 = func.call @my_funcTYPE_PLACEHOLDER(%ldm_index, %x, %m, %i_3, %j_4_s1_s5_index) 
        : (index, memref<?xTYPE_PLACEHOLDER>, memref<?xTYPE_PLACEHOLDER>, index, index) -> TYPE_PLACEHOLDER
        %x_val_3_mm_val_7 = func.call @my_funcTYPE_PLACEHOLDER(%ldm_index, %x, %m, %i_3, %j_4_s1_s6_index) 
        : (index, memref<?xTYPE_PLACEHOLDER>, memref<?xTYPE_PLACEHOLDER>, index, index) -> TYPE_PLACEHOLDER
        %x_val_3_mm_val_8 = func.call @my_funcTYPE_PLACEHOLDER(%ldm_index, %x, %m, %i_3, %j_4_s1_s7_index) 
        : (index, memref<?xTYPE_PLACEHOLDER>, memref<?xTYPE_PLACEHOLDER>, index, index) -> TYPE_PLACEHOLDER
        %val_3_tmp1 = arith.addf %x_val_3_mm_val_1, %x_val_3_mm_val_2 : TYPE_PLACEHOLDER
        %val_3_tmp2 = arith.addf %val_3_tmp1, %x_val_3_mm_val_3 : TYPE_PLACEHOLDER
        %val_3_tmp3 = arith.addf %val_3_tmp2, %x_val_3_mm_val_4 : TYPE_PLACEHOLDER
        %val_3_tmp4 = arith.addf %val_3_tmp3, %x_val_3_mm_val_5 : TYPE_PLACEHOLDER
        %val_3_tmp5 = arith.addf %val_3_tmp4, %x_val_3_mm_val_6 : TYPE_PLACEHOLDER
        %val_3_tmp6 = arith.addf %val_3_tmp5, %x_val_3_mm_val_7 : TYPE_PLACEHOLDER
        %val_3_tmp7 = arith.addf %val_3_tmp6, %x_val_3_mm_val_8 : TYPE_PLACEHOLDER
        %val_3 = arith.addf %y_val_3, %val_3_tmp7 : TYPE_PLACEHOLDER
        memref.store %val_3, %y[%i_3] : memref<?xTYPE_PLACEHOLDER>
      }
    }

%cond5 = arith.cmpi "sge", %j_min, %i8 : i32 
  scf.if %cond5 { 
      %j_min_s1 = arith.subi %j_min, %i1 : i32
      %j_min_s1_index = arith.index_cast %j_min_s1 : i32 to index
      scf.for %j_5 = %j_min_s1_index to %n2_index step %c16{
        scf.for %i_4 = %c0 to %n1_index step %c1 {
          %j_5_s1_index = arith.subi %j_5, %c1 : index
          %j_5_s2_index = arith.subi %j_5_s1_index, %c1 : index
          %j_5_s3_index = arith.subi %j_5_s2_index, %c1 : index
          %j_5_s4_index = arith.subi %j_5_s3_index, %c1 : index
          %j_5_s5_index = arith.subi %j_5_s4_index, %c1 : index
          %j_5_s6_index = arith.subi %j_5_s5_index, %c1 : index
          %j_5_s7_index = arith.subi %j_5_s6_index, %c1 : index
          %j_5_s8_index = arith.subi %j_5_s7_index, %c1 : index
          %j_5_s9_index = arith.subi %j_5_s8_index, %c1 : index
          %j_5_s10_index = arith.subi %j_5_s9_index, %c1 : index
          %j_5_s11_index = arith.subi %j_5_s10_index, %c1 : index
          %j_5_s12_index = arith.subi %j_5_s11_index, %c1 : index
          %j_5_s13_index = arith.subi %j_5_s12_index, %c1 : index
          %j_5_s14_index = arith.subi %j_5_s13_index, %c1 : index
          %j_5_s15_index = arith.subi %j_5_s14_index, %c1 : index

          %y_val_4 = memref.load %y[%i_4] : memref<?xTYPE_PLACEHOLDER>
          %x_val_4_mm_val_1 = func.call @my_funcTYPE_PLACEHOLDER(%ldm_index, %x, %m, %i_4, %j_5) 
          : (index, memref<?xTYPE_PLACEHOLDER>, memref<?xTYPE_PLACEHOLDER>, index, index) -> TYPE_PLACEHOLDER
          %x_val_4_mm_val_2 = func.call @my_funcTYPE_PLACEHOLDER(%ldm_index, %x, %m, %i_4, %j_5_s1_index) 
          : (index, memref<?xTYPE_PLACEHOLDER>, memref<?xTYPE_PLACEHOLDER>, index, index) -> TYPE_PLACEHOLDER
          %x_val_4_mm_val_3 = func.call @my_funcTYPE_PLACEHOLDER(%ldm_index, %x, %m, %i_4, %j_5_s2_index) 
          : (index, memref<?xTYPE_PLACEHOLDER>, memref<?xTYPE_PLACEHOLDER>, index, index) -> TYPE_PLACEHOLDER
          %x_val_4_mm_val_4 = func.call @my_funcTYPE_PLACEHOLDER(%ldm_index, %x, %m, %i_4, %j_5_s3_index) 
          : (index, memref<?xTYPE_PLACEHOLDER>, memref<?xTYPE_PLACEHOLDER>, index, index) -> TYPE_PLACEHOLDER
          %x_val_4_mm_val_5 = func.call @my_funcTYPE_PLACEHOLDER(%ldm_index, %x, %m, %i_4, %j_5_s4_index) 
          : (index, memref<?xTYPE_PLACEHOLDER>, memref<?xTYPE_PLACEHOLDER>, index, index) -> TYPE_PLACEHOLDER
          %x_val_4_mm_val_6 = func.call @my_funcTYPE_PLACEHOLDER(%ldm_index, %x, %m, %i_4, %j_5_s5_index) 
          : (index, memref<?xTYPE_PLACEHOLDER>, memref<?xTYPE_PLACEHOLDER>, index, index) -> TYPE_PLACEHOLDER
          %x_val_4_mm_val_7 = func.call @my_funcTYPE_PLACEHOLDER(%ldm_index, %x, %m, %i_4, %j_5_s6_index) 
          : (index, memref<?xTYPE_PLACEHOLDER>, memref<?xTYPE_PLACEHOLDER>, index, index) -> TYPE_PLACEHOLDER
          %x_val_4_mm_val_8 = func.call @my_funcTYPE_PLACEHOLDER(%ldm_index, %x, %m, %i_4, %j_5_s7_index) 
          : (index, memref<?xTYPE_PLACEHOLDER>, memref<?xTYPE_PLACEHOLDER>, index, index) -> TYPE_PLACEHOLDER
          %x_val_4_mm_val_9 = func.call @my_funcTYPE_PLACEHOLDER(%ldm_index, %x, %m, %i_4, %j_5_s8_index) 
          : (index, memref<?xTYPE_PLACEHOLDER>, memref<?xTYPE_PLACEHOLDER>, index, index) -> TYPE_PLACEHOLDER
          %x_val_4_mm_val_10 = func.call @my_funcTYPE_PLACEHOLDER(%ldm_index, %x, %m, %i_4, %j_5_s9_index) 
          : (index, memref<?xTYPE_PLACEHOLDER>, memref<?xTYPE_PLACEHOLDER>, index, index) -> TYPE_PLACEHOLDER
          %x_val_4_mm_val_11 = func.call @my_funcTYPE_PLACEHOLDER(%ldm_index, %x, %m, %i_4, %j_5_s10_index) 
          : (index, memref<?xTYPE_PLACEHOLDER>, memref<?xTYPE_PLACEHOLDER>, index, index) -> TYPE_PLACEHOLDER
          %x_val_4_mm_val_12 = func.call @my_funcTYPE_PLACEHOLDER(%ldm_index, %x, %m, %i_4, %j_5_s11_index) 
          : (index, memref<?xTYPE_PLACEHOLDER>, memref<?xTYPE_PLACEHOLDER>, index, index) -> TYPE_PLACEHOLDER
          %x_val_4_mm_val_13 = func.call @my_funcTYPE_PLACEHOLDER(%ldm_index, %x, %m, %i_4, %j_5_s12_index) 
          : (index, memref<?xTYPE_PLACEHOLDER>, memref<?xTYPE_PLACEHOLDER>, index, index) -> TYPE_PLACEHOLDER
          %x_val_4_mm_val_14 = func.call @my_funcTYPE_PLACEHOLDER(%ldm_index, %x, %m, %i_4, %j_5_s13_index) 
          : (index, memref<?xTYPE_PLACEHOLDER>, memref<?xTYPE_PLACEHOLDER>, index, index) -> TYPE_PLACEHOLDER
          %x_val_4_mm_val_15 = func.call @my_funcTYPE_PLACEHOLDER(%ldm_index, %x, %m, %i_4, %j_5_s14_index) 
          : (index, memref<?xTYPE_PLACEHOLDER>, memref<?xTYPE_PLACEHOLDER>, index, index) -> TYPE_PLACEHOLDER
          %x_val_4_mm_val_16 = func.call @my_funcTYPE_PLACEHOLDER(%ldm_index, %x, %m, %i_4, %j_5_s15_index) 
          : (index, memref<?xTYPE_PLACEHOLDER>, memref<?xTYPE_PLACEHOLDER>, index, index) -> TYPE_PLACEHOLDER

          %val_4_tmp1 = arith.addf %x_val_4_mm_val_1, %x_val_4_mm_val_2 : TYPE_PLACEHOLDER
          %val_4_tmp2 = arith.addf %val_4_tmp1, %x_val_4_mm_val_3 : TYPE_PLACEHOLDER
          %val_4_tmp3 = arith.addf %val_4_tmp2, %x_val_4_mm_val_4 : TYPE_PLACEHOLDER
          %val_4_tmp4 = arith.addf %val_4_tmp3, %x_val_4_mm_val_5 : TYPE_PLACEHOLDER
          %val_4_tmp5 = arith.addf %val_4_tmp4, %x_val_4_mm_val_6 : TYPE_PLACEHOLDER
          %val_4_tmp6 = arith.addf %val_4_tmp5, %x_val_4_mm_val_7 : TYPE_PLACEHOLDER
          %val_4_tmp7 = arith.addf %val_4_tmp6, %x_val_4_mm_val_8 : TYPE_PLACEHOLDER
          %val_4_tmp8 = arith.addf %val_4_tmp7, %x_val_4_mm_val_9 : TYPE_PLACEHOLDER
          %val_4_tmp9 = arith.addf %val_4_tmp8, %x_val_4_mm_val_10 : TYPE_PLACEHOLDER
          %val_4_tmp10 = arith.addf %val_4_tmp9, %x_val_4_mm_val_11 : TYPE_PLACEHOLDER
          %val_4_tmp11= arith.addf %val_4_tmp10, %x_val_4_mm_val_12: TYPE_PLACEHOLDER
          %val_4_tmp12 = arith.addf %val_4_tmp11, %x_val_4_mm_val_13 : TYPE_PLACEHOLDER
          %val_4_tmp13 = arith.addf %val_4_tmp12, %x_val_4_mm_val_14: TYPE_PLACEHOLDER
          %val_4_tmp14 = arith.addf %val_4_tmp13, %x_val_4_mm_val_15: TYPE_PLACEHOLDER
          %val_4 = arith.addf %y_val_4, %val_4_tmp14 : TYPE_PLACEHOLDER
          memref.store %val_4, %y[%i_4] : memref<?xTYPE_PLACEHOLDER>
        }
      }
    }

  return
}
