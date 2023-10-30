//===- MLIRLinpackCEpslonF32.mlir -----------------------------------------===//
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
// This file provides the MLIR linpackc epslonf32 function.
//
//===----------------------------------------------------------------------===//

func.func @mlir_linpackcepslonf32(%x : f32) -> f32 {
     %f4 = arith.constant 4.0 : f32
     %f3 = arith.constant 3.0 : f32
     %a = arith.divf %f4, %f3 : f32
     %f0 = arith.constant 0.0 : f32
     %f1 = arith.constant 1.0 :f32
     %eps_0 = arith.constant 0.0 :f32
     %eps_final = scf.while (%eps = %eps_0) : (f32) -> f32 {
          %condition = arith.cmpf ueq, %eps, %f0 : f32
          scf.condition(%condition) %eps : f32
     } do {
          ^bb0(%eps: f32):
          %b = arith.subf %a , %f1 : f32
          %c_temp = arith.addf %b, %b :f32
          %c = arith.addf %c_temp, %b :f32
          //   eps = fabs((double)(c-ONE));
          %c_one = arith.subf %c, %f1 : f32
          %c_one_double = arith.extf %c_one : f32 to f64
          %eps_next_tmp = math.absf %c_one_double :f64
          %eps_next = arith.truncf %eps_next_tmp : f64 to f32
          scf.yield %eps_next : f32
     }
// return(eps*fabs((double)x));
     %x_double = arith.extf %x : f32 to f64
     %x_abs = math.absf %x_double : f64
     %eps_final_double = arith.extf %eps_final : f32 to f64
     %res_tmp = arith.mulf %eps_final_double, %x_abs : f64
     %res = arith.truncf %res_tmp : f64 to f32
     return %res : f32    
 }