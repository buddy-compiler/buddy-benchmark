//===- MLIRLinpackCEpslonF64.mlir -----------------------------------------===//
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
// This file provides the MLIR linpackc epslonf64 function.
//
//===----------------------------------------------------------------------===//
func.func @mlir_linpackcepslonf64(%x : f64) -> f64 {
  %f4 = arith.constant 4.0 : f64
  %f3 = arith.constant 3.0 : f64
  %a = arith.divf %f4, %f3 : f64
  %f0 = arith.constant 0.0 : f64
  %f1 = arith.constant 1.0 :f64
  %eps_0 = arith.constant 0.0 :f64
  %eps_final = scf.while (%eps = %eps_0) : (f64) -> f64 {
  %condition = arith.cmpf ueq, %eps, %f0 : f64
    scf.condition(%condition) %eps : f64
  } do {
    ^bb0(%eps: f64):
    %b = arith.subf %a , %f1 : f64
    %c_temp = arith.addf %b, %b :f64
    %c = arith.addf %c_temp, %b :f64
    //   eps = fabs((double)(c-ONE));
    %c_one = arith.subf %c, %f1 : f64
    %eps_next = math.absf %c_one :f64
    scf.yield %eps_next : f64
  }
  // return(eps*fabs((double)x));
  %x_abs = math.absf %x : f64
  %res = arith.mulf %eps_final, %x_abs : f64
  return %res : f64    
}
