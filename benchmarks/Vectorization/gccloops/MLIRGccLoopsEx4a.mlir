//===- MLIRGccLoopsEx4a.mlir ----------------------------------------------------===//
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
// This file provides the MLIR GccLoopsEx4a function.
//
//===----------------------------------------------------------------------===//


func.func @mlir_gccloopsex4a(%N: index, %P: memref<12xi32> {llvm.alignment = 16}, %Q: memref<12xi32> {llvm.alignment = 16}){
  %c0 =  arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c5 = arith.constant 5 : i32

  %n_final = scf.while (%n = %N) : (index) -> (index) {
    %next_n = arith.subi %n, %c1 : index
    %q_value = memref.load %Q[%n] : memref<12xi32>
    %q_value_plus_5 = arith.addi %c5, %q_value : i32
    memref.store %q_value_plus_5, %P[%n] : memref<12xi32>
    %condition = arith.cmpi ugt, %n, %c0 : index
    scf.condition(%condition) %next_n : index
  } do {
  ^bb0(%n: index):
    scf.yield %n : index
  }
  return
}



