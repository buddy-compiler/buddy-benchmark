//===- MLIRIir.mlir -------------------------------------------------------===//

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
// This file provides the MLIR IIR function.
//
//===----------------------------------------------------------------------===//

func.func @mlir_iir(%in : memref<?xf64>, %filter : memref<?x?xf64>, %out : memref<?xf64>){
  %c0 = arith.constant 0 : index
  %N = memref.dim %in, %c0 : memref<?xf64>
  %M = memref.dim %filter, %c0: memref<?x?xf64>

  affine.for %j = 0 to %M iter_args(%inpt = %in) -> (memref<?xf64>){
    %b0 = affine.load %filter[%j, 0] : memref<?x?xf64>
    %b1 = affine.load %filter[%j, 1] : memref<?x?xf64>
    %b2 = affine.load %filter[%j, 2] : memref<?x?xf64>
    %a1 = affine.load %filter[%j, 4] : memref<?x?xf64>
    %a2 = affine.load %filter[%j, 5] : memref<?x?xf64>
    %init_z1 = arith.constant 0.0 : f64
    %init_z2 = arith.constant 0.0 : f64
    %res:2 = affine.for %i = 0 to %N iter_args(%z1 = %init_z1, %z2 = %init_z2) -> (f64, f64) {
        %input = affine.load %inpt[%i] : memref<?xf64>
        %t0 = arith.mulf %b0, %input : f64
        %output = arith.addf %t0, %z1 : f64

        %t1 = arith.mulf %b1, %input : f64
        %t2 = arith.mulf %a1, %output : f64
        %t3 = arith.subf %t1, %t2 : f64
        %z1_next = arith.addf %z2, %t3 : f64

        %t4 = arith.mulf %b2, %input : f64
        %t5 = arith.mulf %a2, %output : f64
        %z2_next = arith.subf %t4, %t5 : f64
        
        affine.store %output, %out[%i] : memref<?xf64>
        affine.yield %z1_next, %z2_next : f64, f64
    }
    affine.yield %out : memref<?xf64>
  }
  return
}
