//===- MLIRFIR.mlir -------------------------------------------------------===//
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
// This file implements the scalar version of the Fir function, following the 
// same algorithm as Buddy's scalar version DAP pass: `--lower-dap`.
//
//===----------------------------------------------------------------------===//

func.func @fir_scalar(%input : memref<?xf32>, %kernel : memref<?xf32>, 
                    %output : memref<?xf32>) -> () {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %f0 = arith.constant 0.0 : f32
  %kernel_size = memref.dim %kernel, %c0 : memref<?xf32>
  %data_size = memref.dim %output, %c0 : memref<?xf32>

  // Populate the FIR pipeline by padding the %input with [%kernel_size-1] zeros 
  // at the beginning. Compute only the padding section of the input data.
  %fill_in_length = arith.subi %kernel_size, %c1 : index
  scf.for %iv_n = %c0 to %fill_in_length step %c1 {
    %upper_bound = arith.addi %iv_n, %c1 : index
    %out_final = scf.for %iv_k = %c0 to %upper_bound step %c1 
        iter_args(%out = %f0) -> (f32) {
      %i = arith.subi %iv_n, %iv_k : index
      %in = memref.load %input[%i] : memref<?xf32>
      %k = memref.load %kernel[%iv_k] : memref<?xf32>
      %mul = arith.mulf %in, %k : f32
      %out_next = arith.addf %out, %mul : f32
      scf.yield %out_next : f32
    }
    memref.store %out_final, %output[%iv_n] : memref<?xf32>
  }

  // Compute the input data following the padding section.
  scf.for %iv_n = %fill_in_length to %data_size step %c1 {
    %out_final = scf.for %iv_k = %c0 to %kernel_size step %c1 
        iter_args(%out = %f0) -> (f32) {
      %i = arith.subi %iv_n, %iv_k : index
      %in = memref.load %input[%i] : memref<?xf32>
      %k = memref.load %kernel[%iv_k] : memref<?xf32>
      %mul = arith.mulf %in, %k : f32
      %out_next = arith.addf %out, %mul : f32
      scf.yield %out_next : f32
    }
    memref.store %out_final, %output[%iv_n] : memref<?xf32>
  }
  return
}
