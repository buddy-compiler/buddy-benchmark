//===- MLIRFIRVectorization.mlir ------------------------------------------===//
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
// This file implements the vectorized MLIR FIR function (without tiling), 
// with a fixed vector size of 16.
//
//===----------------------------------------------------------------------===//

func.func @fir_vector_TYPE_PLACEHOLDER(%input : memref<?xTYPE_PLACEHOLDER>, 
    %kernel : memref<?xTYPE_PLACEHOLDER>, %output : memref<?xTYPE_PLACEHOLDER>) -> () {
  // 1. Get the total length of the workload.
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %input_size = memref.dim %input, %c0 : memref<?xTYPE_PLACEHOLDER>
  %kernel_size = memref.dim %kernel, %c0 : memref<?xTYPE_PLACEHOLDER>

  // 2. Set the iteration step (vector size).
  %vl_step = arith.constant 16 : index
  %vl_step_minus_1 = arith.subi %vl_step, %c1 : index

  // 3. Calculate the upper bound for vectorized processing
  // - Subtract `vl_step` is to avoid overflow at the vectorization tail.
  // - Add 1 to ensure the final loop runs when the workload length is divisible
  //   by the vector size.
  %upbound_ = arith.subi %input_size, %vl_step : index
  %upbound_init = arith.addi %upbound_, %c1 : index

  // 4. Loop through each kernel element
  scf.for %n = %c0 to %kernel_size step %c1 
    iter_args(%upbound = %upbound_init) -> (index) {
    %k_elem = memref.load %kernel[%n] : memref<?xTYPE_PLACEHOLDER>
    %k_vec = vector.splat %k_elem : vector<16xTYPE_PLACEHOLDER>

    // 5. Perform the vectorization body.
    %iter_idx = scf.for %i = %c0 to %upbound step %vl_step 
        iter_args(%iter_init = %c0) -> (index) {
      %in_vec = vector.load %input[%i] : memref<?xTYPE_PLACEHOLDER>, vector<16xTYPE_PLACEHOLDER>
      %out_index = arith.addi %i, %n : index
      %out_vec = vector.load %output[%out_index] : memref<?xTYPE_PLACEHOLDER>, vector<16xTYPE_PLACEHOLDER>
      %fma_vec = vector.fma %k_vec, %in_vec, %out_vec : vector<16xTYPE_PLACEHOLDER>
      vector.store %fma_vec, %output[%out_index] : memref<?xTYPE_PLACEHOLDER>, vector<16xTYPE_PLACEHOLDER>
      %i_next = arith.addi %i, %vl_step : index
      scf.yield %i_next : index
    }

    // 6. Process the remainder of the elements with scalar operations.
    %upbound_scalar = arith.addi %upbound, %vl_step_minus_1 : index
    scf.for %i = %iter_idx to %upbound_scalar step %c1 {
      %in_elem = memref.load %input[%i] : memref<?xTYPE_PLACEHOLDER>
      %out_index = arith.addi %i, %n : index
      %out_elem = memref.load %output[%out_index] : memref<?xTYPE_PLACEHOLDER>
      %mul_elem = arith.mulf %in_elem, %k_elem : TYPE_PLACEHOLDER
      %add_elem = arith.addf %mul_elem, %out_elem : TYPE_PLACEHOLDER
      memref.store %add_elem, %output[%out_index] : memref<?xTYPE_PLACEHOLDER>
    }

    %upbound_next = arith.subi %upbound, %c1 : index
    scf.yield %upbound_next : index
  }

  return
}
