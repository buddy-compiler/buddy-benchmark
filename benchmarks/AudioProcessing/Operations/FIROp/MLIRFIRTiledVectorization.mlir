//===- MLIRFIRTiledVectorization.mlir -------------------------------------===//
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
// This file provides the vectorized MLIR FIR function with tiling.
//
//===----------------------------------------------------------------------===//

// Tail process for fir vectorization algorithm.
func.func @tail_processing(%input : memref<?xf32>, %kernel : memref<?xf32>, 
                           %output : memref<?xf32>, %input_offset : index) -> () {
  // 1. Get the total length of the workload.
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %input_size = memref.dim %input, %c0 : memref<?xf32>
  %kernel_size = memref.dim %kernel, %c0 : memref<?xf32>

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
    %k_elem = memref.load %kernel[%n] : memref<?xf32>
    %k_vec = vector.splat %k_elem : vector<16xf32>

    // 5. Perform the vectorization body.
    %iter_idx = scf.for %i = %input_offset to %upbound step %vl_step // 起始点从`0`改为`input_offset`
        iter_args(%iter_init = %input_offset) -> (index) {
      %in_vec = vector.load %input[%i] : memref<?xf32>, vector<16xf32>
      %out_index = arith.addi %i, %n : index
      %out_vec = vector.load %output[%out_index] : memref<?xf32>, vector<16xf32>  // 需要计算output的偏移量
      %fma_vec = vector.fma %k_vec, %in_vec, %out_vec : vector<16xf32>
      vector.store %fma_vec, %output[%out_index] : memref<?xf32>, vector<16xf32>
      %i_next = arith.addi %i, %vl_step : index
      scf.yield %i_next : index
    }

    // 6. Process the remainder of the elements with scalar operations.
    %upbound_scalar = arith.addi %upbound, %vl_step_minus_1 : index
    scf.for %i = %iter_idx to %upbound_scalar step %c1 {
      %in_elem = memref.load %input[%i] : memref<?xf32>
      %out_index = arith.addi %i, %n : index
      %out_elem = memref.load %output[%out_index] : memref<?xf32>  // ouput index need to change
      %mul_elem = arith.mulf %in_elem, %k_elem : f32
      %add_elem = arith.addf %mul_elem, %out_elem : f32
      memref.store %add_elem, %output[%out_index] : memref<?xf32>  // change output index
    }

    %upbound_next = arith.subi %upbound, %c1 : index
    scf.yield %upbound_next : index
  }

  return 
}

func.func @fir_tiled_vectorization(%input : memref<?xf32>, %kernel : memref<?xf32>, 
                                   %output : memref<?xf32>) -> () {
  // 1. Get the total length of the workload.
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %input_size = memref.dim %input, %c0 : memref<?xf32>
  %kernel_size = memref.dim %kernel, %c0 : memref<?xf32>

  // 2. Set the iteration step (vector size).
  %vl_step = arith.constant 16 : index
  %vl_step_minus_1 = arith.subi %vl_step, %c1 : index

  %tile_step = arith.constant 2048 : index

  // 3. Calculate the upper bound for vectorized processing.
  // The computation times for the last kernel elements(which is the shortest).
  %last_kernel_element_used_input_size_ = arith.subi %input_size, %kernel_size : index
  %last_kernel_element_used_input_size = arith.addi %last_kernel_element_used_input_size_, %c1 : index

  %input_upbound_ = arith.subi %last_kernel_element_used_input_size, %tile_step : index
  %input_upbound = arith.addi %input_upbound_, %c1 : index

  // 4. Do the tiling process, each tile can be fully computed with vector(remainder is zero)
  // Return the offset address for tail process.
  %input_offset = scf.for %address = %c0 to %input_upbound step %tile_step 
      iter_args(%offset = %c0) -> (index) {
    %upbound = arith.addi %address, %tile_step : index

    scf.for %n = %c0 to %kernel_size step %c1 {
      %k_elem = memref.load %kernel[%n] : memref<?xf32>
      %k_vec = vector.splat %k_elem : vector<16xf32>

      // 5. Perform the vectorization body. 
      scf.for %i = %address to %upbound step %vl_step {
        %in_vec = vector.load %input[%i] : memref<?xf32>, vector<16xf32>
        %out_index = arith.addi %i, %n : index
        %out_vec = vector.load %output[%out_index] : memref<?xf32>, vector<16xf32>  // 需要计算output的偏移量
        %fma_vec = vector.fma %k_vec, %in_vec, %out_vec : vector<16xf32>
        vector.store %fma_vec, %output[%out_index] : memref<?xf32>, vector<16xf32>
      }
    }

    scf.yield %upbound : index
  }

  // 6. Tail processing, begin from `input[input_offset]`
  call @tail_processing(%input, %kernel, %output, %input_offset) : (memref<?xf32>, memref<?xf32>, memref<?xf32>, index) -> ()

  return
}
