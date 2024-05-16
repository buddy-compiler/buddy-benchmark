#map = affine_map<(d0) -> (d0 ceildiv 128)>
func.func @conv2d_nchw_fchw(%input: memref<?x?x?x?xf32>, %kernel: memref<?x?x?x?xf32>, %output: memref<?x?x?x?xf32>) {
  %c0 = arith.constant 0 : index
  %c0_f32 = arith.constant 0.0 : f32
  %c0_f32_vec = vector.splat %c0_f32 : vector<128xf32>
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %step = arith.constant 128 : index
  // Get the n size. (batch)
  %n = memref.dim %input, %c0 :  memref<?x?x?x?xf32>
  // Get the f size. (feature)
  %f = memref.dim %kernel, %c0 :  memref<?x?x?x?xf32>
  // Get the c size. (channel)
  %c = memref.dim %kernel, %c1 :  memref<?x?x?x?xf32>
  // Get the 2D output size. (row and column)
  %output_row = memref.dim %output, %c2 :  memref<?x?x?x?xf32>
  %output_col = memref.dim %output, %c3 :  memref<?x?x?x?xf32>
  // Get the 2D kernel size. (row and column)
  %kernel_row = memref.dim %kernel, %c2 :  memref<?x?x?x?xf32>
  %kernel_col = memref.dim %kernel, %c3 :  memref<?x?x?x?xf32>

  affine.for %n_idx = %c0 to %n {
    affine.for %f_idx = %c0 to %f {
      affine.for %c_idx = %c0 to %c {
        affine.for %output_row_idx = %c0 to %output_row {
          affine.for %kernel_row_idx = %c0 to %kernel_row {
            affine.for %kernel_col_idx = %c0 to %kernel_col {
              affine.for %output_col_idx = %c0 to #map(%output_col) {
                // Check sparsity.
                %kernel_ele = memref.load %kernel[%f_idx, %c_idx, %kernel_row_idx, %kernel_col_idx] : memref<?x?x?x?xf32>
                %sparsity_flag = arith.cmpf one, %kernel_ele, %c0_f32 : f32
                scf.if %sparsity_flag {
                  // Check tail.
                  %kernel_vec = vector.broadcast %kernel_ele : f32 to vector<128xf32>
                  %output_col_cur = arith.muli %output_col_idx, %step : index
                  %tail_len = arith.subi %output_col, %output_col_cur : index
                  %tail_flag = arith.cmpi sge, %tail_len, %step : index
                  scf.if %tail_flag {
                    %input_vec = affine.vector_load %input[%n_idx, %c_idx, %output_row_idx + %kernel_row_idx, %kernel_col_idx + %output_col_idx * 128] : memref<?x?x?x?xf32>, vector<128xf32>
                    %output_vec = affine.vector_load %output[%n_idx, %f_idx, %output_row_idx, %output_col_idx * 128] : memref<?x?x?x?xf32>, vector<128xf32>
                    %result_vec = vector.fma %input_vec, %kernel_vec, %output_vec : vector<128xf32>
                    affine.vector_store %result_vec, %output[%n_idx, %f_idx, %output_row_idx, %output_col_idx * 128] : memref<?x?x?x?xf32>, vector<128xf32>
                  } else {
                    %mask_vec = vector.create_mask %tail_len : vector<128xi1>
                    %input_row_idx_tail = arith.addi %output_row_idx, %kernel_row_idx : index
                    %output_col_idx_tail = arith.muli %output_col_idx, %step : index
                    %input_col_idx_tail = arith.addi %kernel_col_idx, %output_col_idx_tail : index
                    %input_vec_tail = vector.maskedload %input[%n_idx, %c_idx, %input_row_idx_tail, %input_col_idx_tail], %mask_vec, %c0_f32_vec : memref<?x?x?x?xf32>, vector<128xi1>, vector<128xf32> into vector<128xf32>
                    %output_vec_tail = vector.maskedload %output[%n_idx, %f_idx, %output_row_idx, %output_col_idx_tail], %mask_vec, %c0_f32_vec : memref<?x?x?x?xf32>, vector<128xi1>, vector<128xf32> into vector<128xf32>
                    %result_vec_tail = vector.fma %input_vec_tail, %kernel_vec, %output_vec_tail : vector<128xf32>
                    vector.maskedstore %output[%n_idx, %f_idx, %output_row_idx, %output_col_idx_tail], %mask_vec, %result_vec_tail : memref<?x?x?x?xf32>, vector<128xi1>, vector<128xf32>
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  return
}
