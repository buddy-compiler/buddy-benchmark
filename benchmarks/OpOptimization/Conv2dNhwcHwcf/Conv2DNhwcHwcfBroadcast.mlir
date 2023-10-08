#map0 = affine_map<(d0, d1, d2, d3) -> (d2)>
#map1 = affine_map<(d0) -> (d0 ceildiv 32)>

func.func @transpose(%input: memref<?x?x?x?xf32>,
                     %output: memref<?x?x?x?xf32>) {

    return
}
   func.func @conv_2d_nhwc_hwcf(%input: memref<?x?x?x?xf32>,
                               %kernel: memref<?x?x?x?xf32>,
                               %output: memref<?x?x?x?xf32>) {
    %c0 = arith.constant 0 : index
    %c0_f32 = arith.constant 0.0 : f32
    %c0_f32_vec = vector.splat %c0_f32 : vector<32xf32>
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c32 = arith.constant 32 : index
    // Get the n size. (batch)
    %n = memref.dim %input, %c0 :  memref<?x?x?x?xf32>
    // Get the f size. (feature)
    %f = memref.dim %kernel, %c3 :  memref<?x?x?x?xf32>
    // Get the c size. (channel)
    %c = memref.dim %kernel, %c2 :  memref<?x?x?x?xf32>
    // Get the 2D output size. (row and column)
    %output_row = memref.dim %output, %c1 :  memref<?x?x?x?xf32>
    %output_col = memref.dim %output, %c2 :  memref<?x?x?x?xf32>
    // Get the 2D kernel size. (row and column)
    %kernel_row = memref.dim %kernel, %c0 :  memref<?x?x?x?xf32>
    %kernel_col = memref.dim %kernel, %c1 :  memref<?x?x?x?xf32>

    %input_row = memref.dim %input, %c1 :  memref<?x?x?x?xf32>
    %input_col = memref.dim %input, %c2 :  memref<?x?x?x?xf32>

    // %input_transpose = memref.alloc(%n, %c, %input_row, %output_col) : memref<?x?x?x?xf32>
    // %kernel_transpose = memref.alloc(%f, %c, %kernel_row, %kernel_col) : memref<?x?x?x?xf32>
    %output_transpose = memref.alloc(%n, %f, %output_row, %output_col) : memref<?x?x?x?xf32>

    // affine.for %n_idx = %c0 to %n {
    //   affine.for %input_row_idx = %c0 to %input_row {
    //     affine.for %input_col_idx = %c0 to %input_col {
    //       affine.for %c_idx = %c0 to %c {
    //         %val = memref.load %input[%n_idx, %input_row_idx, %input_col_idx, %c_idx] : memref<?x?x?x?xf32>
    //         memref.store %val, %input_transpose[%n_idx, %c_idx, %input_row_idx, %input_col_idx] : memref<?x?x?x?xf32>
    //       }
    //     }
    //   }
    // }

    // affine.for %f_idx = %c0 to %f {
    //   affine.for %input_row_idx = %c0 to %kernel_row {
    //     affine.for %input_col_idx = %c0 to %kernel_col {
    //       affine.for %c_idx = %c0 to %c {
    //         %val = memref.load %kernel[%input_row_idx, %input_col_idx, %c_idx, %f_idx] : memref<?x?x?x?xf32>
    //         memref.store %val, %kernel_transpose[%f_idx, %c_idx, %input_row_idx, %input_col_idx] : memref<?x?x?x?xf32>
    //       }
    //     }
    //   }
    // }

    // affine.for %n_idx = %c0 to %n {
    //   affine.for %f_idx = %c0 to %f {
    //     affine.for %c_idx = %c0 to %c {
    //       affine.for %output_row_idx = %c0 to %output_row {
    //         affine.for %kernel_row_idx = %c0 to %kernel_row {
    //           affine.for %kernel_col_idx = %c0 to %kernel_col {
    //             affine.for %output_col_idx = %c0 to #map1(%output_col) {
    //               // Check sparsity.
    //               %kernel_ele = memref.load %kernel_transpose[%f_idx, %c_idx, %kernel_row_idx, %kernel_col_idx] : memref<?x?x?x?xf32>
    //               %sparsity_flag = arith.cmpf one, %kernel_ele, %c0_f32 : f32
    //               scf.if %sparsity_flag {
    //                 // Check tail.
    //                 %kernel_vec = vector.broadcast %kernel_ele : f32 to vector<32xf32>
    //                 %output_col_cur = arith.muli %output_col_idx, %c32 : index
    //                 %tail_len = arith.subi %output_col, %output_col_cur : index
    //                 %tail_flag = arith.cmpi sge, %tail_len, %c32 : index
    //                 scf.if %tail_flag {
    //                   %input_vec = affine.vector_load %input_transpose[%n_idx, %c_idx, %output_row_idx + %kernel_row_idx, %kernel_col_idx + %output_col_idx * 32] : memref<?x?x?x?xf32>, vector<32xf32>
    //                   %output_vec = affine.vector_load %output_transpose[%n_idx, %f_idx, %output_row_idx, %output_col_idx * 32] : memref<?x?x?x?xf32>, vector<32xf32>
    //                   %result_vec = vector.fma %input_vec, %kernel_vec, %output_vec : vector<32xf32>
    //                   affine.vector_store %result_vec, %output_transpose[%n_idx, %f_idx, %output_row_idx, %output_col_idx * 32] : memref<?x?x?x?xf32>, vector<32xf32>
    //                 } else {
    //                   %mask_vec = vector.create_mask %tail_len : vector<32xi1>
    //                   %input_row_idx_tail = arith.addi %output_row_idx, %kernel_row_idx : index
    //                   %output_col_idx_tail = arith.muli %output_col_idx, %c32 : index
    //                   %input_col_idx_tail = arith.addi %kernel_col_idx, %output_col_idx_tail : index
    //                   %input_vec_tail = vector.maskedload %input_transpose[%n_idx, %c_idx, %input_row_idx_tail, %input_col_idx_tail], %mask_vec, %c0_f32_vec : memref<?x?x?x?xf32>, vector<32xi1>, vector<32xf32> into vector<32xf32>
    //                   %output_vec_tail = vector.maskedload %output_transpose[%n_idx, %f_idx, %output_row_idx, %output_col_idx_tail], %mask_vec, %c0_f32_vec : memref<?x?x?x?xf32>, vector<32xi1>, vector<32xf32> into vector<32xf32>
    //                   %result_vec_tail = vector.fma %input_vec_tail, %kernel_vec, %output_vec_tail : vector<32xf32>
    //                   vector.maskedstore %output_transpose[%n_idx, %f_idx, %output_row_idx, %output_col_idx_tail], %mask_vec, %result_vec_tail : memref<?x?x?x?xf32>, vector<32xi1>, vector<32xf32>
    //                 }
    //               }
    //             }
    //           }
    //         }
    //       }
    //     }
    //   }
    // }

    affine.for %n_idx = %c0 to %n {
      affine.for %output_row_idx = %c0 to %output_row {
        affine.for %kernel_row_idx = %c0 to %kernel_row {
          affine.for %kernel_col_idx = %c0 to %kernel_col {
            affine.for %output_col_idx = %c0 to #map1(%output_col) {
              affine.for %c_idx = %c0 to %c {
                affine.for %f_idx = %c0 to %f {
                  // Check sparsity.
                  %kernel_ele = memref.load %kernel[%kernel_row_idx, %kernel_col_idx, %c_idx, %f_idx] : memref<?x?x?x?xf32>
                  %sparsity_flag = arith.cmpf one, %kernel_ele, %c0_f32 : f32
                  scf.if %sparsity_flag {
                    // Check tail.
                    %kernel_vec = vector.broadcast %kernel_ele : f32 to vector<32xf32>
                    %output_col_cur = arith.muli %output_col_idx, %c32 : index
                    %tail_len = arith.subi %output_col, %output_col_cur : index
                    %tail_flag = arith.cmpi sge, %tail_len, %c32 : index
                    %input_row_idx_tail = arith.addi %output_row_idx, %kernel_row_idx : index
                    %output_col_idx_tail = arith.muli %output_col_idx, %c32 : index
                    %input_col_idx_tail = arith.addi %kernel_col_idx, %output_col_idx_tail : index
                    %mask_vec = vector.create_mask %tail_len : vector<32xi1>
                    %input_vec_tail = vector.transfer_read %input[%n_idx, %input_row_idx_tail, %input_col_idx_tail, %c_idx], %c0_f32, %mask_vec {permutation_map = #map0, in_bounds = [true]} : memref<?x?x?x?xf32>, vector<32xf32>
                    %output_vec_tail = vector.maskedload %output_transpose[%n_idx, %f_idx, %output_row_idx, %output_col_idx_tail], %mask_vec, %c0_f32_vec : memref<?x?x?x?xf32>, vector<32xi1>, vector<32xf32> into vector<32xf32>
                    %result_vec_tail = vector.fma %input_vec_tail, %kernel_vec, %output_vec_tail : vector<32xf32>
                    // vector.transfer_write %result_vec_tail, %output[%n_idx, %output_row_idx, %output_col_idx_tail, %f_idx], %mask_vec {permutation_map = #map0, in_bounds = [true]} : vector<32xf32>, memref<?x?x?x?xf32>
                    vector.maskedstore %output_transpose[%n_idx, %f_idx, %output_row_idx, %output_col_idx_tail], %mask_vec, %result_vec_tail : memref<?x?x?x?xf32>, vector<32xi1>, vector<32xf32>
                  }
                }
              }
            }
          }
        }
      }
    }

    // affine.for %n_idx = %c0 to %n {
    //   affine.for %output_row_idx = %c0 to %output_row {
    //         affine.for %output_col_idx = %c0 to %output_col {
    //             affine.for %f_idx = %c0 to #map1(%f) {
    //                 // Check tail.
    //                 %f_cur = arith.muli %f_idx, %c32 : index
    //                 %tail_len = arith.subi %f, %f_cur : index
    //                 %tail_flag = arith.cmpi sge, %tail_len, %c32 : index
    //                 %mask_vec = vector.create_mask %tail_len : vector<32xi1>
    //                 %input_vec_tail = vector.transfer_read %output_transpose[%n_idx, %f_cur, %output_row_idx, %output_col_idx], %c0_f32, %mask_vec {permutation_map = #map0, in_bounds = [true]} : memref<?x?x?x?xf32>, vector<32xf32>
    //                 // vector.transfer_write %result_vec_tail, %output[%n_idx, %output_row_idx, %output_col_idx_tail, %f_idx], %mask_vec {permutation_map = #map0, in_bounds = [true]} : vector<32xf32>, memref<?x?x?x?xf32>
    //                 vector.maskedstore %output[%n_idx, %output_row_idx, %output_col_idx, %f_idx], %mask_vec, %input_vec_tail {permutation_map = #map0, in_bounds = [true]} : memref<?x?x?x?xf32>, vector<32xi1>, vector<32xf32>
    //             }
    //         }
    //   }
    // }
    affine.for %n_idx = %c0 to %n {
      affine.for %output_row_idx = %c0 to %output_row {
            affine.for %output_col_idx = %c0 to %output_col {
                affine.for %f_idx = %c0 to %f {
                  %val = memref.load %output_transpose[%n_idx, %f_idx, %output_row_idx, %output_col_idx] : memref<?x?x?x?xf32>
                  memref.store %val, %output[%n_idx, %output_row_idx, %output_col_idx, %f_idx] : memref<?x?x?x?xf32>
                }
            }
      }
    }
    return
  }