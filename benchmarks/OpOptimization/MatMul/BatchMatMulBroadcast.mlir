// The MLIR prototype of batchmatmul-optimize in buddy-opt.

#map = affine_map<(d0) -> (d0 ceildiv STEP_PLACEHOLDER)>
func.func @batch_matmul_broadcast_STEP_PLACEHOLDER(%a : memref<?x?x?xf32>, %b : memref<?x?x?xf32>, %c : memref<?x?x?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %step = arith.constant STEP_PLACEHOLDER : index
  %c0_f32 = arith.constant 0.0 : f32
  %c0_f32_vec = vector.splat %c0_f32 : vector<STEP_PLACEHOLDERxf32>

  %a_row = memref.dim %a, %c1 : memref<?x?x?xf32>
  %a_col = memref.dim %a, %c2 : memref<?x?x?xf32>
  %b_row = memref.dim %b, %c1 : memref<?x?x?xf32>
  %b_col = memref.dim %b, %c2 : memref<?x?x?xf32>
  %batch = memref.dim %a, %c0 : memref<?x?x?xf32>

  affine.parallel (%batch_idx) = (0) to (%batch){ // Affine.parallel can be lowered to the omp dialect, which enables batch-level parallelization.
    affine.prefetch %a[%batch_idx, %a_row, %a_col], read, locality<3>, data : memref<?x?x?xf32> // Explicitly prefetch, about 5% faster on X86.
    affine.for %b_row_idx = 0 to %b_row {
        affine.for %a_row_idx = 0 to %a_row {
            affine.for %b_col_idx = 0 to #map(%b_col) {
                %a_ele = affine.load %a[%batch_idx, %a_row_idx, %b_row_idx] : memref<?x?x?xf32>
                %a_vec = vector.broadcast %a_ele : f32 to vector<STEP_PLACEHOLDERxf32>
                // Check tail.
                %b_col_cur = arith.muli %b_col_idx, %step : index
                %tail_len = arith.subi %b_col, %b_col_cur : index
                %tail_flag = arith.cmpi sge, %tail_len, %step : index
                scf.if %tail_flag {
                    %b_vec = affine.vector_load %b[%batch_idx, %b_row_idx, %b_col_idx * STEP_PLACEHOLDER] : memref<?x?x?xf32>, vector<STEP_PLACEHOLDERxf32>
                    %c_vec = affine.vector_load %c[%batch_idx, %a_row_idx, %b_col_idx * STEP_PLACEHOLDER] : memref<?x?x?xf32>, vector<STEP_PLACEHOLDERxf32>
                    %result_vec = vector.fma %a_vec, %b_vec, %c_vec : vector<STEP_PLACEHOLDERxf32>
                    affine.vector_store %result_vec, %c[%batch_idx, %a_row_idx, %b_col_idx * STEP_PLACEHOLDER] : memref<?x?x?xf32>, vector<STEP_PLACEHOLDERxf32>
                } else {
                    %mask_vec = vector.create_mask %tail_len : vector<STEP_PLACEHOLDERxi1>
                    %b_col_idx_tail = arith.muli %b_col_idx, %step : index
                    %b_vec_tail = vector.maskedload %b[%batch_idx, %b_row_idx, %b_col_idx_tail], %mask_vec, %c0_f32_vec : memref<?x?x?xf32>, vector<STEP_PLACEHOLDERxi1>, vector<STEP_PLACEHOLDERxf32> into vector<STEP_PLACEHOLDERxf32>
                    %c_vec_tail = vector.maskedload %c[%batch_idx, %a_row_idx, %b_col_idx_tail], %mask_vec, %c0_f32_vec : memref<?x?x?xf32>, vector<STEP_PLACEHOLDERxi1>, vector<STEP_PLACEHOLDERxf32> into vector<STEP_PLACEHOLDERxf32>
                    %result_vec_tail = vector.fma %a_vec, %b_vec_tail, %c_vec_tail : vector<STEP_PLACEHOLDERxf32>
                    vector.maskedstore %c[%batch_idx, %a_row_idx, %b_col_idx_tail], %mask_vec, %result_vec_tail : memref<?x?x?xf32>, vector<STEP_PLACEHOLDERxi1>, vector<STEP_PLACEHOLDERxf32>
                }
            }
        }
    }
  }
  return
}
