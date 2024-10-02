func.func @matmul_fixed_vector_mask(%a : memref<?x?xf32>, %b : memref<?x?xf32>, %c : memref<?x?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %step = arith.constant 16 : index
  %true_vl = arith.constant 8 : index
  %c0_f32_vec = arith.constant dense<0.> : vector<8xf32>

  // M
  %m = memref.dim %a, %c0 : memref<?x?xf32>
  // N
  %n = memref.dim %a, %c1 : memref<?x?xf32>
  // K
  %k = memref.dim %b, %c1 : memref<?x?xf32>

  %k_body_bound = arith.subi %k, %step : index
  %k_fake_bound = arith.subi %k, %true_vl : index

  // 0 -> M
  // body
  affine.for %m_idx = 0 to %m {
    %k_iter_idx = scf.for %k_idx = %c0 to %k_body_bound step %step
        iter_args(%k_iter_idx_init = %c0) -> (index) {
      %sum_init = arith.constant dense<0.> : vector<16xf32>
      %sum_iter_vec = affine.for %n_idx = 0 to %n
          iter_args(%sum_vec = %sum_init) -> (vector<16xf32>) {
        %a_ele = memref.load %a[%m_idx, %n_idx] : memref<?x?xf32>
        %a_vec = vector.broadcast %a_ele : f32 to vector<16xf32>
        %b_vec = vector.load %b[%n_idx, %k_idx] : memref<?x?xf32>, vector<16xf32>
        %res_sum_vec = vector.fma %a_vec, %b_vec, %sum_vec : vector<16xf32>
        affine.yield %res_sum_vec : vector<16xf32>
      }
      vector.store %sum_iter_vec, %c[%m_idx, %k_idx] : memref<?x?xf32>, vector<16xf32>
      scf.yield %k_idx : index
    }
    // fake tail
    %fake_tail_idx = arith.addi %k_iter_idx, %step : index
    %k_fake_iter_idx = scf.for %k_idx = %fake_tail_idx to %k_fake_bound step %true_vl
        iter_args(%k_iter_idx_init = %c0) -> (index) {
      %sum_init = arith.constant dense<0.> : vector<8xf32>
      %sum_iter_vec = affine.for %n_idx = 0 to %n
          iter_args(%sum_vec = %sum_init) -> (vector<8xf32>) {
        %a_ele = memref.load %a[%m_idx, %n_idx] : memref<?x?xf32>
        %a_vec = vector.broadcast %a_ele : f32 to vector<8xf32>
        %b_vec = vector.load %b[%n_idx, %k_idx] : memref<?x?xf32>, vector<8xf32>
        %res_sum_vec = vector.fma %a_vec, %b_vec, %sum_vec : vector<8xf32>
        affine.yield %res_sum_vec : vector<8xf32>
      }
      vector.store %sum_iter_vec, %c[%m_idx, %k_idx] : memref<?x?xf32>, vector<8xf32>
      scf.yield %k_idx : index
    }
    // ture tail
    // tail preprocessing
    %k_true_tail_idx = arith.addi %k_fake_iter_idx, %true_vl : index
    %k_true_tail_len = arith.subi %k, %k_true_tail_idx : index
    %mask_tail = vector.create_mask %k_true_tail_len : vector<8xi1>
    // tail processing
    %true_tail_sum_init = arith.constant dense<0.> : vector<8xf32>
    %true_tail_sum_iter_vec = affine.for %n_idx = 0 to %n 
        iter_args(%sum_vec = %true_tail_sum_init) -> (vector<8xf32>) {
      %a_ele = memref.load %a[%m_idx, %n_idx] : memref<?x?xf32>
      %a_vec = vector.broadcast %a_ele : f32 to vector<8xf32>
      %b_vec = vector.maskedload %b[%n_idx, %k_true_tail_idx], %mask_tail, %c0_f32_vec : memref<?x?xf32>, vector<8xi1>, vector<8xf32> into vector<8xf32>
      %res_sum_vec = vector.fma %a_vec, %b_vec, %sum_vec : vector<8xf32>
      affine.yield %res_sum_vec : vector<8xf32>
    }
    vector.maskedstore %c[%m_idx, %k_true_tail_idx], %mask_tail, %true_tail_sum_iter_vec : memref<?x?xf32>, vector<8xi1>, vector<8xf32>
  }
  return
}
