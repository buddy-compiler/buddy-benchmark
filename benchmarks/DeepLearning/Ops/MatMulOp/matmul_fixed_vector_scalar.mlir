func.func @matmul_fixed_vector_scalar(%a : memref<?x?xf32>, %b : memref<?x?xf32>, %c : memref<?x?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %step = arith.constant 16 : index
  %true_vl = arith.constant 8 : index

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
    // tail processing
    scf.for %k_idx = %k_true_tail_idx to %k step %c1 {
      %sum_init = arith.constant 0. : f32
      %sum_iter = affine.for %n_idx = 0 to %n 
          iter_args(%sum_vec = %sum_init) -> (f32) {
        %a_ele = memref.load %a[%m_idx, %n_idx] : memref<?x?xf32>
        %b_ele = memref.load %b[%n_idx, %k_idx] : memref<?x?xf32>
        %tmp_ele = arith.mulf %a_ele, %b_ele : f32
        %res_sum = arith.addf %tmp_ele, %sum_vec : f32
        affine.yield %res_sum : f32
      }
      memref.store %sum_iter, %c[%m_idx, %k_idx] : memref<?x?xf32>

    }
  }
  return
}
