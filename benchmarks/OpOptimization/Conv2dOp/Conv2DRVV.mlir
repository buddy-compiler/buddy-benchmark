module{
    func.func @conv_2d(%arg0: memref<?x?xi32>, %arg1: memref<?x?xi32>, %arg2: memref<?x?xi32>) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %sew = arith.constant 2 : index
        %dim = memref.dim %arg1, %c0 : memref<?x?xi32>
        %dim_0 = memref.dim %arg1, %c1 : memref<?x?xi32>
        %dim_1 = memref.dim %arg2, %c0 : memref<?x?xi32>
        %dim_2 = memref.dim %arg2, %c1 : memref<?x?xi32>

        affine.for %tmp0 = %c0 to %dim_1 {
            %tmpAVL, %tmpIdx = scf.while (%avl = %dim_2, %idx = %c0) : (index, index) -> (index, index) {
                // If avl greater than zero.
                %cond = arith.cmpi sgt, %avl, %c0 : index
                // Pass avl, idx to the after region.
                scf.condition(%cond) %avl, %idx : index, index
            } do {
            ^bb0(%avl : index, %idx : index):
                %vl = rvv.setvl %avl, %sew, %c1 : index
                %vl_i32 = arith.index_cast %vl : index to i32
                %mask = vector.create_mask %vl : vector<[8]xi1>
                %c_vector = vector_exp.predication %mask, %vl_i32 : vector<[8]xi1>, i32 {
                    %ele = vector.load %arg2[%tmp0, %idx] : memref<?x?xi32>, vector<[8]xi32>
                    vector.yield %ele : vector<[8]xi32>
                } : vector<[8]xi32>
                %tmpvector = affine.for %tmp1 = %c0 to %dim iter_args(%vector_iter0 = %c_vector) -> (vector<[8]xi32>) {
                    %vector_next = affine.for %tmp2 = %c0 to %dim_0 iter_args(%vector_iter1 = %vector_iter0) -> (vector<[8]xi32>)  {
                        %0 = affine.load %arg1[%tmp1, %tmp2] : memref<?x?xi32>
                        %1 = arith.addi %tmp0, %tmp1 : index
                        %2 = arith.addi %idx, %tmp2 : index
                        %input_vector = vector_exp.predication %mask, %vl_i32 : vector<[8]xi1>, i32 {
                            %ele = vector.load %arg0[%1, %2] : memref<?x?xi32>, vector<[8]xi32>
                            vector.yield %ele : vector<[8]xi32>
                        } : vector<[8]xi32>
                
                        %3 = rvv.mul %input_vector, %0, %vl : vector<[8]xi32>, i32, index
                        %output = rvv.add %3, %vector_iter1, %vl : vector<[8]xi32>, vector<[8]xi32>, index
                        
                        affine.yield %output: vector<[8]xi32>
                    }
                    affine.yield %vector_next : vector<[8]xi32>
                }
            vector_exp.predication %mask, %vl_i32 : vector<[8]xi1>, i32 {
                vector.store %tmpvector, %arg2[%tmp0, %idx] : memref<?x?xi32>, vector<[8]xi32>
                vector.yield
            } : () -> ()

            // Update idx and avl.
            %new_idx = arith.addi %idx, %vl : index
            %new_avl = arith.subi %avl, %vl : index
            scf.yield %new_avl, %new_idx : index, index
            } 
        }
        return
    }
}
