func.func @conv_2d(%arg0: memref<10x10xi32>, %arg1: memref<3x3xi32>, %arg2: memref<8x8xi32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %sew = arith.constant 2 : index
    %dim = memref.dim %arg1, %c0 : memref<3x3xi32>
    %dim_0 = memref.dim %arg1, %c1 : memref<3x3xi32>
    %dim_1 = memref.dim %arg2, %c0 : memref<8x8xi32>
    %dim_2 = memref.dim %arg2, %c1 : memref<8x8xi32>

    affine.for %arg3 = #map(%c0) to #map(%dim_1) {
      affine.for %arg4 = #map(%c0) to #map(%dim) {
        affine.for %arg5 = #map(%c0) to #map(%dim_0){
            %1 = memref.load %arg1[%arg4, %arg5] : memref<3x3xi32>
            %tmpAVL, %tmpIdx = scf.while (%avl = %dim_0, %idx = %c0) : (index, index) -> (index, index) {
                // If avl greater than zero.
                %cond = arith.cmpi sgt, %avl, %c0 : index
                // Pass avl, idx to the after region.
                scf.condition(%cond) %avl, %idx : index, index
            } do {
            ^bb0(%avl : index, %idx : index):
                %vl = rvv.setvl %avl, %sew, %c1 : index
                %vl_i32 = arith.index_cast %vl : index to i32
                %mask = vector.create_mask %vl : vector<[8]xi1>
                %input_vector = vector_exp.predication %mask, %vl_i32 : vector<[8]xi1>, i32 {
                    %2 = arith.addi %arg3, %arg4 : index
                    %3 = arith.addi %arg5, %idx : index
                    %ele = vector.load %arg0[%2, %3] : memref<10x10xi32>, vector<[8]xi32>
                    vector.yield %ele : vector<[8]xi32>
                } : vector<[8]xi32>
                %c_vector = vector_exp.predication %mask, %vl_i32 : vector<[8]xi1>, i32 {
                    %ele = vector.load %arg2[%arg3, %idx] : memref<8x8xi32>, vector<[8]xi32>
                    vector.yield %ele : vector<[8]xi32>
                } : vector<[8]xi32>
                %4 = rvv.mul %input_vector, %1, %vl : vector<[8]xi32>, i32, index
                %output = rvv.add %4, %c_vector, %vl : vector<[8]xi32>, vector<[8]xi32>, index
                vector_exp.predication %mask, %vl_i32 : vector<[8]xi1>, i32 {
                    vector.store %output, %arg2[%arg3, %idx] : memref<8x8xi32>, vector<[8]xi32>
                    vector.yield
                } : () -> ()
                // Update idx and avl.
                %new_idx = arith.addi %idx, %vl : index
                %new_avl = arith.subi %avl, %vl : index
                scf.yield %new_avl, %new_idx : index, index
            }
        }
    } 
  }
  return
 }