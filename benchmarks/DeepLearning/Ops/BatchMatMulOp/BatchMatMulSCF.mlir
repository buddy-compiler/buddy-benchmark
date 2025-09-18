// BY: buddy-opt  %s -batchmatmul-scf-optimize

#map = affine_map<(d0) -> (d0 mod 16)>
#map1 = affine_map<(d0) -> ((d0 floordiv 16) * 16)>
module {
  func.func @batch_matmul(%arg0: memref<?x?x?xf32>, %arg1: memref<?x?x?xf32>, %arg2: memref<?x?x?xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c0_0 = arith.constant 0 : index
    %dim = memref.dim %arg0, %c0_0 : memref<?x?x?xf32>
    %c1_1 = arith.constant 1 : index
    %dim_2 = memref.dim %arg0, %c1_1 : memref<?x?x?xf32>
    %c2 = arith.constant 2 : index
    %dim_3 = memref.dim %arg1, %c2 : memref<?x?x?xf32>
    %c1_4 = arith.constant 1 : index
    %dim_5 = memref.dim %arg1, %c1_4 : memref<?x?x?xf32>
    %0 = vector.splat %cst : vector<16xf32>
    %1 = affine.apply #map(%dim_3)
    %2 = vector.create_mask %1 : vector<16xi1>
    %3 = affine.apply #map1(%dim_3)
    scf.forall (%arg3) in (%dim) {
      scf.for %arg4 = %c0 to %dim_2 step %c1 {
        scf.for %arg5 = %c0 to %dim_5 step %c1 {
          %4 = memref.load %arg0[%arg3, %arg4, %arg5] : memref<?x?x?xf32>
          %5 = vector.broadcast %4 : f32 to vector<16xf32>
          scf.for %arg6 = %c0 to %3 step %c16 {
            %7 = vector.load %arg1[%arg3, %arg5, %arg6] : memref<?x?x?xf32>, vector<16xf32>
            %8 = vector.load %arg2[%arg3, %arg4, %arg6] : memref<?x?x?xf32>, vector<16xf32>
            %9 = vector.fma %5, %7, %8 : vector<16xf32>
            vector.store %9, %arg2[%arg3, %arg4, %arg6] : memref<?x?x?xf32>, vector<16xf32>
          }
          %6 = arith.cmpi sgt, %1, %c0 : index
          scf.if %6 {
            %7 = vector.maskedload %arg1[%arg3, %arg5, %3], %2, %0 : memref<?x?x?xf32>, vector<16xi1>, vector<16xf32> into vector<16xf32>
            %8 = vector.maskedload %arg2[%arg3, %arg4, %3], %2, %0 : memref<?x?x?xf32>, vector<16xi1>, vector<16xf32> into vector<16xf32>
            %9 = vector.fma %5, %7, %8 : vector<16xf32>
            vector.maskedstore %arg2[%arg3, %arg4, %3], %2, %9 : memref<?x?x?xf32>, vector<16xi1>, vector<16xf32>
          }
        }
      }
    }
    return
  }
}

