// BY: buddy-opt  %s -batchmatmul-optimize

#map = affine_map<(d0) -> (d0 mod 16)>
#map1 = affine_map<(d0) -> (d0 ceildiv 16)>
#map2 = affine_map<(d0) -> (d0)>
#map3 = affine_map<(d0) -> (d0 * 16)>
#set = affine_set<(d0)[s0] : (d0 * -16 + s0 - 16 >= 0)>
module {
  func.func @batch_matmul(%arg0: memref<?x?x?xf32>, %arg1: memref<?x?x?xf32>, %arg2: memref<?x?x?xf32>) {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = vector.splat %cst : vector<16xf32>
    %c0_0 = arith.constant 0 : index
    %dim = memref.dim %arg0, %c0_0 : memref<?x?x?xf32>
    %c1 = arith.constant 1 : index
    %dim_1 = memref.dim %arg0, %c1 : memref<?x?x?xf32>
    %c2 = arith.constant 2 : index
    %dim_2 = memref.dim %arg1, %c2 : memref<?x?x?xf32>
    %c1_3 = arith.constant 1 : index
    %dim_4 = memref.dim %arg1, %c1_3 : memref<?x?x?xf32>
    %1 = affine.apply #map(%dim_2)
    %2 = vector.create_mask %1 : vector<16xi1>
    %3 = affine.apply #map1(%dim_2)
    affine.parallel (%arg3) = (0) to (%dim) {
      affine.prefetch %arg0[%arg3, %dim_1, %dim_4], read, locality<3>, data : memref<?x?x?xf32>
      affine.for %arg4 = #map2(%c0) to #map2(%3) {
        affine.if #set(%arg3)[%dim_2] {
          affine.for %arg5 = #map2(%c0) to #map2(%dim_4) {
            %4 = affine.vector_load %arg1[%arg3, %arg5, %arg4 * 16] : memref<?x?x?xf32>, vector<16xf32>
            affine.for %arg6 = #map2(%c0) to #map2(%dim_1) {
              %5 = memref.load %arg0[%arg3, %arg6, %arg5] : memref<?x?x?xf32>
              %6 = vector.broadcast %5 : f32 to vector<16xf32>
              %7 = affine.vector_load %arg2[%arg3, %arg6, %arg4 * 16] : memref<?x?x?xf32>, vector<16xf32>
              %8 = vector.fma %6, %4, %7 : vector<16xf32>
              affine.vector_store %8, %arg2[%arg3, %arg6, %arg4 * 16] : memref<?x?x?xf32>, vector<16xf32>
            }
          }
        } else {
          affine.for %arg5 = #map2(%c0) to #map2(%dim_4) {
            %4 = affine.apply #map3(%arg4)
            %5 = vector.maskedload %arg1[%arg3, %arg5, %4], %2, %0 : memref<?x?x?xf32>, vector<16xi1>, vector<16xf32> into vector<16xf32>
            affine.for %arg6 = #map2(%c0) to #map2(%dim_1) {
              %6 = memref.load %arg0[%arg3, %arg6, %arg5] : memref<?x?x?xf32>
              %7 = vector.broadcast %6 : f32 to vector<16xf32>
              %8 = vector.maskedload %arg2[%arg3, %arg6, %4], %2, %0 : memref<?x?x?xf32>, vector<16xi1>, vector<16xf32> into vector<16xf32>
              %9 = vector.fma %7, %5, %8 : vector<16xf32>
              vector.maskedstore %arg2[%arg3, %arg6, %4], %2, %9 : memref<?x?x?xf32>, vector<16xi1>, vector<16xf32>
            }
          }
        }
      }
    }
    return
  }
}

