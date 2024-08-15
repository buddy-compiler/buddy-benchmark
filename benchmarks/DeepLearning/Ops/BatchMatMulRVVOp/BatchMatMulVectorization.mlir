#map = affine_map<(d0) -> (d0 mod 8)>
#map1 = affine_map<(d0) -> (d0 ceildiv 8)>
#map2 = affine_map<(d0) -> (d0)>
#map3 = affine_map<(d0) -> (d0 * 8)>
#set = affine_set<(d0)[s0] : (d0 * -8 + s0 - 8 >= 0)>
module {
  func.func @batch_matmul(%arg0: memref<?x?x?xi32>, %arg1: memref<?x?x?xi32>, %arg2: memref<?x?x?xi32>) {
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %0 = vector.broadcast %c0_i32 : i32 to vector<8xi32>
    %c0_0 = arith.constant 0 : index
    %dim = memref.dim %arg0, %c0_0 : memref<?x?x?xi32>
    %c1 = arith.constant 1 : index
    %dim_1 = memref.dim %arg0, %c1 : memref<?x?x?xi32>
    %c2 = arith.constant 2 : index
    %dim_2 = memref.dim %arg1, %c2 : memref<?x?x?xi32>
    %c1_3 = arith.constant 1 : index
    %dim_4 = memref.dim %arg1, %c1_3 : memref<?x?x?xi32>
    %1 = affine.apply #map(%dim_2)
    %2 = vector.create_mask %1 : vector<8xi1>
    %3 = affine.apply #map1(%dim_2)
    affine.parallel (%arg3) = (0) to (%dim) {
      affine.prefetch %arg0[%arg3, %dim_1, %dim_4], read, locality<3>, data : memref<?x?x?xi32>
      affine.for %arg4 = #map2(%c0) to #map2(%3) {
        affine.if #set(%arg3)[%dim_2] {
          affine.for %arg5 = #map2(%c0) to #map2(%dim_4) {
            %4 = affine.vector_load %arg1[%arg3, %arg5, %arg4 * 8] : memref<?x?x?xi32>, vector<8xi32>
            affine.for %arg6 = #map2(%c0) to #map2(%dim_1) {
              %5 = memref.load %arg0[%arg3, %arg6, %arg5] : memref<?x?x?xi32>
              %6 = vector.broadcast %5 : i32 to vector<8xi32>
              %7 = affine.vector_load %arg2[%arg3, %arg6, %arg4 * 8] : memref<?x?x?xi32>, vector<8xi32>
              %8 = arith.muli %6, %4 : vector<8xi32>
              %9 = arith.addi %8, %7 : vector<8xi32>
              affine.vector_store %9, %arg2[%arg3, %arg6, %arg4 * 8] : memref<?x?x?xi32>, vector<8xi32>
            }
          }
        } else {
          affine.for %arg5 = #map2(%c0) to #map2(%dim_4) {
            %4 = affine.apply #map3(%arg4)
            %5 = vector.maskedload %arg1[%arg3, %arg5, %4], %2, %0 : memref<?x?x?xi32>, vector<8xi1>, vector<8xi32> into vector<8xi32>
            affine.for %arg6 = #map2(%c0) to #map2(%dim_1) {
              %6 = memref.load %arg0[%arg3, %arg6, %arg5] : memref<?x?x?xi32>
              %7 = vector.broadcast %6 : i32 to vector<8xi32>
              %8 = vector.maskedload %arg2[%arg3, %arg6, %4], %2, %0 : memref<?x?x?xi32>, vector<8xi1>, vector<8xi32> into vector<8xi32>
              %9 = arith.muli %7, %5 : vector<8xi32>
              %10 = arith.addi %9, %8 : vector<8xi32>
              vector.maskedstore %arg2[%arg3, %arg6, %4], %2, %10 : memref<?x?x?xi32>, vector<8xi1>, vector<8xi32>
            }
          }
        }
      }
    }
    return
  }
}

