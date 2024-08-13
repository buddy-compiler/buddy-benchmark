#map = affine_map<(d0) -> (d0 mod 4)>
#map1 = affine_map<(d0) -> (d0 ceildiv 4)>
#map2 = affine_map<(d0) -> (d0)>
#map3 = affine_map<(d0) -> (d0 * 4)>
#set = affine_set<(d0)[s0] : (d0 * -4 + s0 - 4 >= 0)>
module {
  func.func @batch_matmul(%A: memref<?x?x?xi32>, %B: memref<?x?x?xi32>, %C: memref<?x?x?xi32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %i0 = arith.constant 0 : i32

    %0 = vector.broadcast %i0 : i32 to vector<4xi32>
    %dim = memref.dim %A, %c0 : memref<?x?x?xi32>
    %dim_1 = memref.dim %A, %c1 : memref<?x?x?xi32>
    %dim_2 = memref.dim %B, %c2 : memref<?x?x?xi32>
    %dim_4 = memref.dim %B, %c1 : memref<?x?x?xi32>

    %1 = affine.apply #map(%dim_2)
    %2 = vector.create_mask %1 : vector<4xi1>
    %3 = affine.apply #map1(%dim_2)
    affine.parallel (%arg3) = (0) to (%dim) {
      affine.prefetch %A[%arg3, %dim_1, %dim_4], read, locality<3>, data : memref<?x?x?xi32>
      affine.for %arg4 = #map2(%c0) to #map2(%3) {
        affine.if #set(%arg3)[%dim_2] {
          affine.for %arg5 = #map2(%c0) to #map2(%dim_4) {
            %4 = affine.vector_load %B[%arg3, %arg5, %arg4 * 4] : memref<?x?x?xi32>, vector<4xi32>
            affine.for %arg6 = #map2(%c0) to #map2(%dim_1) {
              %5 = memref.load %A[%arg3, %arg6, %arg5] : memref<?x?x?xi32>
              %6 = vector.broadcast %5 : i32 to vector<4xi32>
              %7 = affine.vector_load %C[%arg3, %arg6, %arg4 * 4] : memref<?x?x?xi32>, vector<4xi32>
              %tmp8=arith.muli %6, %4 : vector<4xi32>
              %8 =arith.addi %tmp8,%7 : vector<4xi32>
              affine.vector_store %8, %C[%arg3, %arg6, %arg4 * 4] : memref<?x?x?xi32>, vector<4xi32>
            }
          }
        } else {
          affine.for %arg5 = #map2(%c0) to #map2(%dim_4) {
            %4 = affine.apply #map3(%arg4)
            %5 = vector.maskedload %B[%arg3, %arg5, %4], %2, %0 : memref<?x?x?xi32>, vector<4xi1>, vector<4xi32> into vector<4xi32>
            affine.for %arg6 = #map2(%c0) to #map2(%dim_1) {
              %6 = memref.load %A[%arg3, %arg6, %arg5] : memref<?x?x?xi32>
              %7 = vector.broadcast %6 : i32 to vector<4xi32>
              %8 = vector.maskedload %C[%arg3, %arg6, %4], %2, %0 : memref<?x?x?xi32>, vector<4xi1>, vector<4xi32> into vector<4xi32>
              %tmp9 = arith.muli %7,%5: vector<4xi32>
              %9 =arith.addi %tmp9,%8 : vector<4xi32>
              vector.maskedstore %C[%arg3, %arg6, %4], %2, %9 : memref<?x?x?xi32>, vector<4xi1>, vector<4xi32>
            }
          }
        }
      }
    }
    return
  }
}

