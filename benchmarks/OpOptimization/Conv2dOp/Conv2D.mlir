#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> (d0 ceildiv 32)>
module{
func.func @conv_2d(%arg0: memref<?x?xi32>, %arg1: memref<?x?xi32>, %arg2: memref<?x?xi32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %cst = arith.constant 0 : i32
    %0 = vector.splat %cst : vector<32xi32>
    %dim = memref.dim %arg1, %c0 : memref<?x?xi32>
    %dim_0 = memref.dim %arg1, %c1 : memref<?x?xi32>
    %dim_1 = memref.dim %arg2, %c0 : memref<?x?xi32>
    %dim_2 = memref.dim %arg2, %c1 : memref<?x?xi32>
    affine.for %arg3 = #map(%c0) to #map(%dim_1) {
      affine.for %arg4 = #map(%c0) to #map(%dim) {
        affine.for %arg5 = #map(%c0) to #map(%dim_0) {
          affine.for %arg6 = #map(%c0) to #map1(%dim_2) {
            %1 = memref.load %arg1[%arg4, %arg5] : memref<?x?xi32>
            %2 = arith.index_cast %c0 : index to i32
            %4 = arith.cmpi sge, %1, %2 : i32
            scf.if %4 {
              %5 = vector.broadcast %1 : i32 to vector<32xi32>
              %6 = arith.muli %arg6, %c32 : index
              %7 = arith.subi %dim_2, %6 : index
              %8 = arith.cmpi sge, %7, %c32 : index
              scf.if %8 {
                %9 = affine.vector_load %arg0[%arg3 + %arg4, %arg5 + %arg6 * 32] : memref<?x?xi32>, vector<32xi32>
                %10 = affine.vector_load %arg2[%arg3, %arg6 * 32] : memref<?x?xi32>, vector<32xi32>
                %11 = arith.muli %9, %5 : vector<32xi32>
                %12 = arith.addi %10, %11 : vector<32xi32>
                affine.vector_store %12, %arg2[%arg3, %arg6 * 32] : memref<?x?xi32>, vector<32xi32>
              } else {
                %9 = vector.create_mask %7 : vector<32xi1>
                %10 = arith.addi %arg3, %arg4 : index
                %11 = arith.muli %arg6, %c32 : index
                %12 = arith.addi %arg5, %11 : index
                %13 = vector.maskedload %arg0[%10, %12], %9, %0 : memref<?x?xi32>, vector<32xi1>, vector<32xi32> into vector<32xi32>
                %14 = vector.maskedload %arg2[%arg3, %11], %9, %0 : memref<?x?xi32>, vector<32xi1>, vector<32xi32> into vector<32xi32>
                %15 = arith.muli %13, %5 : vector<32xi32>
                %16 = arith.addi %14, %15 : vector<32xi32>
                vector.maskedstore %arg2[%arg3, %11], %9, %16 : memref<?x?xi32>, vector<32xi1>, vector<32xi32>
              }
            }
          }
        }
      }
    }
    return
  }
}
