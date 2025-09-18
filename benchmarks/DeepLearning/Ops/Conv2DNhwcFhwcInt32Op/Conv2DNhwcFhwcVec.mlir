// BY: buddy-opt  %s -conv-nhwc-fhwc-optimize

#map1 = affine_map<(d0, d1)[s0] -> (-d0 + s0, d1)>
#map2 = affine_map<(d0, d1) -> (d0 + d1)>
#map3 = affine_map<(d0) -> (d0)>
module {
  func.func @conv_2d_nhwc_fhwc(%arg0: memref<?x?x?x?xi32>, %arg1: memref<?x?x?x?xi32>, %arg2: memref<?x?x?x?xi32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c0_i32 = arith.constant 0 : i32
    %c0_0 = arith.constant 0 : index
    %dim = memref.dim %arg2, %c0_0 : memref<?x?x?x?xi32>
    %c1_1 = arith.constant 1 : index
    %dim_2 = memref.dim %arg2, %c1_1 : memref<?x?x?x?xi32>
    %c2 = arith.constant 2 : index
    %dim_3 = memref.dim %arg2, %c2 : memref<?x?x?x?xi32>
    %c3 = arith.constant 3 : index
    %dim_4 = memref.dim %arg2, %c3 : memref<?x?x?x?xi32>
    %c3_5 = arith.constant 3 : index
    %dim_6 = memref.dim %arg0, %c3_5 : memref<?x?x?x?xi32>
    %c1_7 = arith.constant 1 : index
    %dim_8 = memref.dim %arg1, %c1_7 : memref<?x?x?x?xi32>
    %c2_9 = arith.constant 2 : index
    %dim_10 = memref.dim %arg1, %c2_9 : memref<?x?x?x?xi32>
    scf.for %arg3 = %c0 to %dim step %c1 {
      scf.for %arg4 = %c0 to %dim_2 step %c1 {
        scf.for %arg5 = %c0 to %dim_3 step %c1 {
          scf.for %arg6 = %c0 to %dim_4 step %c1 {
            %1 = memref.load %arg2[%arg3, %arg4, %arg5, %arg6] : memref<?x?x?x?xi32>
            %2 = scf.for %arg7 = %c0 to %dim_6 step %c16 iter_args(%arg8 = %1) -> (i32) {
              %3 = vector.broadcast %c0_i32 : i32 to vector<16xi32>
              %4 = affine.min #map1(%arg7, %c16)[%dim_6]
              %5 = vector.create_mask %4 : vector<16xi1>
              %6 = scf.for %arg9 = %c0 to %dim_8 step %c1 iter_args(%arg10 = %3) -> (vector<16xi32>) {
                %9 = affine.apply #map2(%arg4, %arg9)
                %10 = scf.for %arg11 = %c0 to %dim_10 step %c1 iter_args(%arg12 = %arg10) -> (vector<16xi32>) {
                  %11 = affine.apply #map2(%arg5, %arg11)
                  %12 = affine.apply #map3(%arg11)
                  %13 = vector.load %arg0[%arg3, %9, %11, %arg7] : memref<?x?x?x?xi32>, vector<16xi32>
                  %14 = vector.load %arg1[%arg6, %arg9, %12, %arg7] : memref<?x?x?x?xi32>, vector<16xi32>
                  %15 = arith.muli %13, %14 : vector<16xi32>
                  %16 = arith.addi %15, %arg12 : vector<16xi32>
                  scf.yield %16 : vector<16xi32>
                }
                scf.yield %10 : vector<16xi32>
              }
              %7 = vector.mask %5 { vector.reduction <add>, %6 : vector<16xi32> into i32 } : vector<16xi1> -> i32
              %8 = arith.addi %arg8, %7 : i32
              scf.yield %8 : i32
            }
            memref.store %2, %arg2[%arg3, %arg4, %arg5, %arg6] : memref<?x?x?x?xi32>
          }
        }
      }
    }
    return
  }
}
