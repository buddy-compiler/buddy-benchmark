// BY: buddy-opt  %s -depthwise-conv-nhwc-hwc-optimize

#map = affine_map<(d0) -> ((d0 floordiv 16) * 16)>
#map1 = affine_map<(d0) -> (d0 mod 16)>
#map2 = affine_map<(d0, d1) -> (d0 + d1)>
#map3 = affine_map<(d0) -> (d0)>
module {
  func.func @depthwise_conv_2d_nhwc_hwc(%arg0: memref<?x?x?x?xf32>, %arg1: memref<?x?x?xf32>, %arg2: memref<?x?x?x?xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = vector.splat %cst : vector<16xf32>
    %c0_0 = arith.constant 0 : index
    %dim = memref.dim %arg2, %c0_0 : memref<?x?x?x?xf32>
    %c1_1 = arith.constant 1 : index
    %dim_2 = memref.dim %arg2, %c1_1 : memref<?x?x?x?xf32>
    %c2 = arith.constant 2 : index
    %dim_3 = memref.dim %arg2, %c2 : memref<?x?x?x?xf32>
    %c3 = arith.constant 3 : index
    %dim_4 = memref.dim %arg2, %c3 : memref<?x?x?x?xf32>
    %1 = affine.apply #map(%dim_4)
    %2 = affine.apply #map1(%dim_4)
    %3 = vector.create_mask %2 : vector<16xi1>
    %c0_5 = arith.constant 0 : index
    %dim_6 = memref.dim %arg1, %c0_5 : memref<?x?x?xf32>
    %c1_7 = arith.constant 1 : index
    %dim_8 = memref.dim %arg1, %c1_7 : memref<?x?x?xf32>
    scf.forall (%arg3, %arg4, %arg5) in (%dim, %dim_2, %dim_3) {
      scf.for %arg6 = %c0 to %1 step %c16 {
        %5 = vector.load %arg2[%arg3, %arg4, %arg5, %arg6] : memref<?x?x?x?xf32>, vector<16xf32>
        %6 = scf.for %arg7 = %c0 to %dim_6 step %c1 iter_args(%arg8 = %5) -> (vector<16xf32>) {
          %7 = affine.apply #map2(%arg4, %arg7)
          %8 = scf.for %arg9 = %c0 to %dim_8 step %c1 iter_args(%arg10 = %arg8) -> (vector<16xf32>) {
            %9 = affine.apply #map2(%arg5, %arg9)
            %10 = affine.apply #map3(%arg9)
            %11 = vector.load %arg0[%arg3, %7, %9, %arg6] : memref<?x?x?x?xf32>, vector<16xf32>
            %12 = vector.load %arg1[%arg7, %10, %arg6] : memref<?x?x?xf32>, vector<16xf32>
            %13 = vector.fma %11, %12, %arg10 : vector<16xf32>
            scf.yield %13 : vector<16xf32>
          }
          scf.yield %8 : vector<16xf32>
        }
        vector.store %6, %arg2[%arg3, %arg4, %arg5, %arg6] : memref<?x?x?x?xf32>, vector<16xf32>
      }
      %4 = arith.cmpi sgt, %2, %c0 : index
      scf.if %4 {
        %5 = vector.maskedload %arg2[%arg3, %arg4, %arg5, %1], %3, %0 : memref<?x?x?x?xf32>, vector<16xi1>, vector<16xf32> into vector<16xf32>
        %6 = scf.for %arg6 = %c0 to %dim_6 step %c1 iter_args(%arg7 = %5) -> (vector<16xf32>) {
          %7 = affine.apply #map2(%arg4, %arg6)
          %8 = scf.for %arg8 = %c0 to %dim_8 step %c1 iter_args(%arg9 = %arg7) -> (vector<16xf32>) {
            %9 = affine.apply #map2(%arg5, %arg8)
            %10 = affine.apply #map3(%arg8)
            %11 = vector.maskedload %arg0[%arg3, %7, %9, %1], %3, %0 : memref<?x?x?x?xf32>, vector<16xi1>, vector<16xf32> into vector<16xf32>
            %12 = vector.maskedload %arg1[%arg6, %10, %1], %3, %0 : memref<?x?x?xf32>, vector<16xi1>, vector<16xf32> into vector<16xf32>
            %13 = vector.fma %11, %12, %arg9 : vector<16xf32>
            scf.yield %13 : vector<16xf32>
          }
          scf.yield %8 : vector<16xf32>
        }
        vector.maskedstore %arg2[%arg3, %arg4, %arg5, %1], %3, %6 : memref<?x?x?x?xf32>, vector<16xi1>, vector<16xf32>
      }
    }
    return
  }
}
