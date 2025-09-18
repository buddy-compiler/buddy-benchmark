// BY: buddy-opt  %s -conv-nhwc-fhwc-tile-optimize

#map = affine_map<(d0) -> (d0 ceildiv 8)>
#map1 = affine_map<(d0) -> (d0 ceildiv 4)>
#map2 = affine_map<(d0) -> (d0)>
#map3 = affine_map<(d0, d1)[s0] -> (d0 + d1, s0)>
#map4 = affine_map<(d0, d1)[s0] -> (-d0 + s0, d1)>
#map5 = affine_map<(d0, d1) -> (d0 + d1)>
module {

  func.func @conv_2d_nhwc_fhwc(%arg0: memref<?x?x?x?xf32>, %arg1: memref<?x?x?x?xf32>, %arg2: memref<?x?x?x?xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c0_0 = arith.constant 0 : index
    %dim = memref.dim %arg2, %c0_0 : memref<?x?x?x?xf32>
    %c1_1 = arith.constant 1 : index
    %dim_2 = memref.dim %arg2, %c1_1 : memref<?x?x?x?xf32>
    %c2 = arith.constant 2 : index
    %dim_3 = memref.dim %arg2, %c2 : memref<?x?x?x?xf32>
    %c3 = arith.constant 3 : index
    %dim_4 = memref.dim %arg2, %c3 : memref<?x?x?x?xf32>
    %c3_5 = arith.constant 3 : index
    %dim_6 = memref.dim %arg0, %c3_5 : memref<?x?x?x?xf32>
    %c1_7 = arith.constant 1 : index
    %dim_8 = memref.dim %arg1, %c1_7 : memref<?x?x?x?xf32>
    %c2_9 = arith.constant 2 : index
    %dim_10 = memref.dim %arg1, %c2_9 : memref<?x?x?x?xf32>
    %0 = affine.apply #map(%dim_2)
    %1 = affine.apply #map1(%dim_3)
    %2 = affine.apply #map2(%dim_4)
    scf.forall (%arg3, %arg4, %arg5, %arg6) = (%c0, %c0, %c0, %c0) to (%dim, %dim_2, %dim_3, %dim_4) step (%c1, %0, %1, %2) {
      %3 = affine.min #map3(%arg4, %0)[%dim_2]
      %4 = affine.min #map3(%arg5, %1)[%dim_3]
      %5 = affine.min #map3(%arg6, %2)[%dim_4]
      scf.forall (%arg7, %arg8, %arg9) = (%arg4, %arg5, %arg6) to (%3, %4, %5) step (%c1, %c1, %c1) {
        %6 = memref.load %arg2[%arg3, %arg7, %arg8, %arg9] : memref<?x?x?x?xf32>
        %7 = scf.for %arg10 = %c0 to %dim_6 step %c16 iter_args(%arg11 = %6) -> (f32) {
          %8 = vector.splat %cst : vector<16xf32>
          %9 = affine.min #map4(%arg10, %c16)[%dim_6]
          %10 = vector.create_mask %9 : vector<16xi1>
          %11 = scf.for %arg12 = %c0 to %dim_8 step %c1 iter_args(%arg13 = %8) -> (vector<16xf32>) {
            %14 = affine.apply #map5(%arg7, %arg12)
            %15 = scf.for %arg14 = %c0 to %dim_10 step %c1 iter_args(%arg15 = %arg13) -> (vector<16xf32>) {
              %16 = affine.apply #map5(%arg8, %arg14)
              %17 = affine.apply #map2(%arg14)
              %18 = vector.load %arg0[%arg3, %14, %16, %arg10] : memref<?x?x?x?xf32>, vector<16xf32>
              %19 = vector.load %arg1[%arg9, %arg12, %17, %arg10] : memref<?x?x?x?xf32>, vector<16xf32>
              %20 = vector.fma %18, %19, %arg15 : vector<16xf32>
              scf.yield %20 : vector<16xf32>
            }
            scf.yield %15 : vector<16xf32>
          }
          %12 = vector.mask %10 { vector.reduction <add>, %11 : vector<16xf32> into f32 } : vector<16xi1> -> f32
          %13 = arith.addf %arg11, %12 : f32
          scf.yield %13 : f32
        }
        memref.store %7, %arg2[%arg3, %arg7, %arg8, %arg9] : memref<?x?x?x?xf32>
      }
    }
    return
  }
}

