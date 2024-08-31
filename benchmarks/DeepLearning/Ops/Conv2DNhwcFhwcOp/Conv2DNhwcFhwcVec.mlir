#map = affine_map<(d0, d1)[s0] -> (-d0 + s0, d1)>
#map1 = affine_map<(d0, d1) -> (d0 + d1)>
#map2 = affine_map<(d0) -> (d0)>
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
    scf.for %arg3 = %c0 to %dim step %c1 {
      scf.for %arg4 = %c0 to %dim_2 step %c1 {
        scf.for %arg5 = %c0 to %dim_3 step %c1 {
          scf.for %arg6 = %c0 to %dim_4 step %c1 {
            %0 = memref.load %arg2[%arg3, %arg4, %arg5, %arg6] : memref<?x?x?x?xf32>
            %1 = scf.for %arg7 = %c0 to %dim_6 step %c16 iter_args(%arg8 = %0) -> (f32) {
              %2 = vector.splat %cst : vector<16xf32>
              %3 = affine.min #map(%arg7, %c16)[%dim_6]
              %4 = vector.create_mask %3 : vector<16xi1>
              %5 = scf.for %arg9 = %c0 to %dim_8 step %c1 iter_args(%arg10 = %2) -> (vector<16xf32>) {
                %8 = affine.apply #map1(%arg4, %arg9)
                %9 = scf.for %arg11 = %c0 to %dim_10 step %c1 iter_args(%arg12 = %arg10) -> (vector<16xf32>) {
                  %10 = affine.apply #map1(%arg5, %arg11)
                  %11 = affine.apply #map2(%arg11)
                  %12 = vector.load %arg0[%arg3, %8, %10, %arg7] : memref<?x?x?x?xf32>, vector<16xf32>
                  %13 = vector.load %arg1[%arg6, %arg9, %11, %arg7] : memref<?x?x?x?xf32>, vector<16xf32>
                  %14 = vector.fma %12, %13, %arg12 : vector<16xf32>
                  scf.yield %14 : vector<16xf32>
                }
                scf.yield %9 : vector<16xf32>
              }
              %6 = vector.mask %4 { vector.reduction <add>, %5 : vector<16xf32> into f32 } : vector<16xi1> -> f32
              %7 = arith.addf %arg8, %6 : f32
              scf.yield %7 : f32
            }
            memref.store %1, %arg2[%arg3, %arg4, %arg5, %arg6] : memref<?x?x?x?xf32>
          }
        }
      }
    }
    return
  }
}
