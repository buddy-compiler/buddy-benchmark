#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1, d2, d3) -> (0)>
#map2 = affine_map<(d0, d1) -> (d0 + d1)>
#map3 = affine_map<(d0) -> (d0 - d0 mod 2)>
#map4 = affine_map<(d0) -> (d0 - d0 mod 32)>
module {
  func.func @conv2d(%arg0: memref<?x?x?x?xf32>, %arg1: memref<?x?x?x?xf32>, %arg2: memref<?x?x?x?xf32>) {
    %c0 = arith.constant 0 : index
    %c0_0 = arith.constant 0 : index
    %0 = memref.dim %arg0, %c0_0 : memref<?x?x?x?xf32>
    %c1 = arith.constant 1 : index
    %1 = memref.dim %arg0, %c1 : memref<?x?x?x?xf32>
    %c3 = arith.constant 3 : index
    %2 = memref.dim %arg0, %c3 : memref<?x?x?x?xf32>
    %c2 = arith.constant 2 : index
    %3 = memref.dim %arg0, %c2 : memref<?x?x?x?xf32>
    %c1_1 = arith.constant 1 : index
    %4 = memref.dim %arg0, %c1_1 : memref<?x?x?x?xf32>
    %c2_2 = arith.constant 2 : index
    %5 = memref.dim %arg1, %c2_2 : memref<?x?x?x?xf32>
    %c3_3 = arith.constant 3 : index
    %6 = memref.dim %arg1, %c3_3 : memref<?x?x?x?xf32>
    %7 = memref.alloc() : memref<1xvector<16xf32>>
    affine.for %arg3 = #map0(%c0) to #map0(%0) {
      affine.for %arg4 = #map0(%c0) to #map0(%1) {
        affine.for %arg5 = #map0(%c0) to #map0(%2) {
          affine.for %arg6 = #map0(%c0) to #map0(%3) {
            %cst = arith.constant 0.000000e+00 : f32
            %8 = vector.transfer_read %arg0[%arg3, %arg4, %arg6, %arg5], %cst {permutation_map = #map1} : memref<?x?x?x?xf32>, vector<16xf32>
            memref.store %8, %7[%c0] : memref<1xvector<16xf32>>
            affine.for %arg7 = #map0(%c0) to #map0(%4) {
              affine.for %arg8 = #map0(%c0) to #map0(%5) {
                affine.for %arg9 = #map0(%c0) to #map0(%6) step 16 {
                  %11 = affine.apply #map2(%arg6, %arg8)
                  %12 = affine.apply #map0(%arg8)
                  %13 = affine.apply #map2(%arg5, %arg9)
                  %14 = affine.apply #map0(%arg9)
                  %cst_4 = arith.constant 0.000000e+00 : f32
                  %15 = vector.transfer_read %arg0[%arg3, %arg7, %11, %13], %cst_4 : memref<?x?x?x?xf32>, vector<16xf32>
                  %cst_5 = arith.constant 0.000000e+00 : f32
                  %16 = vector.transfer_read %arg1[%arg4, %arg7, %12, %14], %cst_5 : memref<?x?x?x?xf32>, vector<16xf32>
                  %17 = memref.load %7[%c0] : memref<1xvector<16xf32>>
                  %18 = vector.fma %15, %16, %17 : vector<16xf32>
                  memref.store %18, %7[%c0] : memref<1xvector<16xf32>>
                }
              }
            }
            %9 = memref.load %7[%c0] : memref<1xvector<16xf32>>
            %10 = vector.reduction <add>, %9 : vector<16xf32> into f32
            memref.store %10, %arg2[%arg3, %arg4, %arg6, %arg5] : memref<?x?x?x?xf32>
            affine.for %arg7 = #map0(%c0) to #map0(%4) {
              %11 = affine.apply #map3(%5)
              affine.for %arg8 = #map0(%11) to #map0(%5) {
                %12 = affine.apply #map4(%6)
                affine.for %arg9 = #map0(%12) to #map0(%6) {
                  %13 = affine.apply #map2(%arg6, %arg8)
                  %14 = affine.apply #map2(%arg5, %arg9)
                  %15 = memref.load %arg0[%arg3, %arg7, %13, %14] : memref<?x?x?x?xf32>
                  %16 = memref.load %arg1[%arg4, %arg7, %arg8, %arg9] : memref<?x?x?x?xf32>
                  %17 = memref.load %arg2[%arg3, %arg4, %arg6, %arg5] : memref<?x?x?x?xf32>
                  %18 = arith.mulf %15, %16 : f32
                  %19 = arith.addf %18, %17 : f32
                  memref.store %19, %arg2[%arg3, %arg4, %arg6, %arg5] : memref<?x?x?x?xf32>
                }
              }
            }
          }
        }
      }
    }
    memref.dealloc %7 : memref<1xvector<16xf32>>
    return
  }
}

